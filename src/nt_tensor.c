/**
 * @file nt_tensor.c
 * @brief NTTESHGNN - Core tensor implementation
 */

#include "ntteshgnn/nt_tensor.h"
#include "ntteshgnn/nt_context.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void compute_strides(int32_t* nb, const int32_t* ne, uint8_t ndim, nt_dtype_t dtype) {
    if (ndim == 0) return;
    
    size_t elem_size = nt_dtype_size(dtype);
    nb[0] = (int32_t)elem_size;
    
    for (uint8_t i = 1; i < ndim; i++) {
        nb[i] = nb[i-1] * ne[i-1];
    }
}

static bool check_contiguous(const int32_t* ne, const int32_t* nb, uint8_t ndim, nt_dtype_t dtype) {
    if (ndim == 0) return true;
    
    size_t elem_size = nt_dtype_size(dtype);
    if (nb[0] != (int32_t)elem_size) return false;
    
    for (uint8_t i = 1; i < ndim; i++) {
        if (nb[i] != nb[i-1] * ne[i-1]) return false;
    }
    
    return true;
}

/* A byte-strided scalar view must fit entirely in its backing storage.
 * Block-quantized and sub-byte dtypes require block-aware addressing, not a
 * scalar memcpy; fail closed until those codecs have their own copy path. */
static bool valid_scalar_span(const nt_storage_t* storage, size_t offset,
                              nt_dtype_t dtype, uint8_t ndim,
                              const int32_t* ne, const int32_t* nb) {
    const size_t elem = nt_dtype_size(dtype);
    if (!storage || !storage->data || !ne || !nb || !elem ||
        !ndim || ndim > NT_MAX_DIMS || offset > storage->size_bytes ||
        nt_dtype_is_quantized(dtype) || nt_dtype_bits(dtype) < 8) return false;

    size_t max_delta = 0;
    size_t count = 1;
    for (uint8_t i = 0; i < ndim; ++i) {
        if (ne[i] <= 0 || nb[i] < 0) return false;
        if ((size_t)ne[i] > SIZE_MAX / count) return false;
        count *= (size_t)ne[i];
        const size_t n = (size_t)(ne[i] - 1);
        const size_t step = (size_t)nb[i];
        if (step && n > (SIZE_MAX - max_delta) / step) return false;
        max_delta += n * step;
    }
    return count <= (size_t)INT64_MAX / elem &&
           elem <= storage->size_bytes - offset &&
           max_delta <= storage->size_bytes - offset - elem;
}

/* Validate active int32 byte strides before compute_strides multiplies them.
 * Tensor ne/nb APIs cannot express a byte stride larger than INT32_MAX. */
static bool valid_shape(nt_dtype_t dtype, uint8_t ndim, const int32_t* shape) {
    if (!shape || !ndim || ndim > NT_MAX_DIMS) return false;
    const size_t elem = nt_dtype_size(dtype);
    if (!elem || elem > INT32_MAX) return false;
    size_t bytes = elem;
    for (uint8_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0 || bytes > INT32_MAX ||
            (size_t)shape[i] > SIZE_MAX / bytes) return false;
        bytes *= (size_t)shape[i];
    }
    return bytes / elem <= (size_t)INT64_MAX;
}

/* Simple xorshift64 RNG */
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rand_uniform(uint64_t* state) {
    return (float)(xorshift64(state) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
}

/* Box-Muller transform for normal distribution */
static float rand_normal(uint64_t* state) {
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    while (u1 == 0.0f) u1 = rand_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

/* ============================================================================
 * TENSOR CREATION
 * ============================================================================ */

nt_tensor_t* nt_tensor_new(nt_context_t* ctx, nt_dtype_t dtype,
                           uint8_t ndim, const int32_t* shape) {
    if (!valid_shape(dtype, ndim, shape)) {
        return NULL;
    }
    
    /* Allocate tensor struct */
    nt_tensor_t* t = (nt_tensor_t*)calloc(1, sizeof(nt_tensor_t));
    if (!t) {
        return NULL;
    }
    
    /* Set shape */
    t->ndim = ndim;
    t->dtype = dtype;
    t->layout = NT_LAYOUT_CONTIGUOUS;
    
    int64_t numel = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        t->ne[i] = shape[i];
        numel *= shape[i];
    }
    
    /* Fill remaining dimensions with 1 */
    for (uint8_t i = ndim; i < NT_MAX_DIMS; i++) {
        t->ne[i] = 1;
    }
    
    /* Compute strides */
    compute_strides(t->nb, t->ne, ndim, dtype);
    for (uint8_t i = ndim; i < NT_MAX_DIMS; i++) {
        t->nb[i] = 0;
    }
    
    /* Allocate storage */
    size_t size_bytes = (size_t)numel * nt_dtype_size(dtype);
    nt_device_id_t device = ctx ? ctx->default_device : NT_DEVICE_CPU;
    nt_allocator_t* allocator = ctx ? ctx->allocator : NULL;
    
    t->storage = nt_storage_new(size_bytes, device, allocator);
    if (!t->storage) {
        free(t);
        return NULL;
    }
    
    t->storage_offset = 0;
    t->data = t->storage->data;
    
    /* Set flags */
    t->flags = NT_FLAG_CONTIGUOUS | NT_FLAG_IS_LEAF;
    
    /* Initialize refcount */
    atomic_init(&t->refcount, 1);
    
    /* No extended metadata initially */
    t->meta = NULL;
    
    /* Register with context */
    if (ctx) {
        nt_context_register_tensor(ctx, t);
    }
    
    return t;
}

nt_tensor_t* nt_tensor_new_1d(nt_context_t* ctx, nt_dtype_t dtype, int32_t n0) {
    int32_t shape[] = {n0};
    return nt_tensor_new(ctx, dtype, 1, shape);
}

nt_tensor_t* nt_tensor_new_2d(nt_context_t* ctx, nt_dtype_t dtype,
                              int32_t n0, int32_t n1) {
    int32_t shape[] = {n0, n1};
    return nt_tensor_new(ctx, dtype, 2, shape);
}

nt_tensor_t* nt_tensor_new_3d(nt_context_t* ctx, nt_dtype_t dtype,
                              int32_t n0, int32_t n1, int32_t n2) {
    int32_t shape[] = {n0, n1, n2};
    return nt_tensor_new(ctx, dtype, 3, shape);
}

nt_tensor_t* nt_tensor_new_4d(nt_context_t* ctx, nt_dtype_t dtype,
                              int32_t n0, int32_t n1, int32_t n2, int32_t n3) {
    int32_t shape[] = {n0, n1, n2, n3};
    return nt_tensor_new(ctx, dtype, 4, shape);
}

nt_tensor_t* nt_tensor_from_ptr(void* data, nt_dtype_t dtype,
                                 uint8_t ndim, const int32_t* shape,
                                 const int32_t* strides) {
    if (!data || !valid_shape(dtype, ndim, shape)) {
        return NULL;
    }
    
    nt_tensor_t* t = (nt_tensor_t*)calloc(1, sizeof(nt_tensor_t));
    if (!t) {
        return NULL;
    }
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->layout = NT_LAYOUT_STRIDED;
    
    int64_t numel = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        t->ne[i] = shape[i];
        numel *= shape[i];
    }
    
    for (uint8_t i = ndim; i < NT_MAX_DIMS; i++) {
        t->ne[i] = 1;
    }
    
    /* Set strides */
    if (strides) {
        for (uint8_t i = 0; i < ndim; i++) {
            t->nb[i] = strides[i];
        }
    } else {
        compute_strides(t->nb, t->ne, ndim, dtype);
    }
    
    for (uint8_t i = ndim; i < NT_MAX_DIMS; i++) {
        t->nb[i] = 0;
    }
    
    /* Create storage from pointer (not owned) */
    size_t size_bytes = (size_t)numel * nt_dtype_size(dtype);
    t->storage = nt_storage_from_ptr(data, size_bytes, NT_DEVICE_CPU);
    if (!t->storage) {
        free(t);
        return NULL;
    }

    /* The legacy pointer API promises only a packed-size allocation. For
     * padded/offset external views use nt_tensor_from_storage with an
     * explicit backing size instead of guessing that inaccessible bytes exist. */
    if (!valid_scalar_span(t->storage, 0, dtype, ndim, t->ne, t->nb)) {
        nt_storage_release(t->storage);
        free(t);
        return NULL;
    }
    
    t->storage_offset = 0;
    t->data = data;
    
    /* Check if contiguous */
    if (check_contiguous(t->ne, t->nb, ndim, dtype)) {
        t->flags = NT_FLAG_CONTIGUOUS;
        t->layout = NT_LAYOUT_CONTIGUOUS;
    }
    
    atomic_init(&t->refcount, 1);
    t->meta = NULL;
    
    return t;
}

nt_tensor_t* nt_tensor_from_storage(nt_storage_t* storage, size_t offset,
                                     nt_dtype_t dtype, uint8_t ndim,
                                     const int32_t* shape, const int32_t* strides) {
    if (!storage || !valid_shape(dtype, ndim, shape)) {
        return NULL;
    }
    
    nt_tensor_t* t = (nt_tensor_t*)calloc(1, sizeof(nt_tensor_t));
    if (!t) {
        return NULL;
    }
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->layout = NT_LAYOUT_STRIDED;
    
    for (uint8_t i = 0; i < ndim; i++) {
        t->ne[i] = shape[i];
    }
    for (uint8_t i = ndim; i < NT_MAX_DIMS; i++) {
        t->ne[i] = 1;
    }
    
    if (strides) {
        for (uint8_t i = 0; i < ndim; i++) {
            t->nb[i] = strides[i];
        }
    } else {
        compute_strides(t->nb, t->ne, ndim, dtype);
    }
    
    for (uint8_t i = ndim; i < NT_MAX_DIMS; i++) {
        t->nb[i] = 0;
    }

    if (!valid_scalar_span(storage, offset, dtype, ndim, t->ne, t->nb)) {
        free(t);
        return NULL;
    }

    t->storage = nt_storage_retain(storage);
    t->storage_offset = offset;
    t->data = (uint8_t*)storage->data + offset;
    
    if (check_contiguous(t->ne, t->nb, ndim, dtype)) {
        t->flags = NT_FLAG_CONTIGUOUS;
        t->layout = NT_LAYOUT_CONTIGUOUS;
    }
    
    t->flags |= NT_FLAG_VIEW;
    
    atomic_init(&t->refcount, 1);
    t->meta = NULL;
    
    return t;
}

nt_tensor_t* nt_tensor_clone(const nt_tensor_t* src) {
    if (!src || !src->storage || !nt_storage_is_cpu(src->storage) ||
        !valid_scalar_span(src->storage, src->storage_offset, src->dtype,
                           src->ndim, src->ne, src->nb) ||
        src->data != (const uint8_t*)src->storage->data + src->storage_offset)
        return NULL;
    
    nt_tensor_t* dst = nt_tensor_new(NULL, src->dtype, src->ndim, src->ne);
    if (!dst) return NULL;
    
    /* Dimension zero varies fastest (GGML ne[0]). A view's raw byte span
     * is not its logical sequence: each destination element must be indexed
     * through the source byte strides. No source alias survives the clone. */
    if (nt_tensor_is_contiguous(src) &&
        check_contiguous(src->ne, src->nb, src->ndim, src->dtype)) {
        memcpy(dst->data, src->data, nt_tensor_nbytes(src));
    } else {
        const size_t elem = nt_dtype_size(src->dtype);
        const size_t count = (size_t)nt_tensor_numel(src);
        for (size_t linear = 0; linear < count; ++linear) {
            size_t position = linear;
            size_t src_offset = 0;
            for (uint8_t axis = 0; axis < src->ndim; ++axis) {
                src_offset += (position % (size_t)src->ne[axis]) * (size_t)src->nb[axis];
                position /= (size_t)src->ne[axis];
            }
            memcpy((uint8_t*)dst->data + linear * elem,
                   (const uint8_t*)src->data + src_offset, elem);
        }
    }
    
    /* Copy metadata if present */
    if (src->meta) {
        dst->meta = (nt_tensor_meta_t*)calloc(1, sizeof(nt_tensor_meta_t));
        if (dst->meta) {
            memcpy(dst->meta->name, src->meta->name, NT_MAX_NAME);
        }
    }
    
    return dst;
}

nt_tensor_t* nt_tensor_contiguous(const nt_tensor_t* src) {
    if (!src) return NULL;
    
    if (nt_tensor_is_contiguous(src)) {
        return nt_tensor_retain((nt_tensor_t*)src);
    }
    
    return nt_tensor_clone(src);
}

/* ============================================================================
 * TENSOR LIFECYCLE
 * ============================================================================ */

nt_tensor_t* nt_tensor_retain(nt_tensor_t* tensor) {
    if (tensor) {
        atomic_fetch_add(&tensor->refcount, 1);
    }
    return tensor;
}

void nt_tensor_release(nt_tensor_t* tensor) {
    if (!tensor) return;
    
    if (atomic_fetch_sub(&tensor->refcount, 1) == 1) {
        nt_tensor_free(tensor);
    }
}

int32_t nt_tensor_refcount(const nt_tensor_t* tensor) {
    return tensor ? atomic_load(&tensor->refcount) : 0;
}

void nt_tensor_free(nt_tensor_t* tensor) {
    if (!tensor) return;
    
    /* Free metadata */
    if (tensor->meta) {
        if (tensor->meta->grad) {
            nt_tensor_release(tensor->meta->grad);
        }
        if (tensor->meta->user_data && tensor->meta->user_free) {
            tensor->meta->user_free(tensor->meta->user_data);
        }
        free(tensor->meta);
    }
    
    /* Release storage */
    if (tensor->storage) {
        nt_storage_release(tensor->storage);
    }
    
    free(tensor);
}

/* ============================================================================
 * TENSOR VIEWS
 * ============================================================================ */

nt_tensor_t* nt_tensor_view(nt_tensor_t* src) {
    if (!src) return NULL;
    
    return nt_tensor_from_storage(src->storage, src->storage_offset,
                                   src->dtype, src->ndim, src->ne, src->nb);
}

nt_tensor_t* nt_tensor_reshape(nt_tensor_t* src, uint8_t ndim, const int32_t* shape) {
    if (!src || !nt_tensor_is_contiguous(src) || !valid_shape(src->dtype, ndim, shape)) {
        return NULL;
    }
    
    /* Check that total elements match */
    int64_t src_numel = nt_tensor_numel(src);
    int64_t dst_numel = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        dst_numel *= shape[i];
    }
    
    if (src_numel != dst_numel) {
        return NULL;
    }
    
    return nt_tensor_from_storage(src->storage, src->storage_offset,
                                   src->dtype, ndim, shape, NULL);
}

nt_tensor_t* nt_tensor_transpose(nt_tensor_t* src, int dim0, int dim1) {
    if (!src) return NULL;
    
    if (dim0 < 0) dim0 += src->ndim;
    if (dim1 < 0) dim1 += src->ndim;
    
    if (dim0 < 0 || dim0 >= src->ndim || dim1 < 0 || dim1 >= src->ndim) {
        return NULL;
    }

    int dims[NT_MAX_DIMS];
    for (uint8_t i = 0; i < src->ndim; ++i) dims[i] = i;
    dims[dim0] = dim1;
    dims[dim1] = dim0;
    nt_tensor_t* t = nt_tensor_permute(src, dims);
    if (!t) return NULL;
    if (dim0 != dim1) t->flags |= NT_FLAG_TRANSPOSED;
    return t;
}

nt_tensor_t* nt_tensor_permute(nt_tensor_t* src, const int* dims) {
    if (!src || !dims || !src->ndim || src->ndim > NT_MAX_DIMS) return NULL;

    /* dims[out_axis] = in_axis, matching Torch's axis-order convention.
     * This is a metadata-only view: the values move only in contiguous(). */
    bool used[NT_MAX_DIMS] = {false};
    int32_t shape[NT_MAX_DIMS];
    int32_t strides[NT_MAX_DIMS];
    for (uint8_t i = 0; i < src->ndim; ++i) {
        const int axis = dims[i];
        if (axis < 0 || axis >= src->ndim || used[axis]) return NULL;
        used[axis] = true;
        shape[i] = src->ne[axis];
        strides[i] = src->nb[axis];
    }
    return nt_tensor_from_storage(src->storage, src->storage_offset,
                                  src->dtype, src->ndim, shape, strides);
}

nt_tensor_t* nt_tensor_slice(nt_tensor_t* src, int dim, int start, int end) {
    if (!src || !src->ndim || src->ndim > NT_MAX_DIMS) return NULL;
    if (dim < 0) dim += src->ndim;
    if (dim < 0 || dim >= src->ndim || start < 0 || end <= start ||
        end > src->ne[dim] || src->nb[dim] < 0) return NULL;

    const size_t delta = (size_t)start * (size_t)src->nb[dim];
    if (delta > SIZE_MAX - src->storage_offset) return NULL;
    int32_t shape[NT_MAX_DIMS];
    for (uint8_t i = 0; i < src->ndim; ++i) shape[i] = src->ne[i];
    shape[dim] = end - start;
    return nt_tensor_from_storage(src->storage, src->storage_offset + delta,
                                  src->dtype, src->ndim, shape, src->nb);
}

nt_tensor_t* nt_tensor_squeeze(nt_tensor_t* src, int dim) {
    if (!src) return NULL;
    
    if (dim < 0) dim += src->ndim;
    if (dim < 0 || dim >= src->ndim || src->ne[dim] != 1) {
        return NULL;
    }
    
    int32_t new_shape[NT_MAX_DIMS];
    int32_t new_strides[NT_MAX_DIMS];
    uint8_t new_ndim = 0;
    
    for (uint8_t i = 0; i < src->ndim; i++) {
        if (i != dim) {
            new_shape[new_ndim] = src->ne[i];
            new_strides[new_ndim] = src->nb[i];
            new_ndim++;
        }
    }
    
    return nt_tensor_from_storage(src->storage, src->storage_offset,
                                   src->dtype, new_ndim, new_shape, new_strides);
}

nt_tensor_t* nt_tensor_unsqueeze(nt_tensor_t* src, int dim) {
    if (!src) return NULL;
    
    if (dim < 0) dim += src->ndim + 1;
    if (dim < 0 || dim > src->ndim || src->ndim >= NT_MAX_DIMS) {
        return NULL;
    }
    
    int32_t new_shape[NT_MAX_DIMS];
    int32_t new_strides[NT_MAX_DIMS];
    
    for (int i = 0; i < dim; i++) {
        new_shape[i] = src->ne[i];
        new_strides[i] = src->nb[i];
    }
    
    new_shape[dim] = 1;
    new_strides[dim] = dim < src->ndim ? src->nb[dim] : src->nb[src->ndim - 1] * src->ne[src->ndim - 1];
    
    for (int i = dim; i < src->ndim; i++) {
        new_shape[i + 1] = src->ne[i];
        new_strides[i + 1] = src->nb[i];
    }
    
    return nt_tensor_from_storage(src->storage, src->storage_offset,
                                   src->dtype, src->ndim + 1, new_shape, new_strides);
}

nt_tensor_t* nt_tensor_flatten(nt_tensor_t* src) {
    if (!src) return NULL;
    
    int32_t shape[] = {(int32_t)nt_tensor_numel(src)};
    return nt_tensor_reshape(src, 1, shape);
}

/* ============================================================================
 * TENSOR INITIALIZATION
 * ============================================================================ */

nt_status_t nt_tensor_zero(nt_tensor_t* t) {
    if (!t || !t->data) return NT_ERR_INVALID_ARG;
    memset(t->data, 0, nt_tensor_nbytes(t));
    return NT_OK;
}

nt_status_t nt_tensor_ones(nt_tensor_t* t) {
    return nt_tensor_fill(t, 1.0f);
}

nt_status_t nt_tensor_fill(nt_tensor_t* t, float value) {
    if (!t || !t->data) return NT_ERR_INVALID_ARG;
    
    int64_t n = nt_tensor_numel(t);
    
    switch (t->dtype) {
        case NT_F32: {
            float* data = (float*)t->data;
            for (int64_t i = 0; i < n; i++) data[i] = value;
            break;
        }
        case NT_F64: {
            double* data = (double*)t->data;
            for (int64_t i = 0; i < n; i++) data[i] = (double)value;
            break;
        }
        case NT_I32: {
            int32_t* data = (int32_t*)t->data;
            int32_t v = (int32_t)value;
            for (int64_t i = 0; i < n; i++) data[i] = v;
            break;
        }
        case NT_I64: {
            int64_t* data = (int64_t*)t->data;
            int64_t v = (int64_t)value;
            for (int64_t i = 0; i < n; i++) data[i] = v;
            break;
        }
        default:
            return NT_ERR_NOT_IMPLEMENTED;
    }
    
    return NT_OK;
}

nt_status_t nt_tensor_rand(nt_tensor_t* t, uint64_t* seed) {
    if (!t || !t->data) return NT_ERR_INVALID_ARG;
    
    uint64_t local_seed = seed ? *seed : 12345;
    int64_t n = nt_tensor_numel(t);
    
    if (t->dtype == NT_F32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < n; i++) {
            data[i] = rand_uniform(&local_seed);
        }
    } else if (t->dtype == NT_F64) {
        double* data = (double*)t->data;
        for (int64_t i = 0; i < n; i++) {
            data[i] = (double)rand_uniform(&local_seed);
        }
    } else {
        return NT_ERR_NOT_IMPLEMENTED;
    }
    
    if (seed) *seed = local_seed;
    return NT_OK;
}

nt_status_t nt_tensor_randn(nt_tensor_t* t, float mean, float std, uint64_t* seed) {
    if (!t || !t->data) return NT_ERR_INVALID_ARG;
    
    uint64_t local_seed = seed ? *seed : 12345;
    int64_t n = nt_tensor_numel(t);
    
    if (t->dtype == NT_F32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < n; i++) {
            data[i] = mean + std * rand_normal(&local_seed);
        }
    } else if (t->dtype == NT_F64) {
        double* data = (double*)t->data;
        for (int64_t i = 0; i < n; i++) {
            data[i] = (double)(mean + std * rand_normal(&local_seed));
        }
    } else {
        return NT_ERR_NOT_IMPLEMENTED;
    }
    
    if (seed) *seed = local_seed;
    return NT_OK;
}

nt_status_t nt_tensor_set_data(nt_tensor_t* t, const void* data, size_t size) {
    if (!t || !t->data || !data) return NT_ERR_INVALID_ARG;
    
    size_t tensor_size = nt_tensor_nbytes(t);
    if (size > tensor_size) size = tensor_size;
    
    memcpy(t->data, data, size);
    return NT_OK;
}

/* ============================================================================
 * TENSOR METADATA
 * ============================================================================ */

nt_tensor_meta_t* nt_tensor_get_meta(nt_tensor_t* t) {
    if (!t) return NULL;
    
    if (!t->meta) {
        t->meta = (nt_tensor_meta_t*)calloc(1, sizeof(nt_tensor_meta_t));
    }
    
    return t->meta;
}

void nt_tensor_set_name(nt_tensor_t* t, const char* name) {
    if (!t || !name) return;
    
    nt_tensor_meta_t* meta = nt_tensor_get_meta(t);
    if (meta) {
        strncpy(meta->name, name, NT_MAX_NAME - 1);
        meta->name[NT_MAX_NAME - 1] = '\0';
    }
}

const char* nt_tensor_get_name(const nt_tensor_t* t) {
    if (!t || !t->meta) return "";
    return t->meta->name;
}

void nt_tensor_set_requires_grad(nt_tensor_t* t, bool requires_grad) {
    if (!t) return;
    
    if (requires_grad) {
        t->flags |= NT_FLAG_REQUIRES_GRAD;
    } else {
        t->flags &= ~NT_FLAG_REQUIRES_GRAD;
    }
}

nt_tensor_t* nt_tensor_grad(const nt_tensor_t* t) {
    if (!t || !t->meta) return NULL;
    return t->meta->grad;
}

/* ============================================================================
 * TENSOR DEVICE OPERATIONS
 * ============================================================================ */

nt_tensor_t* nt_tensor_to_device(const nt_tensor_t* t, nt_device_id_t device) {
    if (!t) return NULL;
    
    nt_device_id_t current = nt_tensor_device(t);
    if (current.type == device.type && current.index == device.index) {
        return nt_tensor_retain((nt_tensor_t*)t);
    }
    
    /* Create new tensor on target device */
    nt_tensor_t* new_t = nt_tensor_new(NULL, t->dtype, t->ndim, t->ne);
    if (!new_t) return NULL;
    
    /* Copy data */
    memcpy(new_t->data, t->data, nt_tensor_nbytes(t));
    
    return new_t;
}

/* ============================================================================
 * TENSOR PRINTING
 * ============================================================================ */

void nt_tensor_print_info(const nt_tensor_t* t) {
    if (!t) {
        printf("Tensor: NULL\n");
        return;
    }
    
    printf("Tensor");
    if (t->meta && t->meta->name[0]) {
        printf(" '%s'", t->meta->name);
    }
    printf(": dtype=%s, shape=[", nt_dtype_name(t->dtype));
    
    for (uint8_t i = 0; i < t->ndim; i++) {
        printf("%d", t->ne[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], ");
    
    printf("strides=[");
    for (uint8_t i = 0; i < t->ndim; i++) {
        printf("%d", t->nb[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], ");
    
    printf("device=%s, contiguous=%s, refcount=%d\n",
           nt_device_name(nt_tensor_device(t).type),
           nt_tensor_is_contiguous(t) ? "true" : "false",
           nt_tensor_refcount(t));
}

void nt_tensor_print(const nt_tensor_t* t) {
    nt_tensor_print_ex(t, 4, 10);
}

void nt_tensor_print_ex(const nt_tensor_t* t, int precision, int max_elements) {
    if (!t || !t->data) {
        printf("Tensor: NULL\n");
        return;
    }
    
    nt_tensor_print_info(t);
    
    int64_t n = nt_tensor_numel(t);
    if (n > max_elements) {
        printf("  [");
        for (int i = 0; i < max_elements / 2; i++) {
            if (t->dtype == NT_F32) {
                printf("%.*f, ", precision, ((float*)t->data)[i]);
            } else if (t->dtype == NT_I32) {
                printf("%d, ", ((int32_t*)t->data)[i]);
            }
        }
        printf("..., ");
        for (int64_t i = n - max_elements / 2; i < n; i++) {
            if (t->dtype == NT_F32) {
                printf("%.*f", precision, ((float*)t->data)[i]);
            } else if (t->dtype == NT_I32) {
                printf("%d", ((int32_t*)t->data)[i]);
            }
            if (i < n - 1) printf(", ");
        }
        printf("]\n");
    } else {
        printf("  [");
        for (int64_t i = 0; i < n; i++) {
            if (t->dtype == NT_F32) {
                printf("%.*f", precision, ((float*)t->data)[i]);
            } else if (t->dtype == NT_F64) {
                printf("%.*f", precision, ((double*)t->data)[i]);
            } else if (t->dtype == NT_I32) {
                printf("%d", ((int32_t*)t->data)[i]);
            } else if (t->dtype == NT_I64) {
                printf("%ld", ((int64_t*)t->data)[i]);
            }
            if (i < n - 1) printf(", ");
        }
        printf("]\n");
    }
}
