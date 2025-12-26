/**
 * @file nt_tensor.h
 * @brief NTTESHGNN - Core Tensor Structure (Layer 1)
 * 
 * The fundamental tensor type that forms the basis of all computation.
 * Designed for minimal overhead (64 bytes hot path) while supporting
 * advanced features through optional extended metadata.
 * 
 * Design principles:
 * - From GGML: 64-byte struct, cache-line aligned
 * - From PyTorch: Storage/View separation, autograd support
 * - From THTensor: Simple shape + stride + data model
 * - Novel: First-class nested tensors, hypergraph participation
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_TENSOR_H
#define NTTESHGNN_TENSOR_H

#include "nt_types.h"
#include "nt_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TENSOR FLAGS
 * ============================================================================ */

#define NT_FLAG_CONTIGUOUS      (1 << 0)    /**< Memory is contiguous */
#define NT_FLAG_VIEW            (1 << 1)    /**< This is a view of another tensor */
#define NT_FLAG_REQUIRES_GRAD   (1 << 2)    /**< Track gradients for autograd */
#define NT_FLAG_IS_LEAF         (1 << 3)    /**< Leaf node in compute graph */
#define NT_FLAG_NESTED          (1 << 4)    /**< Contains nested tensors */
#define NT_FLAG_GRAPH_NODE      (1 << 5)    /**< Part of a computation graph */
#define NT_FLAG_ECHO_STATE      (1 << 6)    /**< Echo state reservoir tensor */
#define NT_FLAG_HYPEREDGE       (1 << 7)    /**< Participates in hyperedge */
#define NT_FLAG_IMMUTABLE       (1 << 8)    /**< Read-only after creation */
#define NT_FLAG_PINNED          (1 << 9)    /**< Pinned in memory */
#define NT_FLAG_MAPPED          (1 << 10)   /**< Memory-mapped from file */
#define NT_FLAG_TRANSPOSED      (1 << 11)   /**< Transposed view */
#define NT_FLAG_SPARSE          (1 << 12)   /**< Sparse tensor */
#define NT_FLAG_QUANTIZED       (1 << 13)   /**< Quantized tensor */

/* ============================================================================
 * EXTENDED METADATA (Cold Path)
 * ============================================================================ */

/**
 * @brief Extended tensor metadata
 * 
 * Allocated separately to keep the main tensor struct small.
 * Only created when needed (gradients, graph ops, etc.)
 */
typedef struct nt_tensor_meta {
    /* Naming (debug/visualization) */
    char                name[NT_MAX_NAME];  /**< Human-readable name */
    
    /* Computation graph (GGML-style) */
    nt_tensor_t*        src[NT_MAX_SRC];    /**< Source tensors for this op */
    uint8_t             n_src;              /**< Number of source tensors */
    uint16_t            op;                 /**< Operation that created this */
    int32_t             op_params[8];       /**< Operation parameters */
    
    /* Autograd */
    nt_tensor_t*        grad;               /**< Gradient tensor */
    void*               grad_fn;            /**< Gradient function */
    
    /* Echo state */
    float               spectral_radius;    /**< For reservoir tensors */
    float               leaking_rate;       /**< Leaking rate α */
    
    /* Hypergraph */
    nt_edge_t*          edges;              /**< Linked list of edges */
    uint32_t            n_edges;            /**< Number of edges */
    
    /* User data */
    void*               user_data;          /**< Custom user data */
    void                (*user_free)(void*);/**< Free function for user_data */
    
    /* Performance tracking */
    uint64_t            compute_time_ns;    /**< Last compute time */
    uint64_t            memory_bytes;       /**< Memory footprint */
    
} nt_tensor_meta_t;

/* ============================================================================
 * CORE TENSOR STRUCTURE (64 bytes, cache-line aligned)
 * ============================================================================ */

/**
 * @brief The fundamental tensor structure
 * 
 * Carefully designed to fit in a single cache line (64 bytes) for
 * optimal performance on the hot path. Extended metadata is stored
 * separately and only allocated when needed.
 */
struct NT_ALIGNED(NT_CACHE_LINE) nt_tensor {
    /* === Shape and stride (32 bytes) === */
    int32_t             ne[NT_MAX_DIMS];    /**< Number of elements per dimension */
    int32_t             nb[NT_MAX_DIMS];    /**< Stride in bytes per dimension */
    
    /* === Type and layout (4 bytes) === */
    nt_dtype_t          dtype;              /**< Data type */
    nt_layout_t         layout;             /**< Memory layout */
    
    /* === Flags (4 bytes) === */
    uint32_t            flags;              /**< Tensor flags (NT_FLAG_*) */
    
    /* === Storage reference (8 bytes) === */
    nt_storage_t*       storage;            /**< Underlying storage */
    size_t              storage_offset;     /**< Offset into storage */
    
    /* === Quick access (8 bytes) === */
    void*               data;               /**< Computed: storage->data + offset */
    
    /* === Reference count (4 bytes) === */
    _Atomic int32_t     refcount;
    
    /* === Dimension info (4 bytes) === */
    uint8_t             ndim;               /**< Number of dimensions used */
    uint8_t             _pad[3];            /**< Padding for alignment */
    
    /* === Extended metadata pointer (8 bytes) === */
    nt_tensor_meta_t*   meta;               /**< NULL if not needed */
};

/* Static assertion to verify size */
_Static_assert(sizeof(nt_tensor_t) == 64 || sizeof(nt_tensor_t) == 128, 
               "nt_tensor_t should be 64 or 128 bytes");

/* ============================================================================
 * TENSOR CREATION
 * ============================================================================ */

/**
 * @brief Create a new tensor with given shape
 * @param ctx Context for allocation (NULL for default)
 * @param dtype Data type
 * @param ndim Number of dimensions
 * @param shape Array of dimension sizes
 * @return New tensor or NULL on failure
 */
nt_tensor_t* nt_tensor_new(nt_context_t* ctx, nt_dtype_t dtype, 
                           uint8_t ndim, const int32_t* shape);

/**
 * @brief Create a 1D tensor
 */
nt_tensor_t* nt_tensor_new_1d(nt_context_t* ctx, nt_dtype_t dtype, int32_t n0);

/**
 * @brief Create a 2D tensor (matrix)
 */
nt_tensor_t* nt_tensor_new_2d(nt_context_t* ctx, nt_dtype_t dtype, 
                              int32_t n0, int32_t n1);

/**
 * @brief Create a 3D tensor
 */
nt_tensor_t* nt_tensor_new_3d(nt_context_t* ctx, nt_dtype_t dtype,
                              int32_t n0, int32_t n1, int32_t n2);

/**
 * @brief Create a 4D tensor
 */
nt_tensor_t* nt_tensor_new_4d(nt_context_t* ctx, nt_dtype_t dtype,
                              int32_t n0, int32_t n1, int32_t n2, int32_t n3);

/**
 * @brief Create tensor from existing data (does not copy)
 * @param data Existing data pointer
 * @param dtype Data type
 * @param ndim Number of dimensions
 * @param shape Dimension sizes
 * @param strides Strides in bytes (NULL for contiguous)
 * @return New tensor wrapping the data
 */
nt_tensor_t* nt_tensor_from_ptr(void* data, nt_dtype_t dtype,
                                 uint8_t ndim, const int32_t* shape,
                                 const int32_t* strides);

/**
 * @brief Create tensor from storage with offset
 */
nt_tensor_t* nt_tensor_from_storage(nt_storage_t* storage, size_t offset,
                                     nt_dtype_t dtype, uint8_t ndim,
                                     const int32_t* shape, const int32_t* strides);

/**
 * @brief Create a copy of a tensor
 */
nt_tensor_t* nt_tensor_clone(const nt_tensor_t* src);

/**
 * @brief Create a contiguous copy of a tensor
 */
nt_tensor_t* nt_tensor_contiguous(const nt_tensor_t* src);

/* ============================================================================
 * TENSOR LIFECYCLE
 * ============================================================================ */

/**
 * @brief Increment reference count
 */
nt_tensor_t* nt_tensor_retain(nt_tensor_t* tensor);

/**
 * @brief Decrement reference count, free if zero
 */
void nt_tensor_release(nt_tensor_t* tensor);

/**
 * @brief Get current reference count
 */
int32_t nt_tensor_refcount(const nt_tensor_t* tensor);

/**
 * @brief Free tensor immediately (ignoring refcount)
 */
void nt_tensor_free(nt_tensor_t* tensor);

/* ============================================================================
 * TENSOR PROPERTIES
 * ============================================================================ */

/**
 * @brief Get total number of elements
 */
NT_INLINE int64_t nt_tensor_numel(const nt_tensor_t* t) {
    int64_t n = 1;
    for (uint8_t i = 0; i < t->ndim; i++) {
        n *= t->ne[i];
    }
    return n;
}

/**
 * @brief Get size in bytes
 */
NT_INLINE size_t nt_tensor_nbytes(const nt_tensor_t* t) {
    return (size_t)nt_tensor_numel(t) * nt_dtype_size(t->dtype);
}

/**
 * @brief Check if tensor is contiguous
 */
NT_INLINE bool nt_tensor_is_contiguous(const nt_tensor_t* t) {
    return (t->flags & NT_FLAG_CONTIGUOUS) != 0;
}

/**
 * @brief Check if tensor is a view
 */
NT_INLINE bool nt_tensor_is_view(const nt_tensor_t* t) {
    return (t->flags & NT_FLAG_VIEW) != 0;
}

/**
 * @brief Check if tensor requires gradient
 */
NT_INLINE bool nt_tensor_requires_grad(const nt_tensor_t* t) {
    return (t->flags & NT_FLAG_REQUIRES_GRAD) != 0;
}

/**
 * @brief Get dimension size
 */
NT_INLINE int32_t nt_tensor_size(const nt_tensor_t* t, int dim) {
    if (dim < 0) dim += t->ndim;
    return (dim >= 0 && dim < t->ndim) ? t->ne[dim] : 0;
}

/**
 * @brief Get stride in bytes
 */
NT_INLINE int32_t nt_tensor_stride(const nt_tensor_t* t, int dim) {
    if (dim < 0) dim += t->ndim;
    return (dim >= 0 && dim < t->ndim) ? t->nb[dim] : 0;
}

/**
 * @brief Get data pointer
 */
NT_INLINE void* nt_tensor_data(nt_tensor_t* t) {
    return t->data;
}

NT_INLINE const void* nt_tensor_data_const(const nt_tensor_t* t) {
    return t->data;
}

/**
 * @brief Get typed data pointer
 */
#define nt_tensor_data_f32(t)   ((float*)(t)->data)
#define nt_tensor_data_f64(t)   ((double*)(t)->data)
#define nt_tensor_data_i32(t)   ((int32_t*)(t)->data)
#define nt_tensor_data_i64(t)   ((int64_t*)(t)->data)
#define nt_tensor_data_u8(t)    ((uint8_t*)(t)->data)

/* ============================================================================
 * TENSOR VIEWS
 * ============================================================================ */

/**
 * @brief Create a view of a tensor (shares storage)
 */
nt_tensor_t* nt_tensor_view(nt_tensor_t* src);

/**
 * @brief Create a reshaped view
 */
nt_tensor_t* nt_tensor_reshape(nt_tensor_t* src, uint8_t ndim, const int32_t* shape);

/**
 * @brief Create a transposed view
 */
nt_tensor_t* nt_tensor_transpose(nt_tensor_t* src, int dim0, int dim1);

/**
 * @brief Create a permuted view
 */
nt_tensor_t* nt_tensor_permute(nt_tensor_t* src, const int* dims);

/**
 * @brief Create a sliced view
 */
nt_tensor_t* nt_tensor_slice(nt_tensor_t* src, int dim, int start, int end);

/**
 * @brief Squeeze dimension of size 1
 */
nt_tensor_t* nt_tensor_squeeze(nt_tensor_t* src, int dim);

/**
 * @brief Unsqueeze (add dimension of size 1)
 */
nt_tensor_t* nt_tensor_unsqueeze(nt_tensor_t* src, int dim);

/**
 * @brief Flatten to 1D
 */
nt_tensor_t* nt_tensor_flatten(nt_tensor_t* src);

/* ============================================================================
 * TENSOR INITIALIZATION
 * ============================================================================ */

/**
 * @brief Fill tensor with zeros
 */
nt_status_t nt_tensor_zero(nt_tensor_t* t);

/**
 * @brief Fill tensor with ones
 */
nt_status_t nt_tensor_ones(nt_tensor_t* t);

/**
 * @brief Fill tensor with a scalar value
 */
nt_status_t nt_tensor_fill(nt_tensor_t* t, float value);

/**
 * @brief Fill tensor with random uniform values
 */
nt_status_t nt_tensor_rand(nt_tensor_t* t, uint64_t* seed);

/**
 * @brief Fill tensor with random normal values
 */
nt_status_t nt_tensor_randn(nt_tensor_t* t, float mean, float std, uint64_t* seed);

/**
 * @brief Fill tensor with values from array
 */
nt_status_t nt_tensor_set_data(nt_tensor_t* t, const void* data, size_t size);

/* ============================================================================
 * TENSOR METADATA
 * ============================================================================ */

/**
 * @brief Get or create extended metadata
 */
nt_tensor_meta_t* nt_tensor_get_meta(nt_tensor_t* t);

/**
 * @brief Set tensor name
 */
void nt_tensor_set_name(nt_tensor_t* t, const char* name);

/**
 * @brief Get tensor name
 */
const char* nt_tensor_get_name(const nt_tensor_t* t);

/**
 * @brief Enable gradient tracking
 */
void nt_tensor_set_requires_grad(nt_tensor_t* t, bool requires_grad);

/**
 * @brief Get gradient tensor
 */
nt_tensor_t* nt_tensor_grad(const nt_tensor_t* t);

/* ============================================================================
 * TENSOR DEVICE OPERATIONS
 * ============================================================================ */

/**
 * @brief Get device where tensor resides
 */
NT_INLINE nt_device_id_t nt_tensor_device(const nt_tensor_t* t) {
    return t->storage ? t->storage->device : NT_DEVICE_CPU;
}

/**
 * @brief Move tensor to device
 */
nt_tensor_t* nt_tensor_to_device(const nt_tensor_t* t, nt_device_id_t device);

/**
 * @brief Move tensor to CPU
 */
NT_INLINE nt_tensor_t* nt_tensor_cpu(const nt_tensor_t* t) {
    return nt_tensor_to_device(t, NT_DEVICE_CPU);
}

/* ============================================================================
 * TENSOR TYPE CONVERSION
 * ============================================================================ */

/**
 * @brief Convert tensor to different dtype
 */
nt_tensor_t* nt_tensor_to_dtype(const nt_tensor_t* t, nt_dtype_t dtype);

/**
 * @brief Quantize tensor
 */
nt_tensor_t* nt_tensor_quantize(const nt_tensor_t* t, nt_dtype_t quant_dtype);

/**
 * @brief Dequantize tensor
 */
nt_tensor_t* nt_tensor_dequantize(const nt_tensor_t* t, nt_dtype_t float_dtype);

/* ============================================================================
 * TENSOR PRINTING
 * ============================================================================ */

/**
 * @brief Print tensor info (shape, dtype, etc.)
 */
void nt_tensor_print_info(const nt_tensor_t* t);

/**
 * @brief Print tensor data
 */
void nt_tensor_print(const nt_tensor_t* t);

/**
 * @brief Print tensor data with options
 */
void nt_tensor_print_ex(const nt_tensor_t* t, int precision, int max_elements);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_TENSOR_H */
