/**
 * @file nt_storage.c
 * @brief NTTESHGNN - Storage implementation
 */

#include "ntteshgnn/nt_storage.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef NT_OS_LINUX
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ============================================================================
 * SYSTEM ALLOCATOR
 * ============================================================================ */

static void* system_alloc(void* ctx, size_t size, size_t align) {
    NT_UNUSED_VAR(ctx);
    if (align <= sizeof(void*)) {
        return malloc(size);
    }
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void* system_realloc(void* ctx, void* ptr, size_t old_size, size_t new_size, size_t align) {
    NT_UNUSED_VAR(ctx);
    NT_UNUSED_VAR(old_size);
    NT_UNUSED_VAR(align);
    return realloc(ptr, new_size);
}

static void system_free(void* ctx, void* ptr, size_t size) {
    NT_UNUSED_VAR(ctx);
    NT_UNUSED_VAR(size);
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static nt_allocator_t g_system_allocator = {
    .name = "system",
    .ctx = NULL,
    .alloc = system_alloc,
    .realloc = system_realloc,
    .free = system_free,
    .total_allocated = 0,
    .peak_allocated = 0,
    .allocation_count = 0,
};

nt_allocator_t* nt_allocator_system(void) {
    return &g_system_allocator;
}

/* ============================================================================
 * ALIGNED ALLOCATOR
 * ============================================================================ */

static void* aligned_alloc_fn(void* ctx, size_t size, size_t align) {
    NT_UNUSED_VAR(ctx);
    if (align < NT_DEFAULT_ALIGN) {
        align = NT_DEFAULT_ALIGN;
    }
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static nt_allocator_t g_aligned_allocator = {
    .name = "aligned",
    .ctx = NULL,
    .alloc = aligned_alloc_fn,
    .realloc = NULL,
    .free = system_free,
    .total_allocated = 0,
    .peak_allocated = 0,
    .allocation_count = 0,
};

nt_allocator_t* nt_allocator_aligned(void) {
    return &g_aligned_allocator;
}

/* ============================================================================
 * BUMP ALLOCATOR
 * ============================================================================ */

typedef struct bump_ctx {
    uint8_t*    buffer;
    size_t      size;
    size_t      offset;
    size_t      peak;
} bump_ctx_t;

static void* bump_alloc(void* ctx, size_t size, size_t align) {
    bump_ctx_t* bump = (bump_ctx_t*)ctx;
    
    /* Align offset */
    size_t aligned_offset = NT_ALIGN_UP(bump->offset, align);
    
    /* Check space */
    if (aligned_offset + size > bump->size) {
        return NULL;
    }
    
    void* ptr = bump->buffer + aligned_offset;
    bump->offset = aligned_offset + size;
    
    if (bump->offset > bump->peak) {
        bump->peak = bump->offset;
    }
    
    return ptr;
}

static void bump_free(void* ctx, void* ptr, size_t size) {
    /* Bump allocator doesn't free individual allocations */
    NT_UNUSED_VAR(ctx);
    NT_UNUSED_VAR(ptr);
    NT_UNUSED_VAR(size);
}

nt_allocator_t* nt_allocator_bump_create(size_t size) {
    nt_allocator_t* alloc = (nt_allocator_t*)malloc(sizeof(nt_allocator_t));
    if (!alloc) return NULL;
    
    bump_ctx_t* ctx = (bump_ctx_t*)malloc(sizeof(bump_ctx_t));
    if (!ctx) {
        free(alloc);
        return NULL;
    }
    
    ctx->buffer = (uint8_t*)aligned_alloc_fn(NULL, size, NT_DEFAULT_ALIGN);
    if (!ctx->buffer) {
        free(ctx);
        free(alloc);
        return NULL;
    }
    
    ctx->size = size;
    ctx->offset = 0;
    ctx->peak = 0;
    
    alloc->name = "bump";
    alloc->ctx = ctx;
    alloc->alloc = bump_alloc;
    alloc->realloc = NULL;
    alloc->free = bump_free;
    alloc->total_allocated = 0;
    alloc->peak_allocated = 0;
    alloc->allocation_count = 0;
    
    return alloc;
}

void nt_allocator_bump_reset(nt_allocator_t* alloc) {
    if (alloc && alloc->ctx) {
        bump_ctx_t* ctx = (bump_ctx_t*)alloc->ctx;
        ctx->offset = 0;
    }
}

void nt_allocator_bump_destroy(nt_allocator_t* alloc) {
    if (alloc) {
        if (alloc->ctx) {
            bump_ctx_t* ctx = (bump_ctx_t*)alloc->ctx;
            if (ctx->buffer) {
                free(ctx->buffer);
            }
            free(ctx);
        }
        free(alloc);
    }
}

/* ============================================================================
 * POOL ALLOCATOR
 * ============================================================================ */

typedef struct pool_ctx {
    uint8_t*    buffer;
    size_t      block_size;
    size_t      num_blocks;
    uint32_t*   free_list;
    uint32_t    free_head;
    uint32_t    allocated;
} pool_ctx_t;

static void* pool_alloc(void* ctx, size_t size, size_t align) {
    pool_ctx_t* pool = (pool_ctx_t*)ctx;
    NT_UNUSED_VAR(align);
    
    if (size > pool->block_size || pool->free_head == UINT32_MAX) {
        return NULL;
    }
    
    uint32_t idx = pool->free_head;
    pool->free_head = pool->free_list[idx];
    pool->allocated++;
    
    return pool->buffer + idx * pool->block_size;
}

static void pool_free(void* ctx, void* ptr, size_t size) {
    pool_ctx_t* pool = (pool_ctx_t*)ctx;
    NT_UNUSED_VAR(size);
    
    if (!ptr) return;
    
    size_t offset = (uint8_t*)ptr - pool->buffer;
    uint32_t idx = (uint32_t)(offset / pool->block_size);
    
    pool->free_list[idx] = pool->free_head;
    pool->free_head = idx;
    pool->allocated--;
}

nt_allocator_t* nt_allocator_pool_create(size_t block_size, size_t num_blocks) {
    nt_allocator_t* alloc = (nt_allocator_t*)malloc(sizeof(nt_allocator_t));
    if (!alloc) return NULL;
    
    pool_ctx_t* ctx = (pool_ctx_t*)malloc(sizeof(pool_ctx_t));
    if (!ctx) {
        free(alloc);
        return NULL;
    }
    
    /* Align block size */
    block_size = NT_ALIGN_UP(block_size, NT_DEFAULT_ALIGN);
    
    ctx->buffer = (uint8_t*)aligned_alloc_fn(NULL, block_size * num_blocks, NT_DEFAULT_ALIGN);
    ctx->free_list = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    
    if (!ctx->buffer || !ctx->free_list) {
        if (ctx->buffer) free(ctx->buffer);
        if (ctx->free_list) free(ctx->free_list);
        free(ctx);
        free(alloc);
        return NULL;
    }
    
    ctx->block_size = block_size;
    ctx->num_blocks = num_blocks;
    ctx->allocated = 0;
    
    /* Initialize free list */
    for (size_t i = 0; i < num_blocks - 1; i++) {
        ctx->free_list[i] = (uint32_t)(i + 1);
    }
    ctx->free_list[num_blocks - 1] = UINT32_MAX;
    ctx->free_head = 0;
    
    alloc->name = "pool";
    alloc->ctx = ctx;
    alloc->alloc = pool_alloc;
    alloc->realloc = NULL;
    alloc->free = pool_free;
    alloc->total_allocated = 0;
    alloc->peak_allocated = 0;
    alloc->allocation_count = 0;
    
    return alloc;
}

void nt_allocator_pool_destroy(nt_allocator_t* alloc) {
    if (alloc) {
        if (alloc->ctx) {
            pool_ctx_t* ctx = (pool_ctx_t*)alloc->ctx;
            if (ctx->buffer) free(ctx->buffer);
            if (ctx->free_list) free(ctx->free_list);
            free(ctx);
        }
        free(alloc);
    }
}

/* ============================================================================
 * STORAGE CREATION
 * ============================================================================ */

nt_storage_t* nt_storage_new(size_t size_bytes, nt_device_id_t device, nt_allocator_t* allocator) {
    if (size_bytes == 0) {
        return NULL;
    }
    
    if (!allocator) {
        allocator = nt_allocator_aligned();
    }
    
    nt_storage_t* storage = (nt_storage_t*)malloc(sizeof(nt_storage_t));
    if (!storage) {
        return NULL;
    }
    
    storage->data = allocator->alloc(allocator->ctx, size_bytes, NT_DEFAULT_ALIGN);
    if (!storage->data) {
        free(storage);
        return NULL;
    }
    
    storage->size_bytes = size_bytes;
    storage->device = device;
    atomic_init(&storage->refcount, 1);
    storage->allocator = allocator;
    storage->flags = NT_STORAGE_FLAG_OWNED;
    storage->fd = -1;
    storage->file_offset = 0;
    storage->debug_name = NULL;
    
    return storage;
}

nt_storage_t* nt_storage_from_ptr(void* data, size_t size_bytes, nt_device_id_t device) {
    if (!data || size_bytes == 0) {
        return NULL;
    }
    
    nt_storage_t* storage = (nt_storage_t*)malloc(sizeof(nt_storage_t));
    if (!storage) {
        return NULL;
    }
    
    storage->data = data;
    storage->size_bytes = size_bytes;
    storage->device = device;
    atomic_init(&storage->refcount, 1);
    storage->allocator = NULL;
    storage->flags = 0;  /* Not owned */
    storage->fd = -1;
    storage->file_offset = 0;
    storage->debug_name = NULL;
    
    return storage;
}

#ifdef NT_OS_LINUX
nt_storage_t* nt_storage_mmap(const char* path, size_t offset, size_t size, bool readonly) {
    int fd = open(path, readonly ? O_RDONLY : O_RDWR);
    if (fd < 0) {
        return NULL;
    }
    
    /* Get file size if size is 0 */
    if (size == 0) {
        off_t file_size = lseek(fd, 0, SEEK_END);
        if (file_size < 0 || (size_t)file_size <= offset) {
            close(fd);
            return NULL;
        }
        size = (size_t)file_size - offset;
    }
    
    int prot = readonly ? PROT_READ : (PROT_READ | PROT_WRITE);
    void* data = mmap(NULL, size, prot, MAP_SHARED, fd, (off_t)offset);
    
    if (data == MAP_FAILED) {
        close(fd);
        return NULL;
    }
    
    nt_storage_t* storage = (nt_storage_t*)malloc(sizeof(nt_storage_t));
    if (!storage) {
        munmap(data, size);
        close(fd);
        return NULL;
    }
    
    storage->data = data;
    storage->size_bytes = size;
    storage->device = NT_DEVICE_CPU;
    atomic_init(&storage->refcount, 1);
    storage->allocator = NULL;
    storage->flags = NT_STORAGE_FLAG_MAPPED | (readonly ? NT_STORAGE_FLAG_READONLY : 0);
    storage->fd = fd;
    storage->file_offset = offset;
    storage->debug_name = NULL;
    
    return storage;
}
#else
nt_storage_t* nt_storage_mmap(const char* path, size_t offset, size_t size, bool readonly) {
    NT_UNUSED_VAR(path);
    NT_UNUSED_VAR(offset);
    NT_UNUSED_VAR(size);
    NT_UNUSED_VAR(readonly);
    return NULL;  /* Not implemented on this platform */
}
#endif

/* ============================================================================
 * STORAGE LIFECYCLE
 * ============================================================================ */

nt_storage_t* nt_storage_retain(nt_storage_t* storage) {
    if (storage) {
        atomic_fetch_add(&storage->refcount, 1);
    }
    return storage;
}

void nt_storage_release(nt_storage_t* storage) {
    if (!storage) return;
    
    if (atomic_fetch_sub(&storage->refcount, 1) == 1) {
        /* Last reference, free the storage */
        if (storage->flags & NT_STORAGE_FLAG_MAPPED) {
#ifdef NT_OS_LINUX
            munmap(storage->data, storage->size_bytes);
            if (storage->fd >= 0) {
                close(storage->fd);
            }
#endif
        } else if (storage->flags & NT_STORAGE_FLAG_OWNED) {
            if (storage->allocator) {
                storage->allocator->free(storage->allocator->ctx, 
                                         storage->data, 
                                         storage->size_bytes);
            } else {
                free(storage->data);
            }
        }
        free(storage);
    }
}

int32_t nt_storage_refcount(const nt_storage_t* storage) {
    return storage ? atomic_load(&storage->refcount) : 0;
}

/* ============================================================================
 * STORAGE OPERATIONS
 * ============================================================================ */

nt_status_t nt_storage_resize(nt_storage_t* storage, size_t new_size) {
    if (!storage) {
        return NT_ERR_INVALID_ARG;
    }
    
    if (storage->flags & NT_STORAGE_FLAG_MAPPED) {
        return NT_ERR_INVALID_STATE;
    }
    
    if (!(storage->flags & NT_STORAGE_FLAG_OWNED)) {
        return NT_ERR_INVALID_STATE;
    }
    
    if (storage->allocator && storage->allocator->realloc) {
        void* new_data = storage->allocator->realloc(
            storage->allocator->ctx,
            storage->data,
            storage->size_bytes,
            new_size,
            NT_DEFAULT_ALIGN
        );
        if (!new_data) {
            return NT_ERR_OUT_OF_MEMORY;
        }
        storage->data = new_data;
    } else {
        void* new_data = realloc(storage->data, new_size);
        if (!new_data) {
            return NT_ERR_OUT_OF_MEMORY;
        }
        storage->data = new_data;
    }
    
    storage->size_bytes = new_size;
    return NT_OK;
}

nt_status_t nt_storage_copy(nt_storage_t* dst, size_t dst_offset,
                            const nt_storage_t* src, size_t src_offset,
                            size_t size) {
    if (!dst || !src) {
        return NT_ERR_INVALID_ARG;
    }
    
    if (dst_offset + size > dst->size_bytes || src_offset + size > src->size_bytes) {
        return NT_ERR_INVALID_ARG;
    }
    
    /* TODO: Handle cross-device copies */
    if (dst->device.type != src->device.type) {
        return NT_ERR_DEVICE_MISMATCH;
    }
    
    memcpy((uint8_t*)dst->data + dst_offset,
           (const uint8_t*)src->data + src_offset,
           size);
    
    return NT_OK;
}

nt_status_t nt_storage_fill(nt_storage_t* storage, uint8_t value, size_t offset, size_t size) {
    if (!storage) {
        return NT_ERR_INVALID_ARG;
    }
    
    if (offset + size > storage->size_bytes) {
        return NT_ERR_INVALID_ARG;
    }
    
    memset((uint8_t*)storage->data + offset, value, size);
    return NT_OK;
}

nt_status_t nt_storage_zero(nt_storage_t* storage) {
    return nt_storage_fill(storage, 0, 0, storage->size_bytes);
}

/* ============================================================================
 * DEVICE TRANSFER
 * ============================================================================ */

nt_storage_t* nt_storage_to_device(const nt_storage_t* storage, nt_device_id_t device,
                                    nt_allocator_t* allocator) {
    if (!storage) {
        return NULL;
    }
    
    /* If already on target device, just retain */
    if (storage->device.type == device.type && storage->device.index == device.index) {
        return nt_storage_retain((nt_storage_t*)storage);
    }
    
    /* Create new storage on target device */
    nt_storage_t* new_storage = nt_storage_new(storage->size_bytes, device, allocator);
    if (!new_storage) {
        return NULL;
    }
    
    /* Copy data (TODO: use device-specific copy for GPU) */
    memcpy(new_storage->data, storage->data, storage->size_bytes);
    
    return new_storage;
}

/* ============================================================================
 * PINNED MEMORY
 * ============================================================================ */

nt_storage_t* nt_storage_pinned(size_t size_bytes) {
    nt_storage_t* storage = nt_storage_new(size_bytes, NT_DEVICE_CPU, NULL);
    if (storage) {
        storage->flags |= NT_STORAGE_FLAG_PINNED;
        /* TODO: Actually pin the memory using mlock or CUDA pinned allocation */
    }
    return storage;
}
