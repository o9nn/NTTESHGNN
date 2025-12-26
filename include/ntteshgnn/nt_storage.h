/**
 * @file nt_storage.h
 * @brief NTTESHGNN - Storage and Memory Management
 * 
 * Storage represents the actual memory allocation that backs tensors.
 * Multiple tensors can share the same storage (views).
 * 
 * Design principles:
 * - From PyTorch: Storage/View separation
 * - From GGML: Explicit memory management, no hidden allocations
 * - Novel: Multi-device support with unified interface
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_STORAGE_H
#define NTTESHGNN_STORAGE_H

#include "nt_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ALLOCATOR INTERFACE
 * ============================================================================ */

/**
 * @brief Memory allocator function signatures
 */
typedef void* (*nt_alloc_fn)(void* ctx, size_t size, size_t align);
typedef void* (*nt_realloc_fn)(void* ctx, void* ptr, size_t old_size, size_t new_size, size_t align);
typedef void  (*nt_free_fn)(void* ctx, void* ptr, size_t size);

/**
 * @brief Allocator interface
 * 
 * Custom allocators can be provided for different memory management strategies:
 * - Bump allocator: Fast, no individual frees
 * - Pool allocator: Fixed-size blocks
 * - Arena allocator: Region-based
 * - System allocator: malloc/free
 */
struct nt_allocator {
    const char*     name;           /**< Allocator name for debugging */
    void*           ctx;            /**< Allocator context/state */
    
    nt_alloc_fn     alloc;          /**< Allocation function */
    nt_realloc_fn   realloc;        /**< Reallocation function (optional) */
    nt_free_fn      free;           /**< Deallocation function */
    
    /* Statistics */
    size_t          total_allocated;
    size_t          peak_allocated;
    size_t          allocation_count;
};

/* Built-in allocators */
nt_allocator_t* nt_allocator_system(void);      /**< System malloc/free */
nt_allocator_t* nt_allocator_aligned(void);     /**< Aligned allocation */

/**
 * @brief Create a bump allocator (arena-style, fast allocation, bulk free)
 * @param size Total arena size
 * @return Allocator instance
 */
nt_allocator_t* nt_allocator_bump_create(size_t size);
void nt_allocator_bump_reset(nt_allocator_t* alloc);
void nt_allocator_bump_destroy(nt_allocator_t* alloc);

/**
 * @brief Create a pool allocator (fixed-size blocks)
 * @param block_size Size of each block
 * @param num_blocks Number of blocks
 * @return Allocator instance
 */
nt_allocator_t* nt_allocator_pool_create(size_t block_size, size_t num_blocks);
void nt_allocator_pool_destroy(nt_allocator_t* alloc);

/* ============================================================================
 * STORAGE STRUCTURE
 * ============================================================================ */

/**
 * @brief Storage flags
 */
#define NT_STORAGE_FLAG_OWNED       (1 << 0)    /**< Storage owns its data */
#define NT_STORAGE_FLAG_PINNED      (1 << 1)    /**< Pinned memory (no swap) */
#define NT_STORAGE_FLAG_MAPPED      (1 << 2)    /**< Memory-mapped from file */
#define NT_STORAGE_FLAG_SHARED      (1 << 3)    /**< Shared memory (IPC) */
#define NT_STORAGE_FLAG_READONLY    (1 << 4)    /**< Read-only storage */

/**
 * @brief Storage structure - the actual memory allocation
 * 
 * Storage is reference-counted and can be shared by multiple tensors.
 * When refcount reaches 0, the memory is freed.
 */
struct nt_storage {
    /* Data pointer and size */
    void*               data;           /**< Raw data pointer */
    size_t              size_bytes;     /**< Total allocation size */
    
    /* Device information */
    nt_device_id_t      device;         /**< Device where memory resides */
    
    /* Reference counting (atomic for thread safety) */
    _Atomic int32_t     refcount;
    
    /* Allocator that created this storage */
    nt_allocator_t*     allocator;
    
    /* Flags */
    uint32_t            flags;
    
    /* Memory-mapped file info (if NT_STORAGE_FLAG_MAPPED) */
    int                 fd;             /**< File descriptor */
    size_t              file_offset;    /**< Offset in file */
    
    /* Debug info */
    const char*         debug_name;     /**< Optional name for debugging */
};

/* ============================================================================
 * STORAGE LIFECYCLE
 * ============================================================================ */

/**
 * @brief Create new storage with given size
 * @param size_bytes Size in bytes
 * @param device Target device
 * @param allocator Allocator to use (NULL for default)
 * @return New storage or NULL on failure
 */
nt_storage_t* nt_storage_new(size_t size_bytes, nt_device_id_t device, nt_allocator_t* allocator);

/**
 * @brief Create storage from existing data (does not take ownership)
 * @param data Existing data pointer
 * @param size_bytes Size in bytes
 * @param device Device where data resides
 * @return New storage wrapping the data
 */
nt_storage_t* nt_storage_from_ptr(void* data, size_t size_bytes, nt_device_id_t device);

/**
 * @brief Create storage by memory-mapping a file
 * @param path File path
 * @param offset Offset in file
 * @param size Size to map (0 for entire file)
 * @param readonly Map as read-only
 * @return New storage or NULL on failure
 */
nt_storage_t* nt_storage_mmap(const char* path, size_t offset, size_t size, bool readonly);

/**
 * @brief Increment reference count
 * @param storage Storage to retain
 * @return Same storage pointer
 */
nt_storage_t* nt_storage_retain(nt_storage_t* storage);

/**
 * @brief Decrement reference count, free if zero
 * @param storage Storage to release
 */
void nt_storage_release(nt_storage_t* storage);

/**
 * @brief Get current reference count
 */
int32_t nt_storage_refcount(const nt_storage_t* storage);

/* ============================================================================
 * STORAGE OPERATIONS
 * ============================================================================ */

/**
 * @brief Resize storage (may reallocate)
 * @param storage Storage to resize
 * @param new_size New size in bytes
 * @return NT_OK on success
 */
nt_status_t nt_storage_resize(nt_storage_t* storage, size_t new_size);

/**
 * @brief Copy data between storages
 * @param dst Destination storage
 * @param dst_offset Offset in destination
 * @param src Source storage
 * @param src_offset Offset in source
 * @param size Number of bytes to copy
 * @return NT_OK on success
 */
nt_status_t nt_storage_copy(nt_storage_t* dst, size_t dst_offset,
                            const nt_storage_t* src, size_t src_offset,
                            size_t size);

/**
 * @brief Fill storage with a byte value
 * @param storage Storage to fill
 * @param value Byte value
 * @param offset Start offset
 * @param size Number of bytes
 * @return NT_OK on success
 */
nt_status_t nt_storage_fill(nt_storage_t* storage, uint8_t value, size_t offset, size_t size);

/**
 * @brief Zero out storage
 */
nt_status_t nt_storage_zero(nt_storage_t* storage);

/**
 * @brief Get data pointer with offset
 */
NT_INLINE void* nt_storage_data_at(nt_storage_t* storage, size_t offset) {
    return (uint8_t*)storage->data + offset;
}

/* ============================================================================
 * DEVICE TRANSFER
 * ============================================================================ */

/**
 * @brief Copy storage to a different device
 * @param storage Source storage
 * @param device Target device
 * @param allocator Allocator for new storage (NULL for default)
 * @return New storage on target device
 */
nt_storage_t* nt_storage_to_device(const nt_storage_t* storage, nt_device_id_t device, 
                                    nt_allocator_t* allocator);

/**
 * @brief Check if storage is on CPU
 */
NT_INLINE bool nt_storage_is_cpu(const nt_storage_t* storage) {
    return storage->device.type == NT_DEV_CPU;
}

/**
 * @brief Check if storage is on GPU (CUDA/Metal/Vulkan)
 */
NT_INLINE bool nt_storage_is_gpu(const nt_storage_t* storage) {
    return storage->device.type == NT_DEV_CUDA ||
           storage->device.type == NT_DEV_METAL ||
           storage->device.type == NT_DEV_VULKAN;
}

/* ============================================================================
 * PINNED MEMORY (for async transfers)
 * ============================================================================ */

/**
 * @brief Create pinned (page-locked) storage for faster GPU transfers
 * @param size_bytes Size in bytes
 * @return Pinned storage or NULL
 */
nt_storage_t* nt_storage_pinned(size_t size_bytes);

/**
 * @brief Check if storage is pinned
 */
NT_INLINE bool nt_storage_is_pinned(const nt_storage_t* storage) {
    return (storage->flags & NT_STORAGE_FLAG_PINNED) != 0;
}

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_STORAGE_H */
