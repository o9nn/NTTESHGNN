/**
 * @file nt_context.h
 * @brief NTTESHGNN - Context and Memory Management (Layer 5)
 * 
 * The context manages memory allocation, tensor lifecycle, and
 * provides scratch buffers for temporary computations.
 * 
 * Design principles:
 * - From GGML: Explicit memory management, no hidden allocations
 * - Arena-based allocation for fast tensor creation
 * - Scratch buffers for temporary computation results
 * - Thread-safe reference counting
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_CONTEXT_H
#define NTTESHGNN_CONTEXT_H

#include "nt_types.h"
#include "nt_storage.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONTEXT FLAGS
 * ============================================================================ */

#define NT_CTX_FLAG_NO_ALLOC        (1 << 0)    /**< Don't allocate tensor data */
#define NT_CTX_FLAG_MEASURE_ONLY    (1 << 1)    /**< Only measure memory, don't allocate */
#define NT_CTX_FLAG_THREAD_SAFE     (1 << 2)    /**< Enable thread-safe operations */

/* ============================================================================
 * SCRATCH BUFFER
 * ============================================================================ */

/**
 * @brief Scratch buffer for temporary allocations
 */
typedef struct nt_scratch {
    void*       data;           /**< Buffer data */
    size_t      size;           /**< Total size */
    size_t      used;           /**< Currently used */
    size_t      peak;           /**< Peak usage */
} nt_scratch_t;

/* ============================================================================
 * CONTEXT STRUCTURE
 * ============================================================================ */

/**
 * @brief Memory context for tensor operations
 */
struct nt_context {
    /* Memory management */
    nt_allocator_t*     allocator;          /**< Primary allocator */
    void*               mem_buffer;         /**< Memory buffer (if using bump allocator) */
    size_t              mem_size;           /**< Total memory size */
    size_t              mem_used;           /**< Memory used */
    size_t              mem_peak;           /**< Peak memory usage */
    
    /* Scratch buffers (for temporary computations) */
    nt_scratch_t        scratch[2];         /**< Double-buffered scratch */
    int                 scratch_idx;        /**< Current scratch buffer index */
    bool                scratch_save;       /**< Save scratch state */
    
    /* Tensor registry */
    nt_tensor_t**       tensors;            /**< Registered tensors */
    uint32_t            n_tensors;          /**< Number of tensors */
    uint32_t            tensors_capacity;   /**< Tensor array capacity */
    
    /* Graph registry */
    nt_graph_t**        graphs;             /**< Registered graphs */
    uint32_t            n_graphs;           /**< Number of graphs */
    
    /* Backend */
    nt_backend_t*       backend;            /**< Compute backend */
    nt_device_id_t      default_device;     /**< Default device for new tensors */
    
    /* Configuration */
    uint32_t            flags;              /**< Context flags */
    int                 n_threads;          /**< Number of threads for computation */
    
    /* Statistics */
    uint64_t            total_allocations;  /**< Total allocation count */
    uint64_t            total_bytes_allocated;
    uint64_t            total_frees;
    
    /* Thread safety */
    void*               mutex;              /**< Mutex for thread safety */
    
    /* Debug */
    bool                debug_mode;         /**< Enable debug output */
    FILE*               debug_file;         /**< Debug output file */
};

/* ============================================================================
 * CONTEXT LIFECYCLE
 * ============================================================================ */

/**
 * @brief Create context with default settings
 * @return New context
 */
nt_context_t* nt_context_new(void);

/**
 * @brief Create context with specific memory size
 * @param mem_size Memory buffer size
 * @return New context
 */
nt_context_t* nt_context_new_with_size(size_t mem_size);

/**
 * @brief Create context with existing memory buffer
 * @param buffer Memory buffer
 * @param size Buffer size
 * @return New context
 */
nt_context_t* nt_context_new_with_buffer(void* buffer, size_t size);

/**
 * @brief Create context with custom allocator
 * @param allocator Custom allocator
 * @return New context
 */
nt_context_t* nt_context_new_with_allocator(nt_allocator_t* allocator);

/**
 * @brief Free context and all associated resources
 */
void nt_context_free(nt_context_t* ctx);

/**
 * @brief Reset context (free all tensors, keep configuration)
 */
void nt_context_reset(nt_context_t* ctx);

/* ============================================================================
 * CONTEXT CONFIGURATION
 * ============================================================================ */

/**
 * @brief Set default device for new tensors
 */
void nt_context_set_device(nt_context_t* ctx, nt_device_id_t device);

/**
 * @brief Set number of threads
 */
void nt_context_set_threads(nt_context_t* ctx, int n_threads);

/**
 * @brief Set compute backend
 */
void nt_context_set_backend(nt_context_t* ctx, nt_backend_t* backend);

/**
 * @brief Enable/disable debug mode
 */
void nt_context_set_debug(nt_context_t* ctx, bool enable, FILE* output);

/**
 * @brief Set context flags
 */
void nt_context_set_flags(nt_context_t* ctx, uint32_t flags);

/* ============================================================================
 * MEMORY ALLOCATION
 * ============================================================================ */

/**
 * @brief Allocate memory from context
 * @param ctx Context
 * @param size Size in bytes
 * @param align Alignment (0 for default)
 * @return Allocated memory or NULL
 */
void* nt_context_alloc(nt_context_t* ctx, size_t size, size_t align);

/**
 * @brief Free memory to context
 */
void nt_context_free_mem(nt_context_t* ctx, void* ptr, size_t size);

/**
 * @brief Get remaining memory
 */
size_t nt_context_remaining(const nt_context_t* ctx);

/**
 * @brief Get used memory
 */
size_t nt_context_used(const nt_context_t* ctx);

/* ============================================================================
 * SCRATCH BUFFER MANAGEMENT
 * ============================================================================ */

/**
 * @brief Set scratch buffer
 * @param ctx Context
 * @param idx Buffer index (0 or 1)
 * @param data Buffer data
 * @param size Buffer size
 */
void nt_context_set_scratch(nt_context_t* ctx, int idx, void* data, size_t size);

/**
 * @brief Get current scratch buffer
 */
nt_scratch_t* nt_context_get_scratch(nt_context_t* ctx);

/**
 * @brief Allocate from scratch buffer
 */
void* nt_scratch_alloc(nt_scratch_t* scratch, size_t size, size_t align);

/**
 * @brief Reset scratch buffer
 */
void nt_scratch_reset(nt_scratch_t* scratch);

/**
 * @brief Save scratch state (for nested operations)
 */
size_t nt_scratch_save(nt_scratch_t* scratch);

/**
 * @brief Restore scratch state
 */
void nt_scratch_restore(nt_scratch_t* scratch, size_t saved);

/**
 * @brief Switch to other scratch buffer
 */
void nt_context_switch_scratch(nt_context_t* ctx);

/* ============================================================================
 * TENSOR REGISTRATION
 * ============================================================================ */

/**
 * @brief Register tensor with context
 */
void nt_context_register_tensor(nt_context_t* ctx, nt_tensor_t* t);

/**
 * @brief Unregister tensor from context
 */
void nt_context_unregister_tensor(nt_context_t* ctx, nt_tensor_t* t);

/**
 * @brief Get all registered tensors
 */
nt_tensor_t** nt_context_get_tensors(nt_context_t* ctx, uint32_t* count);

/**
 * @brief Find tensor by name
 */
nt_tensor_t* nt_context_find_tensor(nt_context_t* ctx, const char* name);

/* ============================================================================
 * GRAPH REGISTRATION
 * ============================================================================ */

/**
 * @brief Register graph with context
 */
void nt_context_register_graph(nt_context_t* ctx, nt_graph_t* g);

/**
 * @brief Unregister graph from context
 */
void nt_context_unregister_graph(nt_context_t* ctx, nt_graph_t* g);

/* ============================================================================
 * CONTEXT STATISTICS
 * ============================================================================ */

/**
 * @brief Context statistics
 */
typedef struct nt_context_stats {
    size_t      mem_total;
    size_t      mem_used;
    size_t      mem_peak;
    size_t      scratch_used[2];
    size_t      scratch_peak[2];
    uint32_t    n_tensors;
    uint32_t    n_graphs;
    uint64_t    total_allocations;
    uint64_t    total_bytes;
} nt_context_stats_t;

/**
 * @brief Get context statistics
 */
void nt_context_get_stats(const nt_context_t* ctx, nt_context_stats_t* stats);

/**
 * @brief Print context statistics
 */
void nt_context_print_stats(const nt_context_t* ctx);

/* ============================================================================
 * GLOBAL CONTEXT
 * ============================================================================ */

/**
 * @brief Get global default context
 */
nt_context_t* nt_context_default(void);

/**
 * @brief Set global default context
 */
void nt_context_set_default(nt_context_t* ctx);

/**
 * @brief Initialize library (creates default context)
 */
void nt_init(void);

/**
 * @brief Cleanup library
 */
void nt_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_CONTEXT_H */
