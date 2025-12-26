/**
 * @file nt_context.c
 * @brief NTTESHGNN - Context implementation
 */

#include "ntteshgnn/nt_context.h"
#include "ntteshgnn/nt_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global default context */
static nt_context_t* g_default_context = NULL;

/* ============================================================================
 * CONTEXT LIFECYCLE
 * ============================================================================ */

nt_context_t* nt_context_new(void) {
    return nt_context_new_with_allocator(nt_allocator_aligned());
}

nt_context_t* nt_context_new_with_size(size_t mem_size) {
    nt_allocator_t* bump = nt_allocator_bump_create(mem_size);
    if (!bump) return NULL;
    
    nt_context_t* ctx = nt_context_new_with_allocator(bump);
    if (!ctx) {
        nt_allocator_bump_destroy(bump);
        return NULL;
    }
    
    ctx->mem_size = mem_size;
    return ctx;
}

nt_context_t* nt_context_new_with_buffer(void* buffer, size_t size) {
    NT_UNUSED_VAR(buffer);
    NT_UNUSED_VAR(size);
    /* TODO: Implement buffer-based context */
    return nt_context_new();
}

nt_context_t* nt_context_new_with_allocator(nt_allocator_t* allocator) {
    nt_context_t* ctx = (nt_context_t*)calloc(1, sizeof(nt_context_t));
    if (!ctx) return NULL;
    
    ctx->allocator = allocator;
    ctx->mem_buffer = NULL;
    ctx->mem_size = 0;
    ctx->mem_used = 0;
    ctx->mem_peak = 0;
    
    /* Initialize scratch buffers */
    ctx->scratch[0].data = NULL;
    ctx->scratch[0].size = 0;
    ctx->scratch[0].used = 0;
    ctx->scratch[0].peak = 0;
    ctx->scratch[1].data = NULL;
    ctx->scratch[1].size = 0;
    ctx->scratch[1].used = 0;
    ctx->scratch[1].peak = 0;
    ctx->scratch_idx = 0;
    ctx->scratch_save = false;
    
    /* Initialize tensor registry */
    ctx->tensors = NULL;
    ctx->n_tensors = 0;
    ctx->tensors_capacity = 0;
    
    /* Initialize graph registry */
    ctx->graphs = NULL;
    ctx->n_graphs = 0;
    
    /* Default settings */
    ctx->backend = NULL;
    ctx->default_device = NT_DEVICE_CPU;
    ctx->flags = 0;
    ctx->n_threads = 1;
    
    /* Statistics */
    ctx->total_allocations = 0;
    ctx->total_bytes_allocated = 0;
    ctx->total_frees = 0;
    
    /* Thread safety */
    ctx->mutex = NULL;
    
    /* Debug */
    ctx->debug_mode = false;
    ctx->debug_file = NULL;
    
    return ctx;
}

void nt_context_free(nt_context_t* ctx) {
    if (!ctx) return;
    
    /* Free all registered tensors that still have references */
    /* Note: We only release our reference, not force free */
    if (ctx->tensors) {
        /* Just free the array, tensors are managed by their own refcount */
        /* Users should release tensors before freeing context */
        free(ctx->tensors);
        ctx->tensors = NULL;
        ctx->n_tensors = 0;
    }
    
    /* Free graphs */
    if (ctx->graphs) {
        free(ctx->graphs);
    }
    
    /* Free scratch buffers */
    if (ctx->scratch[0].data) {
        free(ctx->scratch[0].data);
    }
    if (ctx->scratch[1].data) {
        free(ctx->scratch[1].data);
    }
    
    /* Free allocator if it's a bump allocator we created */
    if (ctx->allocator && strcmp(ctx->allocator->name, "bump") == 0) {
        nt_allocator_bump_destroy(ctx->allocator);
    }
    
    free(ctx);
}

void nt_context_reset(nt_context_t* ctx) {
    if (!ctx) return;
    
    /* Clear tensor registry (don't free, users manage their own tensors) */
    if (ctx->tensors) {
        ctx->n_tensors = 0;
    }
    
    /* Reset allocator if bump */
    if (ctx->allocator && strcmp(ctx->allocator->name, "bump") == 0) {
        nt_allocator_bump_reset(ctx->allocator);
    }
    
    ctx->mem_used = 0;
    
    /* Reset scratch */
    ctx->scratch[0].used = 0;
    ctx->scratch[1].used = 0;
    ctx->scratch_idx = 0;
}

/* ============================================================================
 * CONTEXT CONFIGURATION
 * ============================================================================ */

void nt_context_set_device(nt_context_t* ctx, nt_device_id_t device) {
    if (ctx) {
        ctx->default_device = device;
    }
}

void nt_context_set_threads(nt_context_t* ctx, int n_threads) {
    if (ctx) {
        ctx->n_threads = n_threads > 0 ? n_threads : 1;
    }
}

void nt_context_set_backend(nt_context_t* ctx, nt_backend_t* backend) {
    if (ctx) {
        ctx->backend = backend;
    }
}

void nt_context_set_debug(nt_context_t* ctx, bool enable, FILE* output) {
    if (ctx) {
        ctx->debug_mode = enable;
        ctx->debug_file = output ? output : stderr;
    }
}

void nt_context_set_flags(nt_context_t* ctx, uint32_t flags) {
    if (ctx) {
        ctx->flags = flags;
    }
}

/* ============================================================================
 * MEMORY ALLOCATION
 * ============================================================================ */

void* nt_context_alloc(nt_context_t* ctx, size_t size, size_t align) {
    if (!ctx || !ctx->allocator) {
        return NULL;
    }
    
    if (align == 0) {
        align = NT_DEFAULT_ALIGN;
    }
    
    void* ptr = ctx->allocator->alloc(ctx->allocator->ctx, size, align);
    
    if (ptr) {
        ctx->mem_used += size;
        if (ctx->mem_used > ctx->mem_peak) {
            ctx->mem_peak = ctx->mem_used;
        }
        ctx->total_allocations++;
        ctx->total_bytes_allocated += size;
    }
    
    return ptr;
}

void nt_context_free_mem(nt_context_t* ctx, void* ptr, size_t size) {
    if (!ctx || !ctx->allocator || !ptr) {
        return;
    }
    
    ctx->allocator->free(ctx->allocator->ctx, ptr, size);
    ctx->mem_used -= size;
    ctx->total_frees++;
}

size_t nt_context_remaining(const nt_context_t* ctx) {
    if (!ctx) return 0;
    return ctx->mem_size > ctx->mem_used ? ctx->mem_size - ctx->mem_used : 0;
}

size_t nt_context_used(const nt_context_t* ctx) {
    return ctx ? ctx->mem_used : 0;
}

/* ============================================================================
 * SCRATCH BUFFER MANAGEMENT
 * ============================================================================ */

void nt_context_set_scratch(nt_context_t* ctx, int idx, void* data, size_t size) {
    if (!ctx || idx < 0 || idx > 1) return;
    
    ctx->scratch[idx].data = data;
    ctx->scratch[idx].size = size;
    ctx->scratch[idx].used = 0;
    ctx->scratch[idx].peak = 0;
}

nt_scratch_t* nt_context_get_scratch(nt_context_t* ctx) {
    if (!ctx) return NULL;
    return &ctx->scratch[ctx->scratch_idx];
}

void* nt_scratch_alloc(nt_scratch_t* scratch, size_t size, size_t align) {
    if (!scratch || !scratch->data) return NULL;
    
    size_t aligned_offset = NT_ALIGN_UP(scratch->used, align);
    
    if (aligned_offset + size > scratch->size) {
        return NULL;
    }
    
    void* ptr = (uint8_t*)scratch->data + aligned_offset;
    scratch->used = aligned_offset + size;
    
    if (scratch->used > scratch->peak) {
        scratch->peak = scratch->used;
    }
    
    return ptr;
}

void nt_scratch_reset(nt_scratch_t* scratch) {
    if (scratch) {
        scratch->used = 0;
    }
}

size_t nt_scratch_save(nt_scratch_t* scratch) {
    return scratch ? scratch->used : 0;
}

void nt_scratch_restore(nt_scratch_t* scratch, size_t saved) {
    if (scratch) {
        scratch->used = saved;
    }
}

void nt_context_switch_scratch(nt_context_t* ctx) {
    if (ctx) {
        ctx->scratch_idx = 1 - ctx->scratch_idx;
    }
}

/* ============================================================================
 * TENSOR REGISTRATION
 * ============================================================================ */

void nt_context_register_tensor(nt_context_t* ctx, nt_tensor_t* t) {
    if (!ctx || !t) return;
    
    /* Grow array if needed */
    if (ctx->n_tensors >= ctx->tensors_capacity) {
        uint32_t new_capacity = ctx->tensors_capacity == 0 ? 64 : ctx->tensors_capacity * 2;
        nt_tensor_t** new_tensors = (nt_tensor_t**)realloc(ctx->tensors, 
                                                           new_capacity * sizeof(nt_tensor_t*));
        if (!new_tensors) return;
        
        ctx->tensors = new_tensors;
        ctx->tensors_capacity = new_capacity;
    }
    
    ctx->tensors[ctx->n_tensors++] = t;
}

void nt_context_unregister_tensor(nt_context_t* ctx, nt_tensor_t* t) {
    if (!ctx || !t) return;
    
    for (uint32_t i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i] == t) {
            /* Shift remaining tensors */
            for (uint32_t j = i; j < ctx->n_tensors - 1; j++) {
                ctx->tensors[j] = ctx->tensors[j + 1];
            }
            ctx->n_tensors--;
            return;
        }
    }
}

nt_tensor_t** nt_context_get_tensors(nt_context_t* ctx, uint32_t* count) {
    if (!ctx) {
        if (count) *count = 0;
        return NULL;
    }
    
    if (count) *count = ctx->n_tensors;
    return ctx->tensors;
}

nt_tensor_t* nt_context_find_tensor(nt_context_t* ctx, const char* name) {
    if (!ctx || !name) return NULL;
    
    for (uint32_t i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i] && ctx->tensors[i]->meta) {
            if (strcmp(ctx->tensors[i]->meta->name, name) == 0) {
                return ctx->tensors[i];
            }
        }
    }
    
    return NULL;
}

/* ============================================================================
 * GRAPH REGISTRATION
 * ============================================================================ */

void nt_context_register_graph(nt_context_t* ctx, nt_graph_t* g) {
    NT_UNUSED_VAR(ctx);
    NT_UNUSED_VAR(g);
    /* TODO: Implement */
}

void nt_context_unregister_graph(nt_context_t* ctx, nt_graph_t* g) {
    NT_UNUSED_VAR(ctx);
    NT_UNUSED_VAR(g);
    /* TODO: Implement */
}

/* ============================================================================
 * CONTEXT STATISTICS
 * ============================================================================ */

void nt_context_get_stats(const nt_context_t* ctx, nt_context_stats_t* stats) {
    if (!ctx || !stats) return;
    
    stats->mem_total = ctx->mem_size;
    stats->mem_used = ctx->mem_used;
    stats->mem_peak = ctx->mem_peak;
    stats->scratch_used[0] = ctx->scratch[0].used;
    stats->scratch_used[1] = ctx->scratch[1].used;
    stats->scratch_peak[0] = ctx->scratch[0].peak;
    stats->scratch_peak[1] = ctx->scratch[1].peak;
    stats->n_tensors = ctx->n_tensors;
    stats->n_graphs = ctx->n_graphs;
    stats->total_allocations = ctx->total_allocations;
    stats->total_bytes = ctx->total_bytes_allocated;
}

void nt_context_print_stats(const nt_context_t* ctx) {
    if (!ctx) {
        printf("Context: NULL\n");
        return;
    }
    
    nt_context_stats_t stats;
    nt_context_get_stats(ctx, &stats);
    
    printf("Context Statistics:\n");
    printf("  Memory: %zu / %zu bytes (peak: %zu)\n", 
           stats.mem_used, stats.mem_total, stats.mem_peak);
    printf("  Scratch 0: %zu bytes (peak: %zu)\n", 
           stats.scratch_used[0], stats.scratch_peak[0]);
    printf("  Scratch 1: %zu bytes (peak: %zu)\n", 
           stats.scratch_used[1], stats.scratch_peak[1]);
    printf("  Tensors: %u\n", stats.n_tensors);
    printf("  Graphs: %u\n", stats.n_graphs);
    printf("  Total allocations: %lu (%lu bytes)\n", 
           stats.total_allocations, stats.total_bytes);
}

/* ============================================================================
 * GLOBAL CONTEXT
 * ============================================================================ */

nt_context_t* nt_context_default(void) {
    if (!g_default_context) {
        g_default_context = nt_context_new();
    }
    return g_default_context;
}

void nt_context_set_default(nt_context_t* ctx) {
    g_default_context = ctx;
}

/* ============================================================================
 * LIBRARY INITIALIZATION
 * ============================================================================ */

void nt_init(void) {
    /* Create default context */
    if (!g_default_context) {
        g_default_context = nt_context_new();
    }
    
    /* Initialize common types */
    /* nt_type_init_common(); */
}

void nt_cleanup(void) {
    /* Cleanup common types */
    /* nt_type_cleanup_common(); */
    
    /* Free default context */
    if (g_default_context) {
        nt_context_free(g_default_context);
        g_default_context = NULL;
    }
}
