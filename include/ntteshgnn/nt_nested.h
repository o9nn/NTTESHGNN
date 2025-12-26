/**
 * @file nt_nested.h
 * @brief NTTESHGNN - Nested Tensor Support
 * 
 * Nested tensors (also known as ragged or jagged tensors) allow representing
 * sequences of tensors with varying sizes along one or more dimensions.
 * This is essential for:
 * - Variable-length sequences in NLP
 * - Graphs with varying node counts
 * - Batches of images with different sizes
 * 
 * Design principles:
 * - First-class citizen in the type system
 * - Efficient CSR-style storage for packed data
 * - Seamless integration with regular tensor operations
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_NESTED_H
#define NTTESHGNN_NESTED_H

#include "nt_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * NESTED TENSOR STRUCTURE
 * ============================================================================ */

/**
 * @brief Nested tensor - container for variable-sized sub-tensors
 * 
 * Uses CSR-style indexing for efficient access:
 * - offsets[i] gives the start index of sub-tensor i in the packed data
 * - offsets[n_tensors] gives the total size
 * - sizes[i * ndim_inner + d] gives size of sub-tensor i along dimension d
 */
typedef struct nt_nested_tensor {
    /* Base tensor structure (for unified interface) */
    nt_tensor_t         base;
    
    /* Nested structure info */
    uint32_t            n_tensors;          /**< Number of sub-tensors */
    uint8_t             ndim_inner;         /**< Dimensions per sub-tensor */
    
    /* CSR-style offsets (length: n_tensors + 1) */
    int64_t*            offsets;            /**< Start offset of each sub-tensor */
    
    /* Sizes of each sub-tensor (length: n_tensors * ndim_inner) */
    int32_t*            sizes;              /**< Shape of each sub-tensor */
    
    /* Packed data storage */
    nt_tensor_t*        values;             /**< Packed tensor data */
    
    /* Optional: cached sub-tensor views */
    nt_tensor_t**       cached_views;       /**< Lazily created views */
    
} nt_nested_tensor_t;

/* ============================================================================
 * NESTED TENSOR CREATION
 * ============================================================================ */

/**
 * @brief Create nested tensor from list of tensors
 * @param ctx Context for allocation
 * @param tensors Array of tensors to nest
 * @param n_tensors Number of tensors
 * @return New nested tensor
 */
nt_nested_tensor_t* nt_nested_from_tensors(nt_context_t* ctx,
                                            nt_tensor_t** tensors,
                                            uint32_t n_tensors);

/**
 * @brief Create nested tensor from packed data and offsets
 * @param ctx Context for allocation
 * @param values Packed tensor data
 * @param offsets CSR-style offsets
 * @param sizes Sizes of each sub-tensor
 * @param n_tensors Number of sub-tensors
 * @param ndim_inner Dimensions per sub-tensor
 * @return New nested tensor
 */
nt_nested_tensor_t* nt_nested_from_packed(nt_context_t* ctx,
                                           nt_tensor_t* values,
                                           const int64_t* offsets,
                                           const int32_t* sizes,
                                           uint32_t n_tensors,
                                           uint8_t ndim_inner);

/**
 * @brief Create empty nested tensor with given structure
 * @param ctx Context for allocation
 * @param dtype Data type
 * @param sizes Sizes of each sub-tensor
 * @param n_tensors Number of sub-tensors
 * @param ndim_inner Dimensions per sub-tensor
 * @return New nested tensor
 */
nt_nested_tensor_t* nt_nested_empty(nt_context_t* ctx,
                                     nt_dtype_t dtype,
                                     const int32_t* sizes,
                                     uint32_t n_tensors,
                                     uint8_t ndim_inner);

/**
 * @brief Create nested tensor for sequences (1D inner)
 * @param ctx Context for allocation
 * @param dtype Data type
 * @param lengths Length of each sequence
 * @param n_sequences Number of sequences
 * @return New nested tensor
 */
nt_nested_tensor_t* nt_nested_sequences(nt_context_t* ctx,
                                         nt_dtype_t dtype,
                                         const int32_t* lengths,
                                         uint32_t n_sequences);

/* ============================================================================
 * NESTED TENSOR LIFECYCLE
 * ============================================================================ */

/**
 * @brief Free nested tensor
 */
void nt_nested_free(nt_nested_tensor_t* nested);

/**
 * @brief Retain nested tensor
 */
nt_nested_tensor_t* nt_nested_retain(nt_nested_tensor_t* nested);

/**
 * @brief Release nested tensor
 */
void nt_nested_release(nt_nested_tensor_t* nested);

/* ============================================================================
 * NESTED TENSOR ACCESS
 * ============================================================================ */

/**
 * @brief Get number of sub-tensors
 */
NT_INLINE uint32_t nt_nested_size(const nt_nested_tensor_t* nested) {
    return nested->n_tensors;
}

/**
 * @brief Get sub-tensor at index (creates view)
 * @param nested Nested tensor
 * @param index Sub-tensor index
 * @return View of sub-tensor (do not free separately)
 */
nt_tensor_t* nt_nested_get(nt_nested_tensor_t* nested, uint32_t index);

/**
 * @brief Get sub-tensor shape
 * @param nested Nested tensor
 * @param index Sub-tensor index
 * @param shape Output shape array (must have ndim_inner elements)
 */
void nt_nested_get_shape(const nt_nested_tensor_t* nested, uint32_t index, int32_t* shape);

/**
 * @brief Get sub-tensor size (number of elements)
 */
int64_t nt_nested_get_numel(const nt_nested_tensor_t* nested, uint32_t index);

/**
 * @brief Get offset of sub-tensor in packed data
 */
NT_INLINE int64_t nt_nested_get_offset(const nt_nested_tensor_t* nested, uint32_t index) {
    return nested->offsets[index];
}

/**
 * @brief Get packed values tensor
 */
NT_INLINE nt_tensor_t* nt_nested_values(nt_nested_tensor_t* nested) {
    return nested->values;
}

/* ============================================================================
 * NESTED TENSOR OPERATIONS
 * ============================================================================ */

/**
 * @brief Convert nested tensor to padded regular tensor
 * @param nested Nested tensor
 * @param pad_value Value to use for padding
 * @return Padded tensor with shape [n_tensors, max_size...]
 */
nt_tensor_t* nt_nested_to_padded(const nt_nested_tensor_t* nested, float pad_value);

/**
 * @brief Create padding mask for nested tensor
 * @param nested Nested tensor
 * @return Boolean mask tensor [n_tensors, max_size...]
 */
nt_tensor_t* nt_nested_padding_mask(const nt_nested_tensor_t* nested);

/**
 * @brief Concatenate nested tensors
 * @param nested_list Array of nested tensors
 * @param n Number of nested tensors
 * @return Concatenated nested tensor
 */
nt_nested_tensor_t* nt_nested_cat(nt_nested_tensor_t** nested_list, uint32_t n);

/**
 * @brief Apply function to each sub-tensor
 * @param nested Nested tensor
 * @param fn Function to apply
 * @param ctx User context passed to function
 */
typedef void (*nt_nested_map_fn)(nt_tensor_t* t, uint32_t index, void* ctx);
void nt_nested_map(nt_nested_tensor_t* nested, nt_nested_map_fn fn, void* ctx);

/**
 * @brief Reduce nested tensor along nested dimension
 * @param nested Nested tensor
 * @param op Reduction operation (sum, mean, max, etc.)
 * @return Tensor with shape [n_tensors, ...]
 */
nt_tensor_t* nt_nested_reduce(const nt_nested_tensor_t* nested, uint16_t op);

/* ============================================================================
 * NESTED TENSOR UTILITIES
 * ============================================================================ */

/**
 * @brief Check if tensor is nested
 */
NT_INLINE bool nt_is_nested(const nt_tensor_t* t) {
    return (t->flags & NT_FLAG_NESTED) != 0;
}

/**
 * @brief Cast tensor to nested tensor (unsafe)
 */
NT_INLINE nt_nested_tensor_t* nt_as_nested(nt_tensor_t* t) {
    return (nt_nested_tensor_t*)t;
}

/**
 * @brief Get total number of elements across all sub-tensors
 */
int64_t nt_nested_total_numel(const nt_nested_tensor_t* nested);

/**
 * @brief Get maximum size along a dimension across all sub-tensors
 */
int32_t nt_nested_max_size(const nt_nested_tensor_t* nested, uint8_t dim);

/**
 * @brief Print nested tensor info
 */
void nt_nested_print_info(const nt_nested_tensor_t* nested);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_NESTED_H */
