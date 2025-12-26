/**
 * @file ntteshgnn.h
 * @brief NTTESHGNN - Nested Tensor Typed Echo-State HyperGraph Neural Network
 * 
 * Main header file that includes all NTTESHGNN components.
 * 
 * NTTESHGNN is a unified tensor computation framework that combines:
 * - GGML's efficiency (minimal struct overhead, explicit memory management)
 * - PyTorch's flexibility (dispatch system, storage/view separation)
 * - THTensor's simplicity (shape + stride + data model)
 * - Novel capabilities (nested tensors, type system, echo state, hypergraph)
 * 
 * Architecture Layers:
 * - Layer 0: Primitive types (nt_types.h)
 * - Layer 1: Core tensor structure (nt_tensor.h, nt_storage.h, nt_nested.h)
 * - Layer 2: Type system (nt_typesystem.h)
 * - Layer 3: Hypergraph structure (nt_hypergraph.h)
 * - Layer 4: Echo state networks (nt_echostate.h)
 * - Layer 5: Context management (nt_context.h)
 * - Layer 6: Operations (nt_ops.h)
 * - Layer 7: VTNPU backend (nt_vtnpu.h)
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 * @see https://github.com/o9nn/NTTESHGNN
 */

#ifndef NTTESHGNN_H
#define NTTESHGNN_H

/* Layer 0: Primitive Types */
#include "nt_types.h"

/* Layer 1: Core Structures */
#include "nt_storage.h"
#include "nt_tensor.h"
#include "nt_nested.h"

/* Layer 2: Type System */
#include "nt_typesystem.h"

/* Layer 3: Hypergraph */
#include "nt_hypergraph.h"

/* Layer 4: Echo State Networks */
#include "nt_echostate.h"

/* Layer 5: Context Management */
#include "nt_context.h"

/* Layer 6: Operations */
#include "nt_ops.h"

/* Layer 7: VTNPU Backend (optional) */
#ifdef NT_ENABLE_VTNPU
#include "nt_vtnpu.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * LIBRARY INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize NTTESHGNN library
 * 
 * Must be called before using any other functions.
 * Creates the default context and initializes backends.
 */
void nt_init(void);

/**
 * @brief Cleanup NTTESHGNN library
 * 
 * Frees all resources and shuts down backends.
 * Should be called before program exit.
 */
void nt_cleanup(void);

/**
 * @brief Get library version string
 */
const char* nt_version(void);

/**
 * @brief Get library build info
 */
const char* nt_build_info(void);

/* ============================================================================
 * QUICK START API
 * ============================================================================ */

/**
 * @brief Create a tensor filled with zeros
 */
nt_tensor_t* nt_zeros(int32_t n0, int32_t n1, int32_t n2, int32_t n3);

/**
 * @brief Create a tensor filled with ones
 */
nt_tensor_t* nt_ones(int32_t n0, int32_t n1, int32_t n2, int32_t n3);

/**
 * @brief Create a tensor with random values
 */
nt_tensor_t* nt_rand_tensor(int32_t n0, int32_t n1, int32_t n2, int32_t n3);

/**
 * @brief Create a tensor from float array
 */
nt_tensor_t* nt_from_float(const float* data, int32_t n0, int32_t n1, int32_t n2, int32_t n3);

/**
 * @brief Create identity matrix
 */
nt_tensor_t* nt_eye(int32_t n);

/**
 * @brief Create range tensor [start, end) with step
 */
nt_tensor_t* nt_arange(float start, float end, float step);

/**
 * @brief Create linearly spaced tensor
 */
nt_tensor_t* nt_linspace(float start, float end, int32_t n);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_H */
