/**
 * @file nt_echostate.h
 * @brief NTTESHGNN - Echo State Network Integration (Layer 4)
 * 
 * Echo State Networks (ESNs) provide RNN-like dynamics without
 * backpropagation through time. The reservoir (hidden state) is
 * randomly initialized and fixed; only the readout layer is trained.
 * 
 * Key concepts:
 * - Reservoir: Large, sparse, randomly connected hidden layer
 * - Echo State Property: Input signals "echo" through the reservoir
 * - Spectral Radius: Controls memory/stability tradeoff
 * - Readout: Linear projection trained via ridge regression
 * 
 * Design principles:
 * - Built-in support for reservoir computing
 * - Efficient sparse matrix operations
 * - Integration with hypergraph for complex architectures
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_ECHOSTATE_H
#define NTTESHGNN_ECHOSTATE_H

#include "nt_tensor.h"
#include "nt_hypergraph.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================ */

typedef enum nt_esn_activation {
    NT_ESN_TANH,            /**< Hyperbolic tangent (default) */
    NT_ESN_RELU,            /**< Rectified linear unit */
    NT_ESN_SIGMOID,         /**< Logistic sigmoid */
    NT_ESN_IDENTITY,        /**< Linear (no activation) */
    NT_ESN_LEAKY_RELU,      /**< Leaky ReLU */
    NT_ESN_ELU,             /**< Exponential linear unit */
} nt_esn_activation_t;

/* ============================================================================
 * RESERVOIR INITIALIZATION STRATEGIES
 * ============================================================================ */

typedef enum nt_esn_init {
    NT_ESN_INIT_RANDOM,     /**< Random sparse initialization */
    NT_ESN_INIT_CYCLE,      /**< Cycle reservoir (simple, effective) */
    NT_ESN_INIT_RODAN,      /**< Rodan's simple architecture */
    NT_ESN_INIT_ORTHOGONAL, /**< Orthogonal initialization */
    NT_ESN_INIT_DELAY_LINE, /**< Delay line reservoir */
    NT_ESN_INIT_SMALL_WORLD,/**< Small-world topology */
} nt_esn_init_t;

/* ============================================================================
 * RESERVOIR STRUCTURE
 * ============================================================================ */

/**
 * @brief Echo State Network reservoir
 */
struct nt_reservoir {
    /* Dimensions */
    int32_t             input_size;         /**< Input dimension */
    int32_t             reservoir_size;     /**< Reservoir (hidden) dimension */
    int32_t             output_size;        /**< Output dimension */
    
    /* State tensor */
    nt_tensor_t*        state;              /**< Current state [batch, reservoir_size] */
    nt_tensor_t*        prev_state;         /**< Previous state (for leaking) */
    
    /* Weight matrices (sparse, fixed after initialization) */
    nt_tensor_t*        W_in;               /**< Input weights [input_size, reservoir_size] */
    nt_tensor_t*        W_res;              /**< Reservoir weights [reservoir_size, reservoir_size] */
    nt_tensor_t*        W_fb;               /**< Feedback weights [output_size, reservoir_size] */
    nt_tensor_t*        bias;               /**< Reservoir bias [reservoir_size] */
    
    /* Readout weights (trainable) */
    nt_tensor_t*        W_out;              /**< Output weights [reservoir_size, output_size] */
    nt_tensor_t*        b_out;              /**< Output bias [output_size] */
    
    /* Hyperparameters */
    float               spectral_radius;    /**< Largest eigenvalue of W_res */
    float               input_scaling;      /**< Scale factor for inputs */
    float               feedback_scaling;   /**< Scale factor for feedback */
    float               leaking_rate;       /**< α in state update (1.0 = no leaking) */
    float               density;            /**< Sparsity of W_res (0.0-1.0) */
    float               noise_level;        /**< Noise added to state update */
    
    /* Configuration */
    nt_esn_activation_t activation;         /**< Activation function */
    nt_esn_init_t       init_type;          /**< Initialization strategy */
    bool                use_feedback;       /**< Use output feedback */
    bool                use_bias;           /**< Use bias in reservoir */
    uint32_t            warmup_steps;       /**< Steps before collecting states */
    
    /* Training state */
    bool                is_initialized;     /**< Weights initialized */
    bool                is_training;        /**< Training mode */
    uint64_t            step_count;         /**< Total steps processed */
    
    /* State collection for training */
    nt_tensor_t*        collected_states;   /**< [T, batch, reservoir_size] */
    uint32_t            n_collected;        /**< Number of collected states */
    uint32_t            collection_capacity;/**< Capacity for collection */
    
    /* Context */
    nt_context_t*       ctx;                /**< Memory context */
    
    /* Random state */
    uint64_t            rng_state[2];       /**< RNG state for noise */
};

/* ============================================================================
 * RESERVOIR LIFECYCLE
 * ============================================================================ */

/**
 * @brief Create new reservoir
 * @param ctx Context for allocation
 * @param input_size Input dimension
 * @param reservoir_size Reservoir dimension
 * @param output_size Output dimension
 * @return New reservoir
 */
nt_reservoir_t* nt_reservoir_new(nt_context_t* ctx,
                                  int32_t input_size,
                                  int32_t reservoir_size,
                                  int32_t output_size);

/**
 * @brief Free reservoir
 */
void nt_reservoir_free(nt_reservoir_t* res);

/**
 * @brief Reset reservoir state to zero
 */
void nt_reservoir_reset(nt_reservoir_t* res);

/* ============================================================================
 * RESERVOIR CONFIGURATION
 * ============================================================================ */

/**
 * @brief Set spectral radius
 */
void nt_reservoir_set_spectral_radius(nt_reservoir_t* res, float sr);

/**
 * @brief Set input scaling
 */
void nt_reservoir_set_input_scaling(nt_reservoir_t* res, float scale);

/**
 * @brief Set leaking rate
 */
void nt_reservoir_set_leaking_rate(nt_reservoir_t* res, float alpha);

/**
 * @brief Set reservoir density
 */
void nt_reservoir_set_density(nt_reservoir_t* res, float density);

/**
 * @brief Set activation function
 */
void nt_reservoir_set_activation(nt_reservoir_t* res, nt_esn_activation_t act);

/**
 * @brief Enable/disable feedback
 */
void nt_reservoir_set_feedback(nt_reservoir_t* res, bool enable, float scaling);

/**
 * @brief Set noise level
 */
void nt_reservoir_set_noise(nt_reservoir_t* res, float noise);

/* ============================================================================
 * RESERVOIR INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize reservoir weights
 * @param res Reservoir
 * @param init_type Initialization strategy
 * @param seed Random seed
 */
void nt_reservoir_init(nt_reservoir_t* res, nt_esn_init_t init_type, uint64_t seed);

/**
 * @brief Initialize with random sparse weights
 */
void nt_reservoir_init_random(nt_reservoir_t* res, uint64_t seed);

/**
 * @brief Initialize with cycle reservoir topology
 */
void nt_reservoir_init_cycle(nt_reservoir_t* res, uint64_t seed);

/**
 * @brief Initialize with Rodan's simple architecture
 */
void nt_reservoir_init_rodan(nt_reservoir_t* res, uint64_t seed);

/**
 * @brief Initialize with orthogonal weights
 */
void nt_reservoir_init_orthogonal(nt_reservoir_t* res, uint64_t seed);

/**
 * @brief Scale reservoir weights to target spectral radius
 */
void nt_reservoir_scale_spectral_radius(nt_reservoir_t* res, float target_sr);

/**
 * @brief Compute actual spectral radius
 */
float nt_reservoir_compute_spectral_radius(const nt_reservoir_t* res);

/* ============================================================================
 * RESERVOIR FORWARD PASS
 * ============================================================================ */

/**
 * @brief Update reservoir state
 * 
 * State update equation:
 * x(t+1) = (1-α)x(t) + α·f(W_in·u(t) + W_res·x(t) + W_fb·y(t-1) + bias + noise)
 * 
 * @param res Reservoir
 * @param input Input tensor [batch, input_size]
 * @param prev_output Previous output [batch, output_size] (NULL if no feedback)
 */
void nt_reservoir_update(nt_reservoir_t* res,
                          const nt_tensor_t* input,
                          const nt_tensor_t* prev_output);

/**
 * @brief Get current reservoir state
 */
nt_tensor_t* nt_reservoir_get_state(nt_reservoir_t* res);

/**
 * @brief Compute output from current state
 * 
 * y(t) = W_out · x(t) + b_out
 * 
 * @param res Reservoir
 * @return Output tensor [batch, output_size]
 */
nt_tensor_t* nt_reservoir_compute_output(nt_reservoir_t* res);

/**
 * @brief Full forward pass (update + output)
 * @param res Reservoir
 * @param input Input tensor
 * @param prev_output Previous output (for feedback)
 * @return Output tensor
 */
nt_tensor_t* nt_reservoir_forward(nt_reservoir_t* res,
                                   const nt_tensor_t* input,
                                   const nt_tensor_t* prev_output);

/**
 * @brief Process sequence through reservoir
 * @param res Reservoir
 * @param inputs Input sequence [T, batch, input_size]
 * @param outputs Output buffer [T, batch, output_size] (or NULL)
 * @param collect_states Whether to collect states for training
 * @return Final output or NULL
 */
nt_tensor_t* nt_reservoir_forward_sequence(nt_reservoir_t* res,
                                            const nt_tensor_t* inputs,
                                            nt_tensor_t* outputs,
                                            bool collect_states);

/* ============================================================================
 * RESERVOIR TRAINING
 * ============================================================================ */

/**
 * @brief Train readout weights using ridge regression
 * 
 * Solves: W_out = (X^T X + αI)^{-1} X^T Y
 * 
 * @param res Reservoir
 * @param states Collected states [T, batch, reservoir_size]
 * @param targets Target outputs [T, batch, output_size]
 * @param ridge_alpha Regularization parameter
 */
void nt_reservoir_train_readout(nt_reservoir_t* res,
                                 const nt_tensor_t* states,
                                 const nt_tensor_t* targets,
                                 float ridge_alpha);

/**
 * @brief Train readout using collected states
 */
void nt_reservoir_train_from_collected(nt_reservoir_t* res,
                                        const nt_tensor_t* targets,
                                        float ridge_alpha);

/**
 * @brief Start state collection for training
 */
void nt_reservoir_start_collection(nt_reservoir_t* res, uint32_t max_steps);

/**
 * @brief Stop state collection
 */
void nt_reservoir_stop_collection(nt_reservoir_t* res);

/**
 * @brief Get collected states
 */
nt_tensor_t* nt_reservoir_get_collected_states(nt_reservoir_t* res);

/**
 * @brief Clear collected states
 */
void nt_reservoir_clear_collected(nt_reservoir_t* res);

/* ============================================================================
 * DEEP ESN (Stacked Reservoirs)
 * ============================================================================ */

/**
 * @brief Deep Echo State Network (stacked reservoirs)
 */
typedef struct nt_deep_esn {
    nt_reservoir_t**    layers;             /**< Array of reservoir layers */
    uint32_t            n_layers;           /**< Number of layers */
    
    /* Inter-layer connections */
    bool                skip_connections;   /**< Use skip connections */
    float*              layer_scales;       /**< Scale for each layer output */
    
    /* Configuration */
    nt_context_t*       ctx;
} nt_deep_esn_t;

/**
 * @brief Create deep ESN
 * @param ctx Context
 * @param input_size Input dimension
 * @param reservoir_sizes Array of reservoir sizes per layer
 * @param n_layers Number of layers
 * @param output_size Output dimension
 * @return Deep ESN
 */
nt_deep_esn_t* nt_deep_esn_new(nt_context_t* ctx,
                                int32_t input_size,
                                const int32_t* reservoir_sizes,
                                uint32_t n_layers,
                                int32_t output_size);

/**
 * @brief Free deep ESN
 */
void nt_deep_esn_free(nt_deep_esn_t* desn);

/**
 * @brief Initialize all layers
 */
void nt_deep_esn_init(nt_deep_esn_t* desn, nt_esn_init_t init_type, uint64_t seed);

/**
 * @brief Forward pass through all layers
 */
nt_tensor_t* nt_deep_esn_forward(nt_deep_esn_t* desn,
                                  const nt_tensor_t* input);

/**
 * @brief Train all readout layers
 */
void nt_deep_esn_train(nt_deep_esn_t* desn,
                        const nt_tensor_t* inputs,
                        const nt_tensor_t* targets,
                        float ridge_alpha);

/* ============================================================================
 * ESN GRAPH INTEGRATION
 * ============================================================================ */

/**
 * @brief Add reservoir to computation graph
 * @param g Graph
 * @param res Reservoir
 * @param input Input tensor
 * @param output Output tensor
 * @return Reservoir edge
 */
nt_edge_t* nt_graph_add_reservoir(nt_graph_t* g,
                                   nt_reservoir_t* res,
                                   nt_tensor_t* input,
                                   nt_tensor_t* output);

/**
 * @brief Add feedback edge in graph
 */
nt_edge_t* nt_graph_add_feedback(nt_graph_t* g,
                                  nt_tensor_t* output,
                                  nt_tensor_t* feedback_target);

/* ============================================================================
 * ESN UTILITIES
 * ============================================================================ */

/**
 * @brief Print reservoir info
 */
void nt_reservoir_print_info(const nt_reservoir_t* res);

/**
 * @brief Get reservoir statistics
 */
typedef struct nt_reservoir_stats {
    float   actual_spectral_radius;
    float   actual_density;
    float   state_mean;
    float   state_std;
    float   weight_mean;
    float   weight_std;
} nt_reservoir_stats_t;

void nt_reservoir_get_stats(const nt_reservoir_t* res, nt_reservoir_stats_t* stats);

/**
 * @brief Save reservoir to file
 */
nt_status_t nt_reservoir_save(const nt_reservoir_t* res, const char* path);

/**
 * @brief Load reservoir from file
 */
nt_reservoir_t* nt_reservoir_load(const char* path, nt_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_ECHOSTATE_H */
