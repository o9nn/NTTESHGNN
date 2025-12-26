/**
 * @file nt_hypergraph.h
 * @brief NTTESHGNN - Hypergraph Structure for Computation (Layer 3)
 * 
 * Unlike traditional DAG-based computation graphs, hypergraphs allow
 * N-ary relationships between tensors. This enables:
 * - Multi-head attention as a single hyperedge (Q, K, V → O)
 * - Message passing in GNNs (src[], dst[], edge_attr → msg)
 * - Complex aggregation patterns
 * 
 * Design principles:
 * - From GGML: Simple graph construction API
 * - From PyTorch: Autograd support through edges
 * - Novel: Hyperedges for N-ary operations
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_HYPERGRAPH_H
#define NTTESHGNN_HYPERGRAPH_H

#include "nt_tensor.h"
#include "nt_typesystem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * EDGE KINDS
 * ============================================================================ */

/**
 * @brief Kind of hyperedge in the graph
 */
typedef enum nt_edge_kind {
    /* Computational edges (produce outputs) */
    NT_EDGE_COMPUTE,        /**< Standard compute operation */
    NT_EDGE_REDUCE,         /**< Reduction operation */
    NT_EDGE_SCATTER,        /**< Scatter operation */
    NT_EDGE_GATHER,         /**< Gather operation */
    
    /* Structural edges (relationships without computation) */
    NT_EDGE_VIEW,           /**< View relationship */
    NT_EDGE_GRADIENT,       /**< Gradient relationship */
    NT_EDGE_CHECKPOINT,     /**< Checkpointing boundary */
    NT_EDGE_ALIAS,          /**< Alias (same data) */
    
    /* Echo state edges (reservoir dynamics) */
    NT_EDGE_RESERVOIR,      /**< Reservoir connection */
    NT_EDGE_FEEDBACK,       /**< Feedback loop */
    NT_EDGE_READOUT,        /**< Readout projection */
    
    /* Hypergraph-specific (N-ary) */
    NT_EDGE_ATTENTION,      /**< Multi-head attention (Q, K, V → O) */
    NT_EDGE_MESSAGE,        /**< Message passing (src[], dst[], edge_attr → msg) */
    NT_EDGE_AGGREGATE,      /**< Aggregation (msg[], idx[] → node_features) */
    NT_EDGE_FUSED,          /**< Fused operation sequence */
    
} nt_edge_kind_t;

/* ============================================================================
 * HYPEREDGE STRUCTURE
 * ============================================================================ */

/**
 * @brief Hyperedge - N-ary relationship between tensors
 */
struct nt_edge {
    nt_edge_kind_t      kind;               /**< Edge type */
    uint64_t            edge_id;            /**< Unique identifier */
    
    /* Connected tensors */
    nt_tensor_t**       inputs;             /**< Input tensors */
    uint16_t            n_inputs;           /**< Number of inputs */
    
    nt_tensor_t**       outputs;            /**< Output tensors */
    uint16_t            n_outputs;          /**< Number of outputs */
    
    /* Edge attributes (learnable or fixed) */
    nt_tensor_t*        edge_weight;        /**< Optional edge weight */
    nt_tensor_t*        edge_attr;          /**< Optional edge attributes */
    
    /* Operation info */
    uint16_t            op_code;            /**< Operation to perform */
    int32_t             op_params[8];       /**< Operation parameters */
    float               op_scalars[4];      /**< Scalar parameters */
    
    /* Execution metadata */
    uint32_t            priority;           /**< Scheduling priority */
    uint32_t            group_id;           /**< For parallel execution */
    uint64_t            compute_cost;       /**< Estimated FLOPs */
    
    /* Gradient info */
    struct nt_edge*     grad_edge;          /**< Corresponding gradient edge */
    bool                requires_grad;      /**< Track gradients */
    
    /* Graph linkage */
    struct nt_edge*     prev;               /**< Previous in edge list */
    struct nt_edge*     next;               /**< Next in edge list */
    
    /* Execution state */
    enum {
        NT_EDGE_PENDING,
        NT_EDGE_READY,
        NT_EDGE_RUNNING,
        NT_EDGE_COMPLETED,
        NT_EDGE_FAILED,
    } state;
    
    /* Debug info */
    const char*         name;               /**< Optional name */
    const char*         source_file;        /**< Source file (debug) */
    int                 source_line;        /**< Source line (debug) */
};

/* ============================================================================
 * GRAPH STRUCTURE
 * ============================================================================ */

/**
 * @brief Hypergraph - container for nodes (tensors) and hyperedges
 */
struct nt_graph {
    /* Name and ID */
    char                name[NT_MAX_NAME];  /**< Graph name */
    uint64_t            graph_id;           /**< Unique identifier */
    
    /* Nodes (tensors) */
    nt_tensor_t**       nodes;              /**< Array of node tensors */
    uint32_t            n_nodes;            /**< Number of nodes */
    uint32_t            nodes_capacity;     /**< Allocated capacity */
    
    /* Edges (hyperedges) */
    nt_edge_t*          edges_head;         /**< Head of edge linked list */
    nt_edge_t*          edges_tail;         /**< Tail of edge linked list */
    uint32_t            n_edges;            /**< Number of edges */
    
    /* Execution order (topologically sorted) */
    nt_edge_t**         exec_order;         /**< Execution order array */
    uint32_t            exec_order_len;     /**< Length of exec order */
    bool                exec_order_valid;   /**< Is exec order up to date */
    
    /* Memory planning */
    size_t              peak_memory;        /**< Peak memory usage */
    size_t              current_memory;     /**< Current memory usage */
    nt_tensor_t**       checkpoints;        /**< Tensors to keep in memory */
    uint32_t            n_checkpoints;      /**< Number of checkpoints */
    
    /* Echo state configuration */
    float               global_spectral_radius; /**< Default spectral radius */
    float               input_scaling;      /**< Default input scaling */
    float               reservoir_density;  /**< Default reservoir density */
    
    /* Graph properties */
    bool                is_training;        /**< Training mode */
    bool                enable_grad;        /**< Enable gradient computation */
    
    /* Context */
    nt_context_t*       ctx;                /**< Memory context */
    
    /* Statistics */
    uint64_t            total_flops;        /**< Total FLOPs in graph */
    uint64_t            total_memory;       /**< Total memory required */
};

/* ============================================================================
 * GRAPH LIFECYCLE
 * ============================================================================ */

/**
 * @brief Create new graph
 * @param ctx Context for allocation (NULL for default)
 * @return New graph
 */
nt_graph_t* nt_graph_new(nt_context_t* ctx);

/**
 * @brief Free graph and all its edges
 */
void nt_graph_free(nt_graph_t* g);

/**
 * @brief Clear graph (remove all nodes and edges)
 */
void nt_graph_clear(nt_graph_t* g);

/**
 * @brief Set graph name
 */
void nt_graph_set_name(nt_graph_t* g, const char* name);

/* ============================================================================
 * NODE MANAGEMENT
 * ============================================================================ */

/**
 * @brief Add tensor as node to graph
 * @param g Graph
 * @param t Tensor to add
 * @return Node index
 */
uint32_t nt_graph_add_node(nt_graph_t* g, nt_tensor_t* t);

/**
 * @brief Get node by index
 */
nt_tensor_t* nt_graph_get_node(nt_graph_t* g, uint32_t index);

/**
 * @brief Find node by tensor pointer
 * @return Node index or -1 if not found
 */
int32_t nt_graph_find_node(nt_graph_t* g, const nt_tensor_t* t);

/* ============================================================================
 * EDGE CONSTRUCTION
 * ============================================================================ */

/**
 * @brief Add hyperedge to graph
 * @param g Graph
 * @param kind Edge kind
 * @param inputs Input tensors
 * @param n_in Number of inputs
 * @param outputs Output tensors
 * @param n_out Number of outputs
 * @param op_code Operation code
 * @return New edge
 */
nt_edge_t* nt_graph_add_edge(nt_graph_t* g, nt_edge_kind_t kind,
                              nt_tensor_t** inputs, uint16_t n_in,
                              nt_tensor_t** outputs, uint16_t n_out,
                              uint16_t op_code);

/**
 * @brief Add simple compute edge (GGML-style: dst = op(src0, src1))
 */
nt_edge_t* nt_graph_add_compute(nt_graph_t* g, uint16_t op,
                                 nt_tensor_t* dst,
                                 nt_tensor_t* src0,
                                 nt_tensor_t* src1);

/**
 * @brief Add unary compute edge (dst = op(src))
 */
nt_edge_t* nt_graph_add_unary(nt_graph_t* g, uint16_t op,
                               nt_tensor_t* dst,
                               nt_tensor_t* src);

/**
 * @brief Set edge parameters
 */
void nt_edge_set_params(nt_edge_t* e, const int32_t* params, int n_params);

/**
 * @brief Set edge scalar parameters
 */
void nt_edge_set_scalars(nt_edge_t* e, const float* scalars, int n_scalars);

/**
 * @brief Set edge name (for debugging)
 */
void nt_edge_set_name(nt_edge_t* e, const char* name);

/* ============================================================================
 * SPECIAL HYPEREDGES
 * ============================================================================ */

/**
 * @brief Add attention hyperedge (Q, K, V → Output)
 * @param g Graph
 * @param Q Query tensor
 * @param K Key tensor
 * @param V Value tensor
 * @param output Output tensor
 * @param mask Optional attention mask
 * @param n_heads Number of attention heads
 * @param scale Attention scale factor
 * @return Attention edge
 */
nt_edge_t* nt_graph_add_attention(nt_graph_t* g,
                                   nt_tensor_t* Q,
                                   nt_tensor_t* K,
                                   nt_tensor_t* V,
                                   nt_tensor_t* output,
                                   nt_tensor_t* mask,
                                   int n_heads,
                                   float scale);

/**
 * @brief Add message passing hyperedge (for GNNs)
 * @param g Graph
 * @param node_features Node feature tensor
 * @param edge_index Edge index tensor [2, E]
 * @param edge_attr Edge attribute tensor [E, D]
 * @param output Output tensor
 * @param aggregation Aggregation type (sum, mean, max)
 * @return Message passing edge
 */
nt_edge_t* nt_graph_add_message_pass(nt_graph_t* g,
                                      nt_tensor_t* node_features,
                                      nt_tensor_t* edge_index,
                                      nt_tensor_t* edge_attr,
                                      nt_tensor_t* output,
                                      uint16_t aggregation);

/**
 * @brief Add aggregation hyperedge
 * @param g Graph
 * @param messages Message tensors
 * @param indices Index tensor for aggregation
 * @param output Output tensor
 * @param aggregation Aggregation type
 * @return Aggregation edge
 */
nt_edge_t* nt_graph_add_aggregate(nt_graph_t* g,
                                   nt_tensor_t* messages,
                                   nt_tensor_t* indices,
                                   nt_tensor_t* output,
                                   uint16_t aggregation);

/* ============================================================================
 * GRAPH ANALYSIS
 * ============================================================================ */

/**
 * @brief Build topological execution order
 * @param g Graph
 * @return true if successful (graph is acyclic)
 */
bool nt_graph_build_exec_order(nt_graph_t* g);

/**
 * @brief Check if graph has cycles
 */
bool nt_graph_has_cycles(const nt_graph_t* g);

/**
 * @brief Get edges that depend on a tensor
 */
nt_edge_t** nt_graph_get_consumers(nt_graph_t* g, const nt_tensor_t* t, uint32_t* count);

/**
 * @brief Get edges that produce a tensor
 */
nt_edge_t* nt_graph_get_producer(nt_graph_t* g, const nt_tensor_t* t);

/* ============================================================================
 * GRAPH EXECUTION
 * ============================================================================ */

/**
 * @brief Execute graph
 * @param g Graph to execute
 * @param n_threads Number of threads to use
 */
void nt_graph_compute(nt_graph_t* g, int n_threads);

/**
 * @brief Execute graph with callback
 */
typedef void (*nt_graph_callback_fn)(nt_edge_t* edge, void* ctx);
void nt_graph_compute_with_callback(nt_graph_t* g, int n_threads,
                                     nt_graph_callback_fn callback, void* ctx);

/**
 * @brief Execute single edge
 */
void nt_edge_compute(nt_edge_t* e);

/* ============================================================================
 * MEMORY OPTIMIZATION
 * ============================================================================ */

/**
 * @brief Plan memory allocation for graph
 * @param g Graph
 * @return Peak memory required
 */
size_t nt_graph_plan_memory(nt_graph_t* g);

/**
 * @brief Add checkpoint (tensor to keep in memory)
 */
void nt_graph_add_checkpoint(nt_graph_t* g, nt_tensor_t* t);

/**
 * @brief Enable gradient checkpointing
 */
void nt_graph_enable_checkpointing(nt_graph_t* g, bool enable);

/**
 * @brief Optimize graph (fuse operations, etc.)
 */
void nt_graph_optimize(nt_graph_t* g);

/* ============================================================================
 * GRADIENT COMPUTATION
 * ============================================================================ */

/**
 * @brief Build backward graph for gradient computation
 * @param g Forward graph
 * @param loss Loss tensor
 * @return Backward graph
 */
nt_graph_t* nt_graph_backward(nt_graph_t* g, nt_tensor_t* loss);

/**
 * @brief Compute gradients
 */
void nt_graph_compute_gradients(nt_graph_t* g, nt_tensor_t* loss);

/* ============================================================================
 * GRAPH SERIALIZATION
 * ============================================================================ */

/**
 * @brief Save graph to file
 */
nt_status_t nt_graph_save(const nt_graph_t* g, const char* path);

/**
 * @brief Load graph from file
 */
nt_graph_t* nt_graph_load(const char* path, nt_context_t* ctx);

/* ============================================================================
 * GRAPH VISUALIZATION
 * ============================================================================ */

/**
 * @brief Print graph structure
 */
void nt_graph_print(const nt_graph_t* g);

/**
 * @brief Export graph to DOT format
 */
nt_status_t nt_graph_to_dot(const nt_graph_t* g, const char* path);

/**
 * @brief Get graph statistics
 */
typedef struct nt_graph_stats {
    uint32_t    n_nodes;
    uint32_t    n_edges;
    uint64_t    total_flops;
    size_t      peak_memory;
    size_t      total_params;
    uint32_t    max_depth;
} nt_graph_stats_t;

void nt_graph_get_stats(const nt_graph_t* g, nt_graph_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_HYPERGRAPH_H */
