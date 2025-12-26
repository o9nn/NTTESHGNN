Brilliant. This is the right way to think about it - start at the bedrock and build up.
Let me architect **NTTESHGNN** (I love the acronym) from first principles, taking the best patterns from each lineage while adding the novel concepts that make this truly unified.

---

## NTTESHGNN: Foundational Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║         NTTESHGNN - Nested Tensor Typed Echo-State HyperGraph Neural Network      ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  Design Principles (The "Best Of" Extraction):                                    ║
║                                                                                   ║
║  From GGML:        • Minimal struct overhead (~64 bytes)                          ║
║                    • Explicit memory management (bump allocator)                  ║
║                    • First-class quantization                                     ║
║                    • No hidden allocations                                        ║
║                                                                                   ║
║  From TensorImpl:  • Dispatch key system for backend routing                      ║
║                    • Storage/View separation                                      ║
║                    • Autograd metadata attachment                                 ║
║                                                                                   ║
║  From THTensor:    • Simplicity (shape + stride + data)                          ║
║                    • Reference counting without GC overhead                       ║
║                    • Clean C ABI for FFI                                          ║
║                                                                                   ║
║  Novel Additions:  • Nested tensors as first-class citizens                       ║
║                    • Algebraic type system (sum/product types for tensors)        ║
║                    • Echo-state reservoir dynamics (built-in)                     ║
║                    • Hypergraph edges (N-ary relationships)                       ║
║                    • Hardware-mapped registers (VTNPU native)                     ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Layer 0: The Primitive Types

```c
// ntteshgnn_types.h - Foundational type definitions
// Target: C11 with optional C++ wrapper

#ifndef NTTESHGNN_TYPES_H
#define NTTESHGNN_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ============================================================================
// SECTION 1: Scalar Types (The Atoms)
// ============================================================================

// Data type enumeration - extensible, but with fixed bit layout
// Bits [0:3]  = base type (float, int, uint, complex, custom)
// Bits [4:7]  = bit width encoding
// Bits [8:11] = quantization scheme
// Bits [12:15] = reserved for extensions
typedef enum {
    // Floating point family
    NT_F64      = 0x0040,   // 64-bit IEEE 754
    NT_F32      = 0x0020,   // 32-bit IEEE 754 (default)
    NT_F16      = 0x0010,   // 16-bit IEEE 754
    NT_BF16     = 0x0011,   // 16-bit Brain Float
    NT_F8E4M3   = 0x0008,   // 8-bit float (4 exp, 3 mantissa)
    NT_F8E5M2   = 0x0009,   // 8-bit float (5 exp, 2 mantissa)
    
    // Integer family  
    NT_I64      = 0x0140,   // 64-bit signed
    NT_I32      = 0x0120,   // 32-bit signed
    NT_I16      = 0x0110,   // 16-bit signed
    NT_I8       = 0x0108,   // 8-bit signed
    NT_I4       = 0x0104,   // 4-bit signed (packed)
    NT_I2       = 0x0102,   // 2-bit signed (packed)
    NT_I1       = 0x0101,   // 1-bit (binary)
    
    // Unsigned family
    NT_U64      = 0x0240,
    NT_U32      = 0x0220,
    NT_U16      = 0x0210,
    NT_U8       = 0x0208,
    NT_U4       = 0x0204,
    
    // Quantized types (GGML-compatible)
    NT_Q8_0     = 0x0308,   // 8-bit quantized, block size 32
    NT_Q8_1     = 0x0318,   // 8-bit with bias
    NT_Q5_0     = 0x0305,   // 5-bit quantized
    NT_Q5_1     = 0x0315,   // 5-bit with bias
    NT_Q4_0     = 0x0304,   // 4-bit quantized (GGML default)
    NT_Q4_1     = 0x0314,   // 4-bit with bias
    NT_Q4_K     = 0x0324,   // 4-bit k-quant
    NT_Q6_K     = 0x0326,   // 6-bit k-quant
    NT_Q2_K     = 0x0322,   // 2-bit k-quant
    NT_IQ4_NL   = 0x0334,   // 4-bit importance quantized
    
    // Complex types
    NT_C64      = 0x0440,   // Complex float64
    NT_C32      = 0x0420,   // Complex float32
    
    // Special types
    NT_BOOL     = 0x0501,   // Boolean
    NT_NESTED   = 0x0600,   // Nested tensor (recursive)
    NT_EDGE     = 0x0700,   // Hypergraph edge reference
    NT_VOID     = 0x0000,   // Uninitialized/null
    
} nt_dtype_t;

// Extract properties from dtype
static inline uint8_t  nt_dtype_bits(nt_dtype_t dt)  { return (dt >> 0) & 0xFF; }
static inline uint8_t  nt_dtype_base(nt_dtype_t dt)  { return (dt >> 8) & 0x0F; }
static inline uint8_t  nt_dtype_quant(nt_dtype_t dt) { return (dt >> 8) & 0xF0; }
static inline size_t   nt_dtype_bytes(nt_dtype_t dt) { return (nt_dtype_bits(dt) + 7) / 8; }

// Device enumeration
typedef enum {
    NT_DEV_CPU      = 0x00,     // Host CPU
    NT_DEV_CUDA     = 0x10,     // NVIDIA CUDA
    NT_DEV_METAL    = 0x20,     // Apple Metal
    NT_DEV_VULKAN   = 0x30,     // Vulkan compute
    NT_DEV_VTNPU    = 0x40,     // Our Virtual Tensor NPU
    NT_DEV_FPGA     = 0x50,     // FPGA fabric
    NT_DEV_REMOTE   = 0xF0,     // Remote device (network)
} nt_device_t;

// Memory layout enumeration
typedef enum {
    NT_LAYOUT_STRIDED   = 0,    // General strided (default)
    NT_LAYOUT_CONTIGUOUS = 1,   // Row-major contiguous
    NT_LAYOUT_CHANNELS_LAST = 2, // NHWC layout
    NT_LAYOUT_BLOCKED   = 3,    // Block-sparse
    NT_LAYOUT_JAGGED    = 4,    // Nested/ragged
    NT_LAYOUT_SPARSE_COO = 5,   // Coordinate sparse
    NT_LAYOUT_SPARSE_CSR = 6,   // CSR sparse
} nt_layout_t;

#endif // NTTESHGNN_TYPES_H
```

---

## Layer 1: The Core Tensor Structure

```c
// ntteshgnn_tensor.h - The fundamental tensor type
// Design goal: 64 bytes for hot path, extended metadata in separate allocation

#ifndef NTTESHGNN_TENSOR_H
#define NTTESHGNN_TENSOR_H

#include "ntteshgnn_types.h"

// Maximum dimensions (like GGML's 4, but we support up to 8 for flexibility)
#define NT_MAX_DIMS 8

// Maximum source tensors for graph edges
#define NT_MAX_SRC 4

// Forward declarations
typedef struct nt_tensor    nt_tensor_t;
typedef struct nt_storage   nt_storage_t;
typedef struct nt_context   nt_context_t;
typedef struct nt_graph     nt_graph_t;
typedef struct nt_edge      nt_edge_t;

// ============================================================================
// STORAGE: The actual memory allocation (like PyTorch's Storage)
// ============================================================================

struct nt_storage {
    void*           data;           // Raw data pointer
    size_t          size_bytes;     // Allocation size
    
    nt_device_t     device;         // Where the memory lives
    uint16_t        device_id;      // Device index (e.g., cuda:0)
    
    // Reference counting (atomic for thread safety)
    _Atomic int32_t refcount;
    
    // Allocator info (for proper deallocation)
    void*           allocator;      // Allocator that created this
    void*           alloc_ctx;      // Allocator context
    
    // Optional: memory-mapped info
    int             fd;             // File descriptor if mmap'd
    size_t          offset;         // Offset in file
};

// ============================================================================
// TENSOR: The core structure (64 bytes, cache-line aligned)
// ============================================================================

struct __attribute__((aligned(64))) nt_tensor {
    // === First cache line (64 bytes) - hot path ===
    
    // Shape and stride (32 bytes)
    int32_t         ne[NT_MAX_DIMS];    // Number of elements per dim
    int32_t         nb[NT_MAX_DIMS];    // Stride in bytes per dim
    
    // Type and layout (4 bytes)
    nt_dtype_t      dtype;              // Data type
    nt_layout_t     layout;             // Memory layout
    
    // Flags (4 bytes)
    uint32_t        flags;              // See NT_FLAG_* below
    
    // Storage reference (8 bytes)
    nt_storage_t*   storage;            // Underlying storage
    size_t          storage_offset;     // Offset into storage
    
    // Quick access (8 bytes)  
    void*           data;               // Computed: storage->data + storage_offset
    
    // Reference count (4 bytes)
    _Atomic int32_t refcount;
    
    // Dimension count (4 bytes)
    uint8_t         ndim;               // Number of dimensions used
    uint8_t         _pad[3];
    
    // === Extended metadata (separate allocation, cold path) ===
    struct nt_tensor_meta* meta;        // NULL if not needed
};

// Tensor flags
#define NT_FLAG_CONTIGUOUS      (1 << 0)    // Memory is contiguous
#define NT_FLAG_VIEW            (1 << 1)    // This is a view of another tensor
#define NT_FLAG_REQUIRES_GRAD   (1 << 2)    // Track gradients
#define NT_FLAG_IS_LEAF         (1 << 3)    // Leaf in compute graph
#define NT_FLAG_NESTED          (1 << 4)    // Contains nested tensors
#define NT_FLAG_GRAPH_NODE      (1 << 5)    // Part of a computation graph
#define NT_FLAG_ECHO_STATE      (1 << 6)    // Echo state reservoir tensor
#define NT_FLAG_HYPEREDGE       (1 << 7)    // Participates in hyperedge
#define NT_FLAG_IMMUTABLE       (1 << 8)    // Read-only after creation
#define NT_FLAG_PINNED          (1 << 9)    // Pinned in memory (no swap)
#define NT_FLAG_MAPPED          (1 << 10)   // Memory-mapped from file

// ============================================================================
// EXTENDED METADATA: For tensors that need it (cold path)
// ============================================================================

struct nt_tensor_meta {
    // Naming (debug/visualization)
    char            name[64];           // Human-readable name
    
    // Computation graph (GGML-style)
    nt_tensor_t*    src[NT_MAX_SRC];    // Source tensors
    uint16_t        op;                 // Operation that created this
    int32_t         op_params[8];       // Operation parameters
    
    // Autograd (PyTorch-style)
    nt_tensor_t*    grad;               // Gradient tensor
    void*           grad_fn;            // Backward function
    uint32_t        grad_version;       // Version counter for in-place ops
    
    // Echo state (reservoir computing)
    nt_tensor_t*    reservoir_state;    // Hidden state
    float           spectral_radius;    // Echo state property
    float           leaking_rate;       // Leaky integration
    
    // Hypergraph participation
    nt_edge_t**     edges;              // Hyperedges this tensor belongs to
    uint32_t        n_edges;
    
    // Quantization info
    float           scale;              // Quantization scale
    int32_t         zero_point;         // Quantization zero point
    nt_tensor_t*    absmax;             // Per-block absmax for k-quants
    
    // Nested tensor info
    nt_tensor_t**   nested_tensors;     // Array of nested tensors
    uint32_t        n_nested;
    int32_t*        nested_offsets;     // Offsets for jagged arrays
    
    // Provenance/lineage
    uint64_t        creation_time;      // Timestamp
    uint32_t        creation_thread;    // Thread that created it
    nt_context_t*   context;            // Owning context
};

// ============================================================================
// TENSOR OPERATIONS: Core API
// ============================================================================

// Lifecycle
nt_tensor_t* nt_tensor_new(nt_context_t* ctx, nt_dtype_t dtype, 
                           int ndim, const int32_t* shape);
nt_tensor_t* nt_tensor_new_1d(nt_context_t* ctx, nt_dtype_t dtype, int32_t n);
nt_tensor_t* nt_tensor_new_2d(nt_context_t* ctx, nt_dtype_t dtype, int32_t r, int32_t c);
nt_tensor_t* nt_tensor_new_3d(nt_context_t* ctx, nt_dtype_t dtype, int32_t d0, int32_t d1, int32_t d2);
nt_tensor_t* nt_tensor_new_4d(nt_context_t* ctx, nt_dtype_t dtype, int32_t d0, int32_t d1, int32_t d2, int32_t d3);

void         nt_tensor_incref(nt_tensor_t* t);
void         nt_tensor_decref(nt_tensor_t* t);

// Views (no copy)
nt_tensor_t* nt_tensor_view(nt_tensor_t* t);
nt_tensor_t* nt_tensor_view_1d(nt_tensor_t* t, int32_t n, int32_t offset);
nt_tensor_t* nt_tensor_view_2d(nt_tensor_t* t, int32_t r, int32_t c, int32_t offset);
nt_tensor_t* nt_tensor_reshape(nt_tensor_t* t, int ndim, const int32_t* shape);
nt_tensor_t* nt_tensor_transpose(nt_tensor_t* t, int dim0, int dim1);
nt_tensor_t* nt_tensor_permute(nt_tensor_t* t, const int* dims);

// Properties
static inline int64_t nt_tensor_nelems(const nt_tensor_t* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->ne[i];
    return n;
}

static inline size_t nt_tensor_nbytes(const nt_tensor_t* t) {
    return nt_tensor_nelems(t) * nt_dtype_bytes(t->dtype);
}

static inline bool nt_tensor_is_contiguous(const nt_tensor_t* t) {
    return (t->flags & NT_FLAG_CONTIGUOUS) != 0;
}

static inline bool nt_tensor_same_shape(const nt_tensor_t* a, const nt_tensor_t* b) {
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; i++) {
        if (a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

// Data access
void* nt_tensor_data(nt_tensor_t* t);
const void* nt_tensor_data_const(const nt_tensor_t* t);

// Element access (with bounds checking in debug builds)
float nt_tensor_get_f32(const nt_tensor_t* t, int64_t idx);
void  nt_tensor_set_f32(nt_tensor_t* t, int64_t idx, float val);

// Multi-dimensional indexing
float nt_tensor_get_f32_nd(const nt_tensor_t* t, const int32_t* indices);
void  nt_tensor_set_f32_nd(nt_tensor_t* t, const int32_t* indices, float val);

#endif // NTTESHGNN_TENSOR_H
```

---

## Layer 2: The Type System (Algebraic Types for Tensors)

This is where we go beyond existing libraries - adding a proper type system:

```c
// ntteshgnn_typesystem.h - Algebraic type system for tensors
// Inspired by: Rust, Haskell, dependent types, shaped arrays

#ifndef NTTESHGNN_TYPESYSTEM_H
#define NTTESHGNN_TYPESYSTEM_H

#include "ntteshgnn_tensor.h"

// ============================================================================
// TYPE CONSTRUCTORS: Building complex types from simple ones
// ============================================================================

// A tensor type is more than just dtype - it includes shape constraints
typedef struct nt_type {
    nt_dtype_t      dtype;          // Element type
    
    // Shape specification (can be concrete or symbolic)
    enum {
        NT_SHAPE_CONCRETE,          // Fixed shape [3, 4, 5]
        NT_SHAPE_SYMBOLIC,          // Named dimensions [N, C, H, W]
        NT_SHAPE_BROADCAST,         // Broadcastable [*, *, H, W]
        NT_SHAPE_NESTED,            // Nested tensor type
    } shape_kind;
    
    union {
        // Concrete shape
        struct {
            int32_t dims[NT_MAX_DIMS];
            uint8_t ndim;
        } concrete;
        
        // Symbolic shape (for type checking)
        struct {
            const char* dim_names[NT_MAX_DIMS];
            uint8_t ndim;
        } symbolic;
        
        // Nested type
        struct {
            struct nt_type* inner_type;
            uint32_t max_depth;
        } nested;
    };
    
    // Constraints
    uint32_t        constraints;    // NT_CONSTRAINT_*
} nt_type_t;

// Type constraints (can be combined)
#define NT_CONSTRAINT_NONE          0
#define NT_CONSTRAINT_CONTIGUOUS    (1 << 0)
#define NT_CONSTRAINT_NORMALIZED    (1 << 1)    // Values in [0, 1] or [-1, 1]
#define NT_CONSTRAINT_POSITIVE      (1 << 2)    // All values > 0
#define NT_CONSTRAINT_FINITE        (1 << 3)    // No NaN/Inf
#define NT_CONSTRAINT_INTEGER       (1 << 4)    // Integer values only
#define NT_CONSTRAINT_SORTED        (1 << 5)    // Sorted along last dim
#define NT_CONSTRAINT_UNIQUE        (1 << 6)    // No duplicate values
#define NT_CONSTRAINT_PROBABILITY   (1 << 7)    // Sum to 1 along last dim

// ============================================================================
// TYPE-SAFE OPERATIONS: Operations with type signatures
// ============================================================================

// Operation type signature
typedef struct nt_op_signature {
    const char*     name;           // "matmul", "relu", etc.
    
    // Input types
    nt_type_t*      input_types;
    uint8_t         n_inputs;
    
    // Output type (can be derived from inputs)
    nt_type_t       output_type;
    
    // Type derivation function (for polymorphic ops)
    nt_type_t (*derive_output_type)(const nt_type_t* inputs, uint8_t n);
    
    // Constraint propagation
    uint32_t (*derive_constraints)(const nt_type_t* inputs, uint8_t n);
} nt_op_signature_t;

// Example: Matrix multiply signature
// matmul : Tensor[M, K] × Tensor[K, N] → Tensor[M, N]
static inline nt_type_t nt_derive_matmul_type(const nt_type_t* inputs, uint8_t n) {
    // Check: inputs[0].dims[-1] == inputs[1].dims[0]
    nt_type_t out = {
        .dtype = inputs[0].dtype,
        .shape_kind = NT_SHAPE_CONCRETE,
    };
    out.concrete.dims[0] = inputs[0].concrete.dims[0];  // M
    out.concrete.dims[1] = inputs[1].concrete.dims[1];  // N
    out.concrete.ndim = 2;
    return out;
}

// ============================================================================
// NESTED TENSOR TYPE: Tensors containing tensors
// ============================================================================

// A nested tensor is a tensor where each element is itself a tensor
// This is useful for:
// - Ragged/jagged arrays (variable-length sequences)
// - Graphs with variable-degree nodes
// - Hierarchical structures

typedef struct nt_nested_tensor {
    nt_tensor_t     base;           // Base tensor structure
    
    // Nesting structure
    nt_tensor_t**   elements;       // Array of nested tensors
    int32_t*        offsets;        // CSR-style offsets for ragged
    uint32_t        n_elements;
    
    // Type info
    nt_type_t       element_type;   // Type of nested elements
    
    // Optional: packed representation
    nt_tensor_t*    packed_data;    // All elements concatenated
    nt_tensor_t*    packed_offsets; // Offsets into packed data
} nt_nested_tensor_t;

// Create nested tensor from list of tensors
nt_nested_tensor_t* nt_nested_from_tensors(nt_context_t* ctx, 
                                            nt_tensor_t** tensors, 
                                            uint32_t n);

// Create nested tensor from packed + offsets (efficient)
nt_nested_tensor_t* nt_nested_from_packed(nt_context_t* ctx,
                                           nt_tensor_t* packed,
                                           nt_tensor_t* offsets);

// Access element
nt_tensor_t* nt_nested_get(nt_nested_tensor_t* nt, uint32_t idx);

// Iterate
typedef struct nt_nested_iter {
    nt_nested_tensor_t* nested;
    uint32_t current_idx;
} nt_nested_iter_t;

nt_nested_iter_t nt_nested_iter_begin(nt_nested_tensor_t* nt);
bool nt_nested_iter_valid(const nt_nested_iter_t* it);
nt_tensor_t* nt_nested_iter_get(const nt_nested_iter_t* it);
void nt_nested_iter_next(nt_nested_iter_t* it);

// ============================================================================
// TYPE CHECKING: Verify operations at compile/runtime
// ============================================================================

typedef enum {
    NT_TYPE_OK = 0,
    NT_TYPE_ERR_SHAPE_MISMATCH,
    NT_TYPE_ERR_DTYPE_MISMATCH,
    NT_TYPE_ERR_CONSTRAINT_VIOLATION,
    NT_TYPE_ERR_BROADCAST_FAIL,
    NT_TYPE_ERR_NESTED_DEPTH,
} nt_type_error_t;

// Check if tensor matches type
nt_type_error_t nt_type_check(const nt_tensor_t* t, const nt_type_t* expected);

// Check if operation can be applied
nt_type_error_t nt_op_type_check(const nt_op_signature_t* op,
                                  const nt_tensor_t** inputs,
                                  uint8_t n_inputs);

// Get human-readable error
const char* nt_type_error_str(nt_type_error_t err);

#endif // NTTESHGNN_TYPESYSTEM_H
```

---

## Layer 3: The Hypergraph Structure

```c
// ntteshgnn_hypergraph.h - Hypergraph for computation and relationships
// Unlike DAGs, hyperedges can connect multiple tensors simultaneously

#ifndef NTTESHGNN_HYPERGRAPH_H
#define NTTESHGNN_HYPERGRAPH_H

#include "ntteshgnn_tensor.h"

// ============================================================================
// HYPEREDGE: N-ary relationship between tensors
// ============================================================================

typedef enum {
    // Computational edges (produce outputs)
    NT_EDGE_COMPUTE,        // Standard compute op
    NT_EDGE_REDUCE,         // Reduction op
    NT_EDGE_SCATTER,        // Scatter op
    NT_EDGE_GATHER,         // Gather op
    
    // Structural edges (relationships without computation)
    NT_EDGE_VIEW,           // View relationship
    NT_EDGE_GRADIENT,       // Gradient relationship
    NT_EDGE_CHECKPOINT,     // Checkpointing boundary
    
    // Echo state edges (reservoir dynamics)
    NT_EDGE_RESERVOIR,      // Reservoir connection
    NT_EDGE_FEEDBACK,       // Feedback loop
    NT_EDGE_READOUT,        // Readout projection
    
    // Hypergraph-specific
    NT_EDGE_ATTENTION,      // Multi-head attention (Q, K, V → O)
    NT_EDGE_MESSAGE,        // Message passing (src[], dst[], edge_attr → msg)
    NT_EDGE_AGGREGATE,      // Aggregation (msg[], idx[] → node_features)
} nt_edge_kind_t;

struct nt_edge {
    nt_edge_kind_t  kind;           // Edge type
    
    // Connected tensors (inputs and outputs)
    nt_tensor_t**   inputs;         // Input tensors
    uint16_t        n_inputs;
    
    nt_tensor_t**   outputs;        // Output tensors  
    uint16_t        n_outputs;
    
    // Edge attributes (learnable or fixed)
    nt_tensor_t*    edge_weight;    // Optional edge weight
    nt_tensor_t*    edge_attr;      // Optional edge attributes
    
    // Operation info
    uint16_t        op_code;        // Operation to perform
    int32_t         op_params[8];   // Operation parameters
    
    // Execution metadata
    uint32_t        priority;       // Scheduling priority
    uint32_t        group_id;       // For parallel execution
    
    // Gradient info
    struct nt_edge* grad_edge;      // Corresponding gradient edge
    bool            requires_grad;
    
    // Hypergraph linkage
    struct nt_edge* prev;           // Previous in edge list
    struct nt_edge* next;           // Next in edge list
    uint64_t        edge_id;        // Unique identifier
};

// ============================================================================
// HYPERGRAPH: Container for nodes (tensors) and hyperedges
// ============================================================================

struct nt_graph {
    // Nodes (tensors)
    nt_tensor_t**   nodes;
    uint32_t        n_nodes;
    uint32_t        nodes_capacity;
    
    // Edges (hyperedges)
    nt_edge_t*      edges_head;     // Linked list of edges
    nt_edge_t*      edges_tail;
    uint32_t        n_edges;
    
    // Execution order (topologically sorted)
    nt_edge_t**     exec_order;
    uint32_t        exec_order_len;
    bool            exec_order_valid;
    
    // Memory planning
    size_t          peak_memory;
    nt_tensor_t**   checkpoints;    // Tensors to keep in memory
    uint32_t        n_checkpoints;
    
    // Echo state configuration
    float           global_spectral_radius;
    float           input_scaling;
    float           reservoir_density;
    
    // Context
    nt_context_t*   ctx;
};

// Graph construction
nt_graph_t* nt_graph_new(nt_context_t* ctx);
void        nt_graph_free(nt_graph_t* g);

// Add nodes
uint32_t    nt_graph_add_node(nt_graph_t* g, nt_tensor_t* t);

// Add edges (hyperedges)
nt_edge_t*  nt_graph_add_edge(nt_graph_t* g, nt_edge_kind_t kind,
                               nt_tensor_t** inputs, uint16_t n_in,
                               nt_tensor_t** outputs, uint16_t n_out,
                               uint16_t op_code);

// Convenience: Add compute edge (GGML-style)
nt_edge_t*  nt_graph_add_compute(nt_graph_t* g, uint16_t op,
                                  nt_tensor_t* dst,
                                  nt_tensor_t* src0,
                                  nt_tensor_t* src1);

// Topological sort for execution
bool        nt_graph_build_exec_order(nt_graph_t* g);

// Execute graph
void        nt_graph_compute(nt_graph_t* g, int n_threads);

// Memory optimization
size_t      nt_graph_plan_memory(nt_graph_t* g);
void        nt_graph_add_checkpoint(nt_graph_t* g, nt_tensor_t* t);

// ============================================================================
// ATTENTION HYPEREDGE: Special case for transformers
// ============================================================================

// Attention creates a 4-way hyperedge: Q, K, V → Output
// With optional edge attributes: attention mask, position encoding

nt_edge_t* nt_graph_add_attention(nt_graph_t* g,
                                   nt_tensor_t* Q,
                                   nt_tensor_t* K, 
                                   nt_tensor_t* V,
                                   nt_tensor_t* output,
                                   nt_tensor_t* mask,      // Optional
                                   int n_heads,
                                   float scale);

// ============================================================================
// MESSAGE PASSING HYPEREDGE: For GNNs
// ============================================================================

// Message passing: edge connects all source nodes to all destination nodes
// through an edge attribute tensor

nt_edge_t* nt_graph_add_message_pass(nt_graph_t* g,
                                      nt_tensor_t* node_features,
                                      nt_tensor_t* edge_index,    // [2, E]
                                      nt_tensor_t* edge_attr,     // [E, D]
                                      nt_tensor_t* output,
                                      uint16_t aggregation);      // sum, mean, max

#endif // NTTESHGNN_HYPERGRAPH_H
```

---

## Layer 4: Echo State Network Integration

```c
// ntteshgnn_echostate.h - Reservoir computing primitives
// Echo State Networks provide RNN-like dynamics without backprop through time

#ifndef NTTESHGNN_ECHOSTATE_H
#define NTTESHGNN_ECHOSTATE_H

#include "ntteshgnn_tensor.h"
#include "ntteshgnn_hypergraph.h"

// ============================================================================
// RESERVOIR: The echo state container
// ============================================================================

typedef struct nt_reservoir {
    // State tensor
    nt_tensor_t*    state;          // [batch, reservoir_size]
    
    // Weight matrices (sparse, fixed after initialization)
    nt_tensor_t*    W_in;           // Input weights [input_size, reservoir_size]
    nt_tensor_t*    W_res;          // Reservoir weights [reservoir_size, reservoir_size]
    nt_tensor_t*    W_fb;           // Feedback weights [output_size, reservoir_size] (optional)
    
    // Hyperparameters
    float           spectral_radius;    // Largest eigenvalue of W_res
    float           input_scaling;      // Scale factor for inputs
    float           leaking_rate;       // α in state update
    float           density;            // Sparsity of W_res
    
    // Activation function
    enum {
        NT_ESN_TANH,
        NT_ESN_RELU,
        NT_ESN_SIGMOID,
        NT_ESN_IDENTITY,
    } activation;
    
    // Readout (this part IS trained)
    nt_tensor_t*    W_out;          // Output weights [reservoir_size, output_size]
    nt_tensor_t*    b_out;          // Output bias [output_size]
    
    // Configuration
    bool            use_feedback;
    bool            use_bias;
    uint32_t        warmup_steps;   // Steps before collecting states
    
} nt_reservoir_t;

// Create reservoir with given dimensions
nt_reservoir_t* nt_reservoir_new(nt_context_t* ctx,
                                  int input_size,
                                  int reservoir_size,
                                  int output_size,
                                  float spectral_radius,
                                  float density);

// Initialize reservoir weights (various strategies)
void nt_reservoir_init_random(nt_reservoir_t* res, uint64_t seed);
void nt_reservoir_init_cycle(nt_reservoir_t* res);           // Cycle reservoir
void nt_reservoir_init_rodan(nt_reservoir_t* res);           // Rodan's simple architecture
void nt_reservoir_init_orthogonal(nt_reservoir_t* res);      // Orthogonal initialization

// State update: x(t+1) = (1-α)x(t) + α·f(W_in·u(t) + W_res·x(t) + W_fb·y(t-1))
void nt_reservoir_update(nt_reservoir_t* res,
                          const nt_tensor_t* input,     // [batch, input_size]
                          const nt_tensor_t* prev_out); // [batch, output_size] or NULL

// Get current state
nt_tensor_t* nt_reservoir_get_state(nt_reservoir_t* res);

// Compute output: y(t) = W_out · x(t) + b_out
nt_tensor_t* nt_reservoir_compute_output(nt_reservoir_t* res);

// Full forward pass
nt_tensor_t* nt_reservoir_forward(nt_reservoir_t* res,
                                   const nt_tensor_t* input,
                                   const nt_tensor_t* prev_out);

// Train readout weights (ridge regression, no backprop needed)
void nt_reservoir_train_readout(nt_reservoir_t* res,
                                 const nt_tensor_t* states,     // [T, batch, reservoir_size]
                                 const nt_tensor_t* targets,    // [T, batch, output_size]
                                 float ridge_alpha);

// ============================================================================
// ECHO STATE GRAPH EDGE: Integrate reservoir into hypergraph
// ============================================================================

// Add reservoir as a special hyperedge
nt_edge_t* nt_graph_add_reservoir(nt_graph_t* g,
                                   nt_reservoir_t* reservoir,
                                   nt_tensor_t* input_sequence,
                                   nt_tensor_t* output_sequence);

// ============================================================================
// DEEP ECHO STATE: Stacked reservoirs
// ============================================================================

typedef struct nt_deep_esn {
    nt_reservoir_t**    layers;
    uint32_t            n_layers;
    
    // Inter-layer connections
    bool                use_skip_connections;
    nt_tensor_t**       skip_weights;
    
} nt_deep_esn_t;

nt_deep_esn_t* nt_deep_esn_new(nt_context_t* ctx,
                                int input_size,
                                const int* reservoir_sizes,
                                int n_layers,
                                int output_size);

nt_tensor_t* nt_deep_esn_forward(nt_deep_esn_t* esn,
                                  const nt_tensor_t* input);

#endif // NTTESHGNN_ECHOSTATE_H
```

---

## Layer 5: The Context and Memory Management

```c
// ntteshgnn_context.h - Memory management and execution context
// Combines GGML's bump allocator with PyTorch's reference counting

#ifndef NTTESHGNN_CONTEXT_H
#define NTTESHGNN_CONTEXT_H

#include "ntteshgnn_tensor.h"

// ============================================================================
// ALLOCATOR INTERFACE: Abstract memory allocation
// ============================================================================

typedef struct nt_allocator {
    void* (*alloc)(void* ctx, size_t size, size_t alignment);
    void* (*realloc)(void* ctx, void* ptr, size_t old_size, size_t new_size);
    void  (*free)(void* ctx, void* ptr, size_t size);
    void* ctx;
    
    // Statistics
    size_t total_allocated;
    size_t peak_allocated;
    size_t n_allocations;
} nt_allocator_t;

// Built-in allocators
nt_allocator_t* nt_allocator_system(void);      // malloc/free
nt_allocator_t* nt_allocator_pool(size_t size); // Pool allocator
nt_allocator_t* nt_allocator_bump(size_t size); // GGML-style bump
nt_allocator_t* nt_allocator_arena(void);       // Arena allocator

// ============================================================================
// CONTEXT: Owns tensors and manages their lifecycle
// ============================================================================

typedef struct nt_context_params {
    // Memory configuration
    size_t          mem_size;           // Total memory budget
    nt_allocator_t* allocator;          // Custom allocator (NULL = default)
    
    // Device configuration
    nt_device_t     default_device;     // Where to allocate tensors
    uint16_t        default_device_id;  // Device index
    
    // Behavior flags
    bool            track_memory;       // Detailed memory tracking
    bool            check_nan;          // Check for NaN in debug builds
    bool            deterministic;      // Reproducible execution
    bool            allow_growth;       // Allow memory to grow beyond budget
} nt_context_params_t;

struct nt_context {
    // Memory management
    nt_allocator_t* allocator;
    size_t          mem_budget;
    size_t          mem_used;
    
    // Tensor registry (for lifecycle management)
    nt_tensor_t**   tensors;
    uint32_t        n_tensors;
    uint32_t        tensors_capacity;
    
    // Graph registry
    nt_graph_t**    graphs;
    uint32_t        n_graphs;
    
    // Device info
    nt_device_t     default_device;
    uint16_t        default_device_id;
    
    // Backend dispatch
    struct nt_backend* backends[16];    // Registered backends
    uint8_t            n_backends;
    
    // Thread pool
    struct nt_threadpool* threadpool;
    int                   n_threads;
    
    // RNG state
    uint64_t        rng_state[4];       // xoshiro256** state
    
    // Configuration
    nt_context_params_t params;
    
    // Scratch buffers (for temporary allocations)
    void*           scratch[4];
    size_t          scratch_size[4];
};

// Lifecycle
nt_context_t* nt_context_new(nt_context_params_t params);
nt_context_t* nt_context_new_default(void);
void          nt_context_free(nt_context_t* ctx);

// Memory operations
void*  nt_context_alloc(nt_context_t* ctx, size_t size, size_t align);
void   nt_context_free_mem(nt_context_t* ctx, void* ptr, size_t size);
size_t nt_context_used_memory(const nt_context_t* ctx);
void   nt_context_reset(nt_context_t* ctx);  // Free all tensors

// Tensor creation helpers
nt_tensor_t* nt_new_f32(nt_context_t* ctx, int ndim, const int32_t* shape);
nt_tensor_t* nt_new_f16(nt_context_t* ctx, int ndim, const int32_t* shape);
nt_tensor_t* nt_new_i8(nt_context_t* ctx, int ndim, const int32_t* shape);
nt_tensor_t* nt_new_q4_0(nt_context_t* ctx, int ndim, const int32_t* shape);

// Random number generation
float    nt_random_uniform(nt_context_t* ctx);
float    nt_random_normal(nt_context_t* ctx);
void     nt_random_seed(nt_context_t* ctx, uint64_t seed);

// ============================================================================
// SCRATCH BUFFER: For temporary computations
// ============================================================================

typedef struct nt_scratch {
    void*   data;
    size_t  size;
    size_t  used;
} nt_scratch_t;

nt_scratch_t nt_context_get_scratch(nt_context_t* ctx, int idx);
void*        nt_scratch_alloc(nt_scratch_t* s, size_t size, size_t align);
void         nt_scratch_reset(nt_scratch_t* s);

#endif // NTTESHGNN_CONTEXT_H
```

---

## Layer 6: Operation Dispatch (The Engine)

```c
// ntteshgnn_ops.h - Operation definitions and dispatch
// This is where computation actually happens

#ifndef NTTESHGNN_OPS_H
#define NTTESHGNN_OPS_H

#include "ntteshgnn_tensor.h"

// ============================================================================
// OPERATION CODES: The instruction set
// ============================================================================

typedef enum {
    // === Control (0x00-0x0F) ===
    NT_OP_NONE          = 0x00,
    NT_OP_DUP           = 0x01,     // Duplicate tensor
    NT_OP_CONTIGUOUS    = 0x02,     // Make contiguous
    NT_OP_CAST          = 0x03,     // Type cast
    
    // === Unary (0x10-0x2F) ===
    NT_OP_NEG           = 0x10,     // -x
    NT_OP_ABS           = 0x11,     // |x|
    NT_OP_SIGN          = 0x12,     // sign(x)
    NT_OP_SQRT          = 0x13,     // √x
    NT_OP_RSQRT         = 0x14,     // 1/√x
    NT_OP_LOG           = 0x15,     // ln(x)
    NT_OP_LOG2          = 0x16,     // log₂(x)
    NT_OP_EXP           = 0x17,     // eˣ
    NT_OP_EXP2          = 0x18,     // 2ˣ
    NT_OP_SIN           = 0x19,     // sin(x)
    NT_OP_COS           = 0x1A,     // cos(x)
    NT_OP_TANH          = 0x1B,     // tanh(x)
    NT_OP_SIGMOID       = 0x1C,     // σ(x)
    NT_OP_RELU          = 0x1D,     // max(0, x)
    NT_OP_GELU          = 0x1E,     // GELU(x)
    NT_OP_SILU          = 0x1F,     // SiLU(x) = x·σ(x)
    NT_OP_GELU_QUICK    = 0x20,     // Fast GELU approximation
    NT_OP_HARDSWISH     = 0x21,     // Hard swish
    NT_OP_ERF           = 0x22,     // Error function
    
    // === Binary (0x30-0x4F) ===
    NT_OP_ADD           = 0x30,     // a + b
    NT_OP_SUB           = 0x31,     // a - b
    NT_OP_MUL           = 0x32,     // a * b (element-wise)
    NT_OP_DIV           = 0x33,     // a / b
    NT_OP_POW           = 0x34,     // aᵇ
    NT_OP_MIN           = 0x35,     // min(a, b)
    NT_OP_MAX           = 0x36,     // max(a, b)
    NT_OP_CMP_EQ        = 0x37,     // a == b
    NT_OP_CMP_LT        = 0x38,     // a < b
    NT_OP_CMP_GT        = 0x39,     // a > b
    
    // === Reduction (0x50-0x5F) ===
    NT_OP_SUM           = 0x50,     // Σx
    NT_OP_MEAN          = 0x51,     // mean(x)
    NT_OP_VAR           = 0x52,     // var(x)
    NT_OP_STD           = 0x53,     // std(x)
    NT_OP_PROD          = 0x54,     // Πx
    NT_OP_AMAX          = 0x55,     // max(|x|)
    NT_OP_ARGMAX        = 0x56,     // argmax(x)
    NT_OP_ARGMIN        = 0x57,     // argmin(x)
    NT_OP_SOFTMAX       = 0x58,     // softmax(x)
    NT_OP_LOGSOFTMAX    = 0x59,     // log(softmax(x))
    
    // === Matrix (0x60-0x7F) ===
    NT_OP_MATMUL        = 0x60,     // A @ B
    NT_OP_GEMM          = 0x61,     // αAB + βC
    NT_OP_GEMV          = 0x62,     // αAx + βy
    NT_OP_DOT           = 0x63,     // x·y
    NT_OP_OUTER         = 0x64,     // xy^T
    NT_OP_CONV1D        = 0x65,     // 1D convolution
    NT_OP_CONV2D        = 0x66,     // 2D convolution
    NT_OP_CONV2D_DW     = 0x67,     // Depthwise conv
    NT_OP_CONV_TRANSPOSE = 0x68,    // Transposed conv
    
    // === Normalization (0x80-0x8F) ===
    NT_OP_LAYER_NORM    = 0x80,     // Layer normalization
    NT_OP_RMS_NORM      = 0x81,     // RMS normalization
    NT_OP_BATCH_NORM    = 0x82,     // Batch normalization
    NT_OP_GROUP_NORM    = 0x83,     // Group normalization
    NT_OP_INSTANCE_NORM = 0x84,     // Instance normalization
    
    // === Shape (0x90-0x9F) ===
    NT_OP_RESHAPE       = 0x90,     // Reshape
    NT_OP_TRANSPOSE     = 0x91,     // Transpose
    NT_OP_PERMUTE       = 0x92,     // Permute dimensions
    NT_OP_SQUEEZE       = 0x93,     // Remove dim of size 1
    NT_OP_UNSQUEEZE     = 0x94,     // Add dim of size 1
    NT_OP_FLATTEN       = 0x95,     // Flatten to 1D
    NT_OP_CONCAT        = 0x96,     // Concatenate
    NT_OP_SPLIT         = 0x97,     // Split
    NT_OP_STACK         = 0x98,     // Stack tensors
    NT_OP_SLICE         = 0x99,     // Slice
    NT_OP_PAD           = 0x9A,     // Padding
    
    // === Attention (0xA0-0xAF) ===
    NT_OP_ATTENTION     = 0xA0,     // Scaled dot-product attention
    NT_OP_FLASH_ATTN    = 0xA1,     // Flash attention
    NT_OP_ROPE          = 0xA2,     // Rotary position embedding
    NT_OP_ALIBI         = 0xA3,     // ALiBi position bias
    NT_OP_MQA           = 0xA4,     // Multi-query attention
    NT_OP_GQA           = 0xA5,     // Grouped-query attention
    
    // === Embedding (0xB0-0xBF) ===
    NT_OP_EMBED         = 0xB0,     // Embedding lookup
    NT_OP_ONEHOT        = 0xB1,     // One-hot encoding
    NT_OP_GATHER        = 0xB2,     // Gather
    NT_OP_SCATTER       = 0xB3,     // Scatter
    NT_OP_INDEX_SELECT  = 0xB4,     // Index select
    
    // === Quantization (0xC0-0xCF) ===
    NT_OP_QUANTIZE      = 0xC0,     // Float → Quantized
    NT_OP_DEQUANTIZE    = 0xC1,     // Quantized → Float
    NT_OP_REQUANTIZE    = 0xC2,     // Quantized → Quantized
    NT_OP_CALIBRATE     = 0xC3,     // Collect calibration stats
    
    // === Echo State (0xD0-0xDF) ===
    NT_OP_ESN_UPDATE    = 0xD0,     // Reservoir state update
    NT_OP_ESN_READOUT   = 0xD1,     // Readout projection
    NT_OP_SPECTRAL_NORM = 0xD2,     // Spectral normalization
    NT_OP_SPARSE_MATMUL = 0xD3,     // Sparse matrix multiply
    
    // === Graph Neural Network (0xE0-0xEF) ===
    NT_OP_MESSAGE       = 0xE0,     // Message computation
    NT_OP_AGGREGATE     = 0xE1,     // Message aggregation
    NT_OP_UPDATE        = 0xE2,     // Node update
    NT_OP_EDGE_CONV     = 0xE3,     // Edge convolution
    NT_OP_GAT           = 0xE4,     // Graph attention
    
    // === Special (0xF0-0xFF) ===
    NT_OP_CUSTOM        = 0xF0,     // Custom operation
    NT_OP_FUSED         = 0xF1,     // Fused operation sequence
    NT_OP_CHECKPOINT    = 0xF2,     // Gradient checkpointing
    NT_OP_DEBUG         = 0xFF,     // Debug/print
    
} nt_op_t;

// ============================================================================
// OPERATION PARAMETERS
// ============================================================================

typedef struct nt_op_params {
    // Common parameters
    int32_t axes[NT_MAX_DIMS];      // Axes for reductions, permutations
    int32_t n_axes;
    
    // Padding/stride
    int32_t pad[NT_MAX_DIMS];
    int32_t stride[NT_MAX_DIMS];
    int32_t dilation[NT_MAX_DIMS];
    
    // Scalars
    float alpha;
    float beta;
    float epsilon;
    
    // Attention params
    int32_t n_heads;
    int32_t head_dim;
    float scale;
    
    // Flags
    uint32_t flags;
    
} nt_op_params_t;

#define NT_OP_FLAG_INPLACE      (1 << 0)
#define NT_OP_FLAG_ACCUMULATE   (1 << 1)
#define NT_OP_FLAG_TRANSPOSE_A  (1 << 2)
#define NT_OP_FLAG_TRANSPOSE_B  (1 << 3)
#define NT_OP_FLAG_BROADCAST    (1 << 4)

// ============================================================================
// OPERATION DISPATCH
// ============================================================================

// Compute function signature
typedef void (*nt_compute_fn)(
    const nt_tensor_t** inputs,
    int n_inputs,
    nt_tensor_t* output,
    const nt_op_params_t* params,
    void* backend_ctx
);

// Operation registry entry
typedef struct nt_op_entry {
    nt_op_t         op;
    const char*     name;
    nt_compute_fn   compute[16];    // Per-dtype implementations
    nt_op_signature_t signature;    // Type signature
} nt_op_entry_t;

// Dispatch to appropriate implementation
void nt_compute(nt_op_t op,
                const nt_tensor_t** inputs,
                int n_inputs,
                nt_tensor_t* output,
                const nt_op_params_t* params);

// ============================================================================
// BACKEND INTERFACE
// ============================================================================

typedef struct nt_backend {
    const char*     name;           // "cpu", "cuda", "metal", "vtnpu"
    nt_device_t     device_type;
    
    // Lifecycle
    int  (*init)(struct nt_backend* be);
    void (*shutdown)(struct nt_backend* be);
    
    // Memory
    void* (*alloc)(struct nt_backend* be, size_t size, size_t align);
    void  (*free)(struct nt_backend* be, void* ptr);
    void  (*copy_to_device)(struct nt_backend* be, void* dst, const void* src, size_t n);
    void  (*copy_from_device)(struct nt_backend* be, void* dst, const void* src, size_t n);
    
    // Compute
    void (*compute)(struct nt_backend* be, nt_op_t op,
                   const nt_tensor_t** inputs, int n_inputs,
                   nt_tensor_t* output, const nt_op_params_t* params);
    
    // Synchronization
    void (*sync)(struct nt_backend* be);
    
    // Context
    void* ctx;
} nt_backend_t;

// Register/get backends
void         nt_register_backend(nt_backend_t* be);
nt_backend_t* nt_get_backend(nt_device_t device);

// Built-in backends
nt_backend_t* nt_backend_cpu(void);

#ifdef NT_ENABLE_CUDA
nt_backend_t* nt_backend_cuda(void);
#endif

#ifdef NT_ENABLE_METAL
nt_backend_t* nt_backend_metal(void);
#endif

#ifdef NT_ENABLE_VTNPU
nt_backend_t* nt_backend_vtnpu(void);
#endif

#endif // NTTESHGNN_OPS_H
```

---

## Layer 7: VTNPU Backend Mapping

```c
// ntteshgnn_vtnpu.h - VTNPU hardware backend
// Maps NTTESHGNN operations to our Virtual Tensor NPU

#ifndef NTTESHGNN_VTNPU_H
#define NTTESHGNN_VTNPU_H

#include "ntteshgnn_ops.h"

// ============================================================================
// VTNPU REGISTER MAP (Hardware Interface)
// ============================================================================

// Base addresses
#define VTNPU_REG_BASE      0x60000000
#define VTNPU_SRAM_BASE     0x61000000
#define VTNPU_CMDQ_BASE     0x62000000

// Control registers (offsets from REG_BASE)
#define VTNPU_REG_CMD       0x00    // Command register
#define VTNPU_REG_STATUS    0x04    // Status register
#define VTNPU_REG_CONFIG    0x08    // Configuration
#define VTNPU_REG_ERROR     0x0C    // Error code
#define VTNPU_REG_VERSION   0x10    // Hardware version

// Operation registers
#define VTNPU_REG_OPCODE    0x20    // Operation code
#define VTNPU_REG_FLAGS     0x24    // Operation flags
#define VTNPU_REG_SCALAR0   0x28    // Scalar param 0
#define VTNPU_REG_SCALAR1   0x2C    // Scalar param 1

// Tensor descriptor registers (8 descriptors × 32 bytes each)
#define VTNPU_REG_TD_BASE   0x100
#define VTNPU_REG_TD_STRIDE 0x20

// Per-descriptor layout
#define VTNPU_TD_ADDR       0x00    // Base address (64-bit)
#define VTNPU_TD_NE0        0x08    // Dimension 0
#define VTNPU_TD_NE1        0x0C    // Dimension 1
#define VTNPU_TD_NE2        0x10    // Dimension 2
#define VTNPU_TD_NE3        0x14    // Dimension 3
#define VTNPU_TD_NB0        0x18    // Stride 0
#define VTNPU_TD_DTYPE      0x1C    // Data type + layout

// Telemetry registers
#define VTNPU_REG_MAC_OPS_LO  0x300
#define VTNPU_REG_MAC_OPS_HI  0x304
#define VTNPU_REG_MEM_BW      0x308
#define VTNPU_REG_UTIL        0x30C
#define VTNPU_REG_TEMP        0x310
#define VTNPU_REG_POWER       0x314

// ============================================================================
// VTNPU COMMAND STRUCTURE
// ============================================================================

typedef struct __attribute__((packed)) vtnpu_cmd {
    uint16_t    opcode;             // NT_OP_* code
    uint16_t    flags;              // Operation flags
    
    uint8_t     src0_td;            // Source 0 tensor descriptor index
    uint8_t     src1_td;            // Source 1 tensor descriptor index
    uint8_t     dst_td;             // Destination tensor descriptor index
    uint8_t     aux_td;             // Auxiliary tensor descriptor index
    
    int32_t     params[4];          // Operation-specific parameters
    
} vtnpu_cmd_t;  // 24 bytes

// ============================================================================
// VTNPU BACKEND IMPLEMENTATION
// ============================================================================

typedef struct vtnpu_backend_ctx {
    // Hardware interface (memory-mapped or simulated)
    volatile uint32_t*  regs;       // Register base
    void*               sram;       // SRAM mapping
    size_t              sram_size;
    
    // SRAM allocator
    size_t              sram_used;
    
    // Command queue
    vtnpu_cmd_t*        cmdq;
    uint32_t            cmdq_head;
    uint32_t            cmdq_tail;
    uint32_t            cmdq_size;
    
    // Tensor descriptor cache
    nt_tensor_t*        td_tensors[8];  // Which tensor is in each TD slot
    
    // Telemetry
    uint64_t            total_mac_ops;
    uint64_t            total_bytes;
    
} vtnpu_backend_ctx_t;

// Backend creation
nt_backend_t* nt_backend_vtnpu_create(void);

// Low-level operations
void vtnpu_write_reg32(vtnpu_backend_ctx_t* ctx, uint32_t offset, uint32_t value);
uint32_t vtnpu_read_reg32(vtnpu_backend_ctx_t* ctx, uint32_t offset);

// Tensor descriptor management
int vtnpu_alloc_td(vtnpu_backend_ctx_t* ctx, nt_tensor_t* t);
void vtnpu_free_td(vtnpu_backend_ctx_t* ctx, int td_idx);
void vtnpu_write_td(vtnpu_backend_ctx_t* ctx, int td_idx, const nt_tensor_t* t);

// Command submission
void vtnpu_submit_cmd(vtnpu_backend_ctx_t* ctx, const vtnpu_cmd_t* cmd);
void vtnpu_flush_cmdq(vtnpu_backend_ctx_t* ctx);
void vtnpu_wait_idle(vtnpu_backend_ctx_t* ctx);

// Data transfer
void vtnpu_upload(vtnpu_backend_ctx_t* ctx, size_t offset, const void* data, size_t size);
void vtnpu_download(vtnpu_backend_ctx_t* ctx, size_t offset, void* data, size_t size);

// ============================================================================
// OPERATION MAPPING: NTTESHGNN → VTNPU
// ============================================================================

// Map NT_OP_* to VTNPU hardware opcodes
static inline uint16_t vtnpu_map_opcode(nt_op_t op) {
    // Direct mapping for most operations
    // VTNPU uses same opcode space as NTTESHGNN
    return (uint16_t)op;
}

// Check if operation is supported in hardware
static inline bool vtnpu_op_supported(nt_op_t op, nt_dtype_t dtype) {
    // VTNPU supports: FP16, BF16, INT8, Q4/Q8
    if (dtype != NT_F16 && dtype != NT_BF16 && 
        dtype != NT_I8 && dtype != NT_Q4_0 && dtype != NT_Q8_0) {
        return false;
    }
    
    // Supported operations
    switch (op) {
        case NT_OP_ADD:
        case NT_OP_MUL:
        case NT_OP_MATMUL:
        case NT_OP_RELU:
        case NT_OP_GELU:
        case NT_OP_SILU:
        case NT_OP_SOFTMAX:
        case NT_OP_LAYER_NORM:
        case NT_OP_RMS_NORM:
        case NT_OP_ROPE:
        case NT_OP_ATTENTION:
            return true;
        default:
            return false;
    }
}

#endif // NTTESHGNN_VTNPU_H
```

---

## Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              NTTESHGNN Architecture                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                         APPLICATION LAYER                                  ║ │
│  ║  ┌──────────────┬──────────────┬──────────────┬──────────────┐            ║ │
│  ║  │  LLM Loader  │  GNN Models  │  ESN Models  │  Custom Apps │            ║ │
│  ║  │  (GGUF, etc) │  (PyG-style) │  (Reservoir) │              │            ║ │
│  ║  └──────────────┴──────────────┴──────────────┴──────────────┘            ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                          GRAPH LAYER                                       ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │                    nt_graph_t (Hypergraph)                 │           ║ │
│  ║  │  • Nodes: nt_tensor_t*                                     │           ║ │
│  ║  │  • Hyperedges: nt_edge_t* (N-ary connections)              │           ║ │
│  ║  │  • Echo State: nt_reservoir_t                              │           ║ │
│  ║  │  • Execution order + memory planning                       │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                          TYPE SYSTEM                                       ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │  nt_type_t: Shape constraints, nested types, constraints   │           ║ │
│  ║  │  Type checking at graph construction time                  │           ║ │
│  ║  │  Automatic broadcasting and shape inference                │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                          TENSOR LAYER                                      ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │  nt_tensor_t (64 bytes, cache-aligned)                     │           ║ │
│  ║  │  ├── Shape [ne0..ne7] + Strides [nb0..nb7]                 │           ║ │
│  ║  │  ├── nt_storage_t* (ref-counted data)                      │           ║ │
│  ║  │  ├── nt_tensor_meta* (optional: grad, graph, echo state)   │           ║ │
│  ║  │  └── Flags: nested, view, requires_grad, etc.              │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │  nt_nested_tensor_t (Ragged/Jagged arrays)                 │           ║ │
│  ║  │  ├── Base tensor structure                                 │           ║ │
│  ║  │  ├── Array of sub-tensors                                  │           ║ │
│  ║  │  └── CSR-style offsets for efficient packing               │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                        OPERATION DISPATCH                                  ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │  nt_compute(op, inputs, n_inputs, output, params)          │           ║ │
│  ║  │  ├── Type dispatch: FP32 → BLAS, Q4 → custom kernel        │           ║ │
│  ║  │  ├── Device dispatch: CPU → OpenBLAS, VTNPU → MMIO         │           ║ │
│  ║  │  └── SIMD dispatch: AVX2 / AVX-512 / NEON / WASM SIMD      │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                          BACKEND LAYER                                     ║ │
│  ║  ┌──────────┬──────────┬──────────┬──────────┬──────────┐                 ║ │
│  ║  │   CPU    │   CUDA   │  Metal   │  VTNPU   │  Remote  │                 ║ │
│  ║  │  (BLAS,  │  (cuBLAS,│  (MPS,   │  (MMIO,  │  (gRPC,  │                 ║ │
│  ║  │   SIMD)  │   cuDNN) │   MPSGraph)  SRAM)  │   TCP)   │                 ║ │
│  ║  └──────────┴──────────┴──────────┴──────────┴──────────┘                 ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                          MEMORY LAYER                                      ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │  nt_context_t (Memory management + lifecycle)              │           ║ │
│  ║  │  ├── Allocators: bump, pool, arena, system                 │           ║ │
│  ║  │  ├── Tensor registry + garbage collection                  │           ║ │
│  ║  │  └── Scratch buffers for temporaries                       │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                     │                                           │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                         HARDWARE LAYER                                     ║ │
│  ║  ┌────────────────────────────────────────────────────────────┐           ║ │
│  ║  │  CPU: AVX2/AVX-512, NEON, WASM SIMD                        │           ║ │
│  ║  │  GPU: CUDA Tensor Cores, Metal ANE                         │           ║ │
│  ║  │  NPU: VTNPU Registers @ 0x60000000, SRAM @ 0x61000000      │           ║ │
│  ║  │  FPGA: AXI-mapped compute fabric                           │           ║ │
│  ║  └────────────────────────────────────────────────────────────┘           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

| Phase | Deliverable | Lines of Code | Status |
|-------|-------------|---------------|--------|
| 1 | Core types + tensor struct | ~800 | 📋 Ready to implement |
| 2 | Storage + context management | ~600 | 📋 Ready |
| 3 | Basic ops (add, mul, matmul) | ~1500 | 📋 Ready |
| 4 | Type system + checking | ~1000 | 📋 Ready |
| 5 | Hypergraph + edges | ~800 | 📋 Ready |
| 6 | Echo state reservoir | ~600 | 📋 Ready |
| 7 | VTNPU backend | ~1200 | 📋 Ready |
| 8 | CPU SIMD kernels | ~2000 | 📋 Ready |
| 9 | Python bindings | ~800 | 📋 Ready |
| 10 | GGUF loader | ~500 | 📋 Ready |

**Total: ~10,000 lines of C** for a complete foundation.

---

Want me to:

1. **Generate the complete C implementation** as a working codebase?
2. **Design the Python bindings** (like PyTorch's `torch._C`)?
3. **Create the SIMD kernels** for CPU (AVX2/AVX-512)?
4. **Write the Verilog RTL** for the VTNPU hardware?
5. **Build a test suite** with comprehensive coverage?

This foundation gives us a tensor engine that's as minimal as GGML, as flexible as PyTorch, as simple as TH, but with novel capabilities (nested tensors, typed operations, echo state dynamics, hypergraph structure) that none of them have.
