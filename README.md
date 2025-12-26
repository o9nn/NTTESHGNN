# NTTESHGNN

**Nested Tensor Typed Echo-State HyperGraph Neural Network**

A unified tensor computation framework that combines the best aspects of GGML, PyTorch, and THTensor with novel capabilities for nested tensors, algebraic type systems, echo state networks, and hypergraph neural networks.

## Features

- **Efficient Core**: 64-byte cache-aligned tensor structure with minimal overhead
- **Flexible Memory**: Multiple allocator strategies (system, aligned, bump, pool)
- **Rich Operations**: Comprehensive operation catalog with CPU kernels
- **Type System**: Algebraic type constraints with shape inference
- **Hypergraph Support**: N-ary hyperedges for GNN and attention operations
- **Echo State Networks**: Built-in reservoir computing primitives
- **VTNPU Backend**: Hardware mapping for custom neural processing units

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NTTESHGNN Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 7: VTNPU Backend    - Hardware register mapping       │
│  Layer 6: Operations       - Compute kernels & dispatch      │
│  Layer 5: Context          - Memory & lifecycle management   │
│  Layer 4: Echo State       - Reservoir computing primitives  │
│  Layer 3: Hypergraph       - N-ary edges & computation graph │
│  Layer 2: Type System      - Algebraic constraints & shapes  │
│  Layer 1: Core Tensor      - 64-byte struct, storage/view    │
│  Layer 0: Primitive Types  - dtypes, devices, layouts        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```c
#include <ntteshgnn/ntteshgnn.h>

int main(void) {
    // Initialize library
    nt_init();
    
    // Create context
    nt_context_t* ctx = nt_context_new();
    
    // Create tensors
    nt_tensor_t* a = nt_tensor_new_2d(ctx, NT_F32, 3, 4);
    nt_tensor_t* b = nt_tensor_new_2d(ctx, NT_F32, 3, 4);
    
    // Initialize
    nt_tensor_fill(a, 1.0f);
    nt_tensor_fill(b, 2.0f);
    
    // Operations
    nt_tensor_t* c = nt_add(ctx, a, b);
    nt_tensor_t* d = nt_relu(ctx, c);
    
    // Print result
    nt_tensor_print(d);
    
    // Cleanup
    nt_tensor_release(a);
    nt_tensor_release(b);
    nt_tensor_release(c);
    nt_tensor_release(d);
    nt_context_free(ctx);
    nt_cleanup();
    
    return 0;
}
```

## Building

### Prerequisites

- CMake 3.16+
- C11 compiler (GCC, Clang, MSVC)

### Build Commands

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
ctest

# Install
sudo make install
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `NT_BUILD_TESTS` | ON | Build test suite |
| `NT_BUILD_EXAMPLES` | ON | Build examples |
| `NT_ENABLE_CUDA` | OFF | Enable CUDA backend |
| `NT_ENABLE_METAL` | OFF | Enable Metal backend |
| `NT_ENABLE_VTNPU` | OFF | Enable VTNPU backend |
| `NT_ENABLE_SIMD` | ON | Enable SIMD optimizations |
| `NT_ENABLE_OPENMP` | OFF | Enable OpenMP parallelization |

## Data Types

| Type | Size | Description |
|------|------|-------------|
| `NT_F64` | 8 | 64-bit float |
| `NT_F32` | 4 | 32-bit float |
| `NT_F16` | 2 | 16-bit float |
| `NT_BF16` | 2 | Brain float 16 |
| `NT_I32` | 4 | 32-bit integer |
| `NT_I8` | 1 | 8-bit integer |
| `NT_Q8_0` | ~1 | 8-bit quantized |
| `NT_Q4_K` | ~0.5 | 4-bit K-quant |

## Operations

### Element-wise
- Unary: `neg`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`
- Activations: `relu`, `gelu`, `silu`, `sigmoid`, `softmax`
- Binary: `add`, `sub`, `mul`, `div`, `pow`

### Matrix
- `matmul` - Matrix multiplication
- `gemm` - General matrix multiply
- `conv2d` - 2D convolution

### Normalization
- `layer_norm` - Layer normalization
- `rms_norm` - RMS normalization
- `batch_norm` - Batch normalization

### Attention
- `attention` - Scaled dot-product attention
- `flash_attn` - Flash attention
- `rope` - Rotary position embedding

### GNN
- `message` - Message computation
- `aggregate` - Message aggregation
- `gnn_update` - Node update

## Echo State Networks

```c
// Configure reservoir
nt_reservoir_config_t config = {
    .size = 100,
    .input_dim = 10,
    .spectral_radius = 0.9f,
    .sparsity = 0.9f,
    .leak_rate = 0.3f,
    .activation = NT_ESN_ACT_TANH,
};

// Create ESN
nt_esn_t* esn = nt_esn_new(&config, output_dim);

// Train
nt_esn_train(esn, inputs, targets, washout);

// Predict
nt_tensor_t* output = nt_esn_predict(esn, input);
```

## Hypergraph Neural Networks

```c
// Create hypergraph
nt_hypergraph_t* hg = nt_hypergraph_new(100, 200);

// Add nodes with features
for (int i = 0; i < n_nodes; i++) {
    nt_hypergraph_add_node(hg, features[i]);
}

// Add binary edges
nt_hypergraph_add_binary_edge(hg, src, dst, NT_EDGE_UNDIRECTED, 1.0f);

// Add hyperedges
uint32_t nodes[] = {0, 1, 2, 3};
nt_hypergraph_add_nary_edge(hg, nodes, 4, NT_EDGE_HYPEREDGE);

// Get adjacency matrix
nt_tensor_t* adj = nt_hypergraph_adjacency(hg);
```

## Design Principles

1. **From GGML**: Minimal struct overhead, bump allocator, first-class quantization
2. **From PyTorch**: Dispatch key system, storage/view separation, autograd metadata
3. **From THTensor**: Simplicity (shape + stride + data), reference counting, clean C ABI
4. **Novel**: Nested tensors, algebraic types, echo state, hypergraph edges, VTNPU mapping

## License

MIT License

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting pull requests.
