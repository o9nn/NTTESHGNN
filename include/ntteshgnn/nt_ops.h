/**
 * @file nt_ops.h
 * @brief NTTESHGNN - Operations Catalog (Layer 6)
 * 
 * Complete catalog of tensor operations with dispatch to backends.
 * Operations are organized by category and support multiple dtypes.
 * 
 * Design principles:
 * - From GGML: Simple operation codes, efficient dispatch
 * - From PyTorch: Rich operation set, broadcasting support
 * - Novel: Echo state and GNN operations as first-class citizens
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_OPS_H
#define NTTESHGNN_OPS_H

#include "nt_tensor.h"
#include "nt_typesystem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * OPERATION CODES
 * ============================================================================ */

typedef enum nt_op {
    /* === No operation (0x00) === */
    NT_OP_NONE          = 0x00,     /**< No operation */
    
    /* === Unary element-wise (0x01-0x1F) === */
    NT_OP_NEG           = 0x01,     /**< -x */
    NT_OP_ABS           = 0x02,     /**< |x| */
    NT_OP_SIGN          = 0x03,     /**< sign(x) */
    NT_OP_SQRT          = 0x04,     /**< √x */
    NT_OP_RSQRT         = 0x05,     /**< 1/√x */
    NT_OP_SQUARE        = 0x06,     /**< x² */
    NT_OP_CUBE          = 0x07,     /**< x³ */
    NT_OP_EXP           = 0x08,     /**< eˣ */
    NT_OP_LOG           = 0x09,     /**< ln(x) */
    NT_OP_LOG2          = 0x0A,     /**< log₂(x) */
    NT_OP_LOG10         = 0x0B,     /**< log₁₀(x) */
    NT_OP_SIN           = 0x0C,     /**< sin(x) */
    NT_OP_COS           = 0x0D,     /**< cos(x) */
    NT_OP_TAN           = 0x0E,     /**< tan(x) */
    NT_OP_TANH          = 0x0F,     /**< tanh(x) */
    NT_OP_SIGMOID       = 0x10,     /**< σ(x) = 1/(1+e⁻ˣ) */
    NT_OP_RELU          = 0x11,     /**< max(0, x) */
    NT_OP_GELU          = 0x12,     /**< GELU activation */
    NT_OP_SILU          = 0x13,     /**< SiLU/Swish: x·σ(x) */
    NT_OP_LEAKY_RELU    = 0x14,     /**< Leaky ReLU */
    NT_OP_ELU           = 0x15,     /**< ELU */
    NT_OP_SOFTPLUS      = 0x16,     /**< log(1 + eˣ) */
    NT_OP_MISH          = 0x17,     /**< x·tanh(softplus(x)) */
    NT_OP_HARDSWISH     = 0x18,     /**< Hard Swish */
    NT_OP_HARDSIGMOID   = 0x19,     /**< Hard Sigmoid */
    NT_OP_RECIPROCAL    = 0x1A,     /**< 1/x */
    NT_OP_FLOOR         = 0x1B,     /**< floor(x) */
    NT_OP_CEIL          = 0x1C,     /**< ceil(x) */
    NT_OP_ROUND         = 0x1D,     /**< round(x) */
    NT_OP_CLAMP         = 0x1E,     /**< clamp(x, min, max) */
    NT_OP_CAST          = 0x1F,     /**< Type cast */
    
    /* === Binary element-wise (0x20-0x3F) === */
    NT_OP_ADD           = 0x20,     /**< x + y */
    NT_OP_SUB           = 0x21,     /**< x - y */
    NT_OP_MUL           = 0x22,     /**< x * y */
    NT_OP_DIV           = 0x23,     /**< x / y */
    NT_OP_POW           = 0x24,     /**< xʸ */
    NT_OP_MAX           = 0x25,     /**< max(x, y) */
    NT_OP_MIN           = 0x26,     /**< min(x, y) */
    NT_OP_MOD           = 0x27,     /**< x mod y */
    NT_OP_FMOD          = 0x28,     /**< fmod(x, y) */
    NT_OP_ATAN2         = 0x29,     /**< atan2(y, x) */
    NT_OP_HYPOT         = 0x2A,     /**< √(x² + y²) */
    NT_OP_COPYSIGN      = 0x2B,     /**< copysign(x, y) */
    
    /* === Comparison (0x30-0x3F) === */
    NT_OP_EQ            = 0x30,     /**< x == y */
    NT_OP_NE            = 0x31,     /**< x != y */
    NT_OP_LT            = 0x32,     /**< x < y */
    NT_OP_LE            = 0x33,     /**< x <= y */
    NT_OP_GT            = 0x34,     /**< x > y */
    NT_OP_GE            = 0x35,     /**< x >= y */
    NT_OP_ISNAN         = 0x36,     /**< isnan(x) */
    NT_OP_ISINF         = 0x37,     /**< isinf(x) */
    NT_OP_WHERE         = 0x38,     /**< where(cond, x, y) */
    
    /* === Logical (0x40-0x4F) === */
    NT_OP_AND           = 0x40,     /**< x && y */
    NT_OP_OR            = 0x41,     /**< x || y */
    NT_OP_XOR           = 0x42,     /**< x ^ y */
    NT_OP_NOT           = 0x43,     /**< !x */
    NT_OP_BITAND        = 0x44,     /**< x & y */
    NT_OP_BITOR         = 0x45,     /**< x | y */
    NT_OP_BITXOR        = 0x46,     /**< x ^ y (bitwise) */
    NT_OP_BITNOT        = 0x47,     /**< ~x */
    NT_OP_LSHIFT        = 0x48,     /**< x << y */
    NT_OP_RSHIFT        = 0x49,     /**< x >> y */
    
    /* === Reduction (0x50-0x5F) === */
    NT_OP_SUM           = 0x50,     /**< Σx */
    NT_OP_MEAN          = 0x51,     /**< mean(x) */
    NT_OP_VAR           = 0x52,     /**< var(x) */
    NT_OP_STD           = 0x53,     /**< std(x) */
    NT_OP_PROD          = 0x54,     /**< Πx */
    NT_OP_AMAX          = 0x55,     /**< max(|x|) */
    NT_OP_AMIN          = 0x56,     /**< min(|x|) */
    NT_OP_ARGMAX        = 0x57,     /**< argmax(x) */
    NT_OP_ARGMIN        = 0x58,     /**< argmin(x) */
    NT_OP_SOFTMAX       = 0x59,     /**< softmax(x) */
    NT_OP_LOGSOFTMAX    = 0x5A,     /**< log(softmax(x)) */
    NT_OP_NORM          = 0x5B,     /**< ||x||_p */
    NT_OP_CUMSUM        = 0x5C,     /**< Cumulative sum */
    NT_OP_CUMPROD       = 0x5D,     /**< Cumulative product */
    NT_OP_ALL           = 0x5E,     /**< all(x) */
    NT_OP_ANY           = 0x5F,     /**< any(x) */
    
    /* === Matrix operations (0x60-0x7F) === */
    NT_OP_MATMUL        = 0x60,     /**< A @ B */
    NT_OP_GEMM          = 0x61,     /**< αAB + βC */
    NT_OP_GEMV          = 0x62,     /**< αAx + βy */
    NT_OP_DOT           = 0x63,     /**< x·y */
    NT_OP_OUTER         = 0x64,     /**< xy^T */
    NT_OP_CONV1D        = 0x65,     /**< 1D convolution */
    NT_OP_CONV2D        = 0x66,     /**< 2D convolution */
    NT_OP_CONV2D_DW     = 0x67,     /**< Depthwise conv */
    NT_OP_CONV_TRANSPOSE = 0x68,    /**< Transposed conv */
    NT_OP_POOL_MAX      = 0x69,     /**< Max pooling */
    NT_OP_POOL_AVG      = 0x6A,     /**< Average pooling */
    NT_OP_POOL_ADAPTIVE = 0x6B,     /**< Adaptive pooling */
    NT_OP_UPSAMPLE      = 0x6C,     /**< Upsampling */
    NT_OP_EINSUM        = 0x6D,     /**< Einstein summation */
    NT_OP_TENSORDOT     = 0x6E,     /**< Tensor dot product */
    NT_OP_KRON          = 0x6F,     /**< Kronecker product */
    NT_OP_TRACE         = 0x70,     /**< Matrix trace */
    NT_OP_DET           = 0x71,     /**< Determinant */
    NT_OP_INV           = 0x72,     /**< Matrix inverse */
    NT_OP_SOLVE         = 0x73,     /**< Linear solve */
    NT_OP_SVD           = 0x74,     /**< SVD */
    NT_OP_EIG           = 0x75,     /**< Eigendecomposition */
    NT_OP_QR            = 0x76,     /**< QR decomposition */
    NT_OP_CHOLESKY      = 0x77,     /**< Cholesky decomposition */
    NT_OP_LU            = 0x78,     /**< LU decomposition */
    
    /* === Normalization (0x80-0x8F) === */
    NT_OP_LAYER_NORM    = 0x80,     /**< Layer normalization */
    NT_OP_RMS_NORM      = 0x81,     /**< RMS normalization */
    NT_OP_BATCH_NORM    = 0x82,     /**< Batch normalization */
    NT_OP_GROUP_NORM    = 0x83,     /**< Group normalization */
    NT_OP_INSTANCE_NORM = 0x84,     /**< Instance normalization */
    NT_OP_L2_NORM       = 0x85,     /**< L2 normalization */
    
    /* === Shape operations (0x90-0x9F) === */
    NT_OP_RESHAPE       = 0x90,     /**< Reshape */
    NT_OP_TRANSPOSE     = 0x91,     /**< Transpose */
    NT_OP_PERMUTE       = 0x92,     /**< Permute dimensions */
    NT_OP_SQUEEZE       = 0x93,     /**< Remove dim of size 1 */
    NT_OP_UNSQUEEZE     = 0x94,     /**< Add dim of size 1 */
    NT_OP_FLATTEN       = 0x95,     /**< Flatten to 1D */
    NT_OP_CONCAT        = 0x96,     /**< Concatenate */
    NT_OP_SPLIT         = 0x97,     /**< Split */
    NT_OP_STACK         = 0x98,     /**< Stack tensors */
    NT_OP_SLICE         = 0x99,     /**< Slice */
    NT_OP_PAD           = 0x9A,     /**< Padding */
    NT_OP_TILE          = 0x9B,     /**< Tile/repeat */
    NT_OP_EXPAND        = 0x9C,     /**< Expand (broadcast) */
    NT_OP_NARROW        = 0x9D,     /**< Narrow dimension */
    NT_OP_FLIP          = 0x9E,     /**< Flip along axis */
    NT_OP_ROLL          = 0x9F,     /**< Roll along axis */
    
    /* === Attention (0xA0-0xAF) === */
    NT_OP_ATTENTION     = 0xA0,     /**< Scaled dot-product attention */
    NT_OP_FLASH_ATTN    = 0xA1,     /**< Flash attention */
    NT_OP_ROPE          = 0xA2,     /**< Rotary position embedding */
    NT_OP_ALIBI         = 0xA3,     /**< ALiBi position bias */
    NT_OP_MQA           = 0xA4,     /**< Multi-query attention */
    NT_OP_GQA           = 0xA5,     /**< Grouped-query attention */
    NT_OP_CROSS_ATTN    = 0xA6,     /**< Cross attention */
    NT_OP_SLIDING_ATTN  = 0xA7,     /**< Sliding window attention */
    
    /* === Embedding (0xB0-0xBF) === */
    NT_OP_EMBED         = 0xB0,     /**< Embedding lookup */
    NT_OP_ONEHOT        = 0xB1,     /**< One-hot encoding */
    NT_OP_GATHER        = 0xB2,     /**< Gather */
    NT_OP_SCATTER       = 0xB3,     /**< Scatter */
    NT_OP_INDEX_SELECT  = 0xB4,     /**< Index select */
    NT_OP_INDEX_ADD     = 0xB5,     /**< Index add */
    NT_OP_INDEX_COPY    = 0xB6,     /**< Index copy */
    NT_OP_MASKED_FILL   = 0xB7,     /**< Masked fill */
    NT_OP_MASKED_SELECT = 0xB8,     /**< Masked select */
    
    /* === Quantization (0xC0-0xCF) === */
    NT_OP_QUANTIZE      = 0xC0,     /**< Float → Quantized */
    NT_OP_DEQUANTIZE    = 0xC1,     /**< Quantized → Float */
    NT_OP_REQUANTIZE    = 0xC2,     /**< Quantized → Quantized */
    NT_OP_CALIBRATE     = 0xC3,     /**< Collect calibration stats */
    NT_OP_Q_MATMUL      = 0xC4,     /**< Quantized matmul */
    NT_OP_Q_ADD         = 0xC5,     /**< Quantized add */
    
    /* === Echo State (0xD0-0xDF) === */
    NT_OP_ESN_UPDATE    = 0xD0,     /**< Reservoir state update */
    NT_OP_ESN_READOUT   = 0xD1,     /**< Readout projection */
    NT_OP_SPECTRAL_NORM = 0xD2,     /**< Spectral normalization */
    NT_OP_SPARSE_MATMUL = 0xD3,     /**< Sparse matrix multiply */
    NT_OP_LEAKY_INTEGRATOR = 0xD4, /**< Leaky integrator */
    
    /* === Graph Neural Network (0xE0-0xEF) === */
    NT_OP_MESSAGE       = 0xE0,     /**< Message computation */
    NT_OP_AGGREGATE     = 0xE1,     /**< Message aggregation */
    NT_OP_GNN_UPDATE    = 0xE2,     /**< Node update */
    NT_OP_EDGE_CONV     = 0xE3,     /**< Edge convolution */
    NT_OP_GAT           = 0xE4,     /**< Graph attention */
    NT_OP_GCN           = 0xE5,     /**< Graph convolution */
    NT_OP_SAGE          = 0xE6,     /**< GraphSAGE */
    NT_OP_GIN           = 0xE7,     /**< Graph Isomorphism Network */
    NT_OP_GLOBAL_POOL   = 0xE8,     /**< Global graph pooling */
    
    /* === Special (0xF0-0xFF) === */
    NT_OP_CUSTOM        = 0xF0,     /**< Custom operation */
    NT_OP_FUSED         = 0xF1,     /**< Fused operation sequence */
    NT_OP_CHECKPOINT    = 0xF2,     /**< Gradient checkpointing */
    NT_OP_COPY          = 0xF3,     /**< Copy tensor */
    NT_OP_CONTIGUOUS    = 0xF4,     /**< Make contiguous */
    NT_OP_VIEW          = 0xF5,     /**< View (no copy) */
    NT_OP_DEBUG         = 0xFF,     /**< Debug/print */
    
} nt_op_t;

/* ============================================================================
 * OPERATION PARAMETERS
 * ============================================================================ */

/**
 * @brief Operation parameters
 */
typedef struct nt_op_params {
    /* Common parameters */
    int32_t         axes[NT_MAX_DIMS];      /**< Axes for reductions, permutations */
    int32_t         n_axes;                 /**< Number of axes */
    
    /* Padding/stride */
    int32_t         pad[NT_MAX_DIMS];       /**< Padding per dimension */
    int32_t         stride[NT_MAX_DIMS];    /**< Stride per dimension */
    int32_t         dilation[NT_MAX_DIMS];  /**< Dilation per dimension */
    
    /* Scalars */
    float           alpha;                  /**< Scale factor α */
    float           beta;                   /**< Scale factor β */
    float           epsilon;                /**< Small constant ε */
    
    /* Attention params */
    int32_t         n_heads;                /**< Number of attention heads */
    int32_t         head_dim;               /**< Dimension per head */
    float           scale;                  /**< Attention scale */
    
    /* Normalization params */
    float           momentum;               /**< Batch norm momentum */
    int32_t         num_groups;             /**< Group norm groups */
    
    /* Activation params */
    float           negative_slope;         /**< Leaky ReLU slope */
    
    /* Flags */
    uint32_t        flags;                  /**< Operation flags */
    
} nt_op_params_t;

/* Operation flags */
#define NT_OP_FLAG_INPLACE      (1 << 0)    /**< In-place operation */
#define NT_OP_FLAG_ACCUMULATE   (1 << 1)    /**< Accumulate result */
#define NT_OP_FLAG_TRANSPOSE_A  (1 << 2)    /**< Transpose first input */
#define NT_OP_FLAG_TRANSPOSE_B  (1 << 3)    /**< Transpose second input */
#define NT_OP_FLAG_BROADCAST    (1 << 4)    /**< Enable broadcasting */
#define NT_OP_FLAG_KEEPDIM      (1 << 5)    /**< Keep reduced dimensions */
#define NT_OP_FLAG_TRAINING     (1 << 6)    /**< Training mode */
#define NT_OP_FLAG_CAUSAL       (1 << 7)    /**< Causal attention mask */

/* ============================================================================
 * OPERATION DISPATCH
 * ============================================================================ */

/**
 * @brief Compute function signature
 */
typedef void (*nt_compute_fn)(
    const nt_tensor_t** inputs,
    int n_inputs,
    nt_tensor_t* output,
    const nt_op_params_t* params,
    void* backend_ctx
);

/**
 * @brief Operation registry entry
 */
typedef struct nt_op_entry {
    nt_op_t             op;                 /**< Operation code */
    const char*         name;               /**< Operation name */
    nt_compute_fn       compute[16];        /**< Per-dtype implementations */
    nt_op_signature_t   signature;          /**< Type signature */
    uint32_t            supported_dtypes;   /**< Bitmask of supported dtypes */
} nt_op_entry_t;

/**
 * @brief Dispatch to appropriate implementation
 */
void nt_compute(nt_op_t op,
                const nt_tensor_t** inputs,
                int n_inputs,
                nt_tensor_t* output,
                const nt_op_params_t* params);

/**
 * @brief Get operation name
 */
const char* nt_op_name(nt_op_t op);

/**
 * @brief Check if operation is supported for dtype
 */
bool nt_op_supports_dtype(nt_op_t op, nt_dtype_t dtype);

/* ============================================================================
 * CONVENIENCE FUNCTIONS
 * ============================================================================ */

/* Unary operations */
nt_tensor_t* nt_neg(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_abs(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_sqrt(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_exp(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_log(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_sin(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_cos(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_tanh(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_sigmoid(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_relu(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_gelu(nt_context_t* ctx, nt_tensor_t* x);
nt_tensor_t* nt_silu(nt_context_t* ctx, nt_tensor_t* x);

/* Binary operations */
nt_tensor_t* nt_add(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);
nt_tensor_t* nt_sub(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);
nt_tensor_t* nt_mul(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);
nt_tensor_t* nt_div(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);
nt_tensor_t* nt_pow(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);

/* Scalar operations */
nt_tensor_t* nt_add_scalar(nt_context_t* ctx, nt_tensor_t* x, float s);
nt_tensor_t* nt_mul_scalar(nt_context_t* ctx, nt_tensor_t* x, float s);
nt_tensor_t* nt_pow_scalar(nt_context_t* ctx, nt_tensor_t* x, float s);

/* Reductions */
nt_tensor_t* nt_sum(nt_context_t* ctx, nt_tensor_t* x, int axis, bool keepdim);
nt_tensor_t* nt_mean(nt_context_t* ctx, nt_tensor_t* x, int axis, bool keepdim);
nt_tensor_t* nt_max(nt_context_t* ctx, nt_tensor_t* x, int axis, bool keepdim);
nt_tensor_t* nt_min(nt_context_t* ctx, nt_tensor_t* x, int axis, bool keepdim);
nt_tensor_t* nt_softmax(nt_context_t* ctx, nt_tensor_t* x, int axis);

/* Matrix operations */
nt_tensor_t* nt_matmul(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);
nt_tensor_t* nt_dot(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);
nt_tensor_t* nt_outer(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b);

/* Normalization */
nt_tensor_t* nt_layer_norm(nt_context_t* ctx, nt_tensor_t* x, 
                           nt_tensor_t* weight, nt_tensor_t* bias, float eps);
nt_tensor_t* nt_rms_norm(nt_context_t* ctx, nt_tensor_t* x, 
                          nt_tensor_t* weight, float eps);

/* Attention */
nt_tensor_t* nt_attention(nt_context_t* ctx, 
                          nt_tensor_t* q, nt_tensor_t* k, nt_tensor_t* v,
                          nt_tensor_t* mask, int n_heads, float scale);

/* ============================================================================
 * BACKEND INTERFACE
 * ============================================================================ */

/**
 * @brief Backend interface
 */
struct nt_backend {
    const char*         name;               /**< Backend name */
    nt_device_t         device_type;        /**< Device type */
    
    /* Lifecycle */
    int  (*init)(struct nt_backend* be);
    void (*shutdown)(struct nt_backend* be);
    
    /* Memory */
    void* (*alloc)(struct nt_backend* be, size_t size, size_t align);
    void  (*free)(struct nt_backend* be, void* ptr);
    void  (*copy_to_device)(struct nt_backend* be, void* dst, const void* src, size_t n);
    void  (*copy_from_device)(struct nt_backend* be, void* dst, const void* src, size_t n);
    void  (*copy_device_to_device)(struct nt_backend* be, void* dst, const void* src, size_t n);
    
    /* Compute */
    void (*compute)(struct nt_backend* be, nt_op_t op,
                   const nt_tensor_t** inputs, int n_inputs,
                   nt_tensor_t* output, const nt_op_params_t* params);
    
    /* Synchronization */
    void (*sync)(struct nt_backend* be);
    
    /* Graph execution */
    void (*graph_compute)(struct nt_backend* be, nt_graph_t* g, int n_threads);
    
    /* Capabilities */
    bool (*supports_op)(struct nt_backend* be, nt_op_t op, nt_dtype_t dtype);
    
    /* Context */
    void* ctx;
};

/* Register/get backends */
void nt_register_backend(nt_backend_t* be);
nt_backend_t* nt_get_backend(nt_device_t device);

/* Built-in backends */
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

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_OPS_H */
