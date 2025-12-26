/**
 * @file nt_types.h
 * @brief NTTESHGNN - Foundational Type Definitions (Layer 0)
 * 
 * This file defines the primitive types that form the foundation of the
 * NTTESHGNN tensor framework. All other components build upon these types.
 * 
 * Design principles:
 * - From GGML: Minimal overhead, explicit bit layouts
 * - From PyTorch: Extensible type enumeration
 * - Novel: Nested tensor and hyperedge type support
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_TYPES_H
#define NTTESHGNN_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * VERSION INFO
 * ============================================================================ */

#define NT_VERSION_MAJOR 0
#define NT_VERSION_MINOR 1
#define NT_VERSION_PATCH 0
#define NT_VERSION_STRING "0.1.0"

/* ============================================================================
 * PLATFORM DETECTION
 * ============================================================================ */

#if defined(__x86_64__) || defined(_M_X64)
    #define NT_ARCH_X64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define NT_ARCH_ARM64 1
#elif defined(__wasm__)
    #define NT_ARCH_WASM 1
#endif

#if defined(_WIN32) || defined(_WIN64)
    #define NT_OS_WINDOWS 1
#elif defined(__APPLE__)
    #define NT_OS_MACOS 1
#elif defined(__linux__)
    #define NT_OS_LINUX 1
#endif

/* ============================================================================
 * COMPILER ATTRIBUTES
 * ============================================================================ */

#ifdef __GNUC__
    #define NT_ALIGNED(x)    __attribute__((aligned(x)))
    #define NT_PACKED        __attribute__((packed))
    #define NT_INLINE        static inline __attribute__((always_inline))
    #define NT_NOINLINE      __attribute__((noinline))
    #define NT_UNUSED        __attribute__((unused))
    #define NT_LIKELY(x)     __builtin_expect(!!(x), 1)
    #define NT_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#else
    #define NT_ALIGNED(x)
    #define NT_PACKED
    #define NT_INLINE        static inline
    #define NT_NOINLINE
    #define NT_UNUSED
    #define NT_LIKELY(x)     (x)
    #define NT_UNLIKELY(x)   (x)
#endif

/* ============================================================================
 * SECTION 1: SCALAR DATA TYPES (The Atoms)
 * ============================================================================
 * 
 * Data type enumeration with fixed bit layout for efficient dispatch:
 * Bits [0:7]   = bit width / sub-type encoding
 * Bits [8:11]  = base type (float, int, uint, complex, special)
 * Bits [12:15] = quantization scheme / variant
 */

typedef enum nt_dtype {
    /* Floating point family (base = 0x0) */
    NT_F64      = 0x0040,   /**< 64-bit IEEE 754 double precision */
    NT_F32      = 0x0020,   /**< 32-bit IEEE 754 single precision (default) */
    NT_F16      = 0x0010,   /**< 16-bit IEEE 754 half precision */
    NT_BF16     = 0x0011,   /**< 16-bit Brain Float (Google format) */
    NT_F8E4M3   = 0x0008,   /**< 8-bit float (4 exp, 3 mantissa) - FP8 */
    NT_F8E5M2   = 0x0009,   /**< 8-bit float (5 exp, 2 mantissa) - FP8 */
    
    /* Signed integer family (base = 0x1) */
    NT_I64      = 0x0140,   /**< 64-bit signed integer */
    NT_I32      = 0x0120,   /**< 32-bit signed integer */
    NT_I16      = 0x0110,   /**< 16-bit signed integer */
    NT_I8       = 0x0108,   /**< 8-bit signed integer */
    NT_I4       = 0x0104,   /**< 4-bit signed integer (packed) */
    NT_I2       = 0x0102,   /**< 2-bit signed integer (packed) */
    NT_I1       = 0x0101,   /**< 1-bit binary */
    
    /* Unsigned integer family (base = 0x2) */
    NT_U64      = 0x0240,   /**< 64-bit unsigned integer */
    NT_U32      = 0x0220,   /**< 32-bit unsigned integer */
    NT_U16      = 0x0210,   /**< 16-bit unsigned integer */
    NT_U8       = 0x0208,   /**< 8-bit unsigned integer */
    NT_U4       = 0x0204,   /**< 4-bit unsigned integer (packed) */
    
    /* Quantized types - GGML compatible (base = 0x3) */
    NT_Q8_0     = 0x0308,   /**< 8-bit quantized, block size 32, no bias */
    NT_Q8_1     = 0x0318,   /**< 8-bit quantized with bias */
    NT_Q5_0     = 0x0305,   /**< 5-bit quantized, no bias */
    NT_Q5_1     = 0x0315,   /**< 5-bit quantized with bias */
    NT_Q4_0     = 0x0304,   /**< 4-bit quantized (GGML default) */
    NT_Q4_1     = 0x0314,   /**< 4-bit quantized with bias */
    NT_Q4_K     = 0x0324,   /**< 4-bit k-quant (improved) */
    NT_Q6_K     = 0x0326,   /**< 6-bit k-quant */
    NT_Q2_K     = 0x0322,   /**< 2-bit k-quant */
    NT_IQ4_NL   = 0x0334,   /**< 4-bit importance quantized, non-linear */
    NT_IQ3_XXS  = 0x0333,   /**< 3-bit importance quantized, extra small */
    NT_IQ2_XXS  = 0x0332,   /**< 2-bit importance quantized, extra small */
    
    /* Complex types (base = 0x4) */
    NT_C64      = 0x0440,   /**< Complex float64 (real + imag) */
    NT_C32      = 0x0420,   /**< Complex float32 (real + imag) */
    NT_C16      = 0x0410,   /**< Complex float16 (real + imag) */
    
    /* Special types (base = 0x5+) */
    NT_BOOL     = 0x0501,   /**< Boolean (1 byte storage) */
    NT_NESTED   = 0x0600,   /**< Nested tensor (recursive structure) */
    NT_EDGE     = 0x0700,   /**< Hypergraph edge reference */
    NT_PTR      = 0x0800,   /**< Generic pointer type */
    NT_VOID     = 0x0000,   /**< Uninitialized / null type */
    
} nt_dtype_t;

/* Data type property extraction macros */
#define NT_DTYPE_BITS(dt)   ((uint8_t)((dt) & 0xFF))
#define NT_DTYPE_BASE(dt)   ((uint8_t)(((dt) >> 8) & 0x0F))
#define NT_DTYPE_QUANT(dt)  ((uint8_t)(((dt) >> 12) & 0x0F))

/* Data type property functions */
NT_INLINE uint8_t nt_dtype_bits(nt_dtype_t dt) {
    uint8_t raw = NT_DTYPE_BITS(dt);
    /* Handle special encodings */
    if (raw <= 8) return raw;
    if (raw == 0x10) return 16;
    if (raw == 0x11) return 16;  /* BF16 */
    if (raw == 0x20) return 32;
    if (raw == 0x40) return 64;
    return raw;
}

NT_INLINE uint8_t nt_dtype_base(nt_dtype_t dt) {
    return NT_DTYPE_BASE(dt);
}

NT_INLINE bool nt_dtype_is_float(nt_dtype_t dt) {
    return NT_DTYPE_BASE(dt) == 0x0;
}

NT_INLINE bool nt_dtype_is_int(nt_dtype_t dt) {
    uint8_t base = NT_DTYPE_BASE(dt);
    return base == 0x1 || base == 0x2;
}

NT_INLINE bool nt_dtype_is_quantized(nt_dtype_t dt) {
    return NT_DTYPE_BASE(dt) == 0x3;
}

NT_INLINE bool nt_dtype_is_complex(nt_dtype_t dt) {
    return NT_DTYPE_BASE(dt) == 0x4;
}

NT_INLINE size_t nt_dtype_size(nt_dtype_t dt) {
    /* Return size in bytes for one element */
    switch (dt) {
        case NT_F64:
        case NT_I64:
        case NT_U64:
        case NT_C64:
            return 8;
        case NT_F32:
        case NT_I32:
        case NT_U32:
        case NT_C32:
            return 4;
        case NT_F16:
        case NT_BF16:
        case NT_I16:
        case NT_U16:
        case NT_C16:
            return 2;
        case NT_F8E4M3:
        case NT_F8E5M2:
        case NT_I8:
        case NT_U8:
        case NT_Q8_0:
        case NT_Q8_1:
        case NT_BOOL:
            return 1;
        /* Sub-byte types return 1 (packed handling elsewhere) */
        case NT_I4:
        case NT_U4:
        case NT_Q4_0:
        case NT_Q4_1:
        case NT_Q4_K:
        case NT_Q5_0:
        case NT_Q5_1:
        case NT_Q6_K:
        case NT_Q2_K:
        case NT_IQ4_NL:
        case NT_IQ3_XXS:
        case NT_IQ2_XXS:
        case NT_I2:
        case NT_I1:
            return 1;  /* Minimum addressable unit */
        case NT_PTR:
            return sizeof(void*);
        default:
            return 0;
    }
}

/* Get human-readable name for dtype */
const char* nt_dtype_name(nt_dtype_t dt);

/* ============================================================================
 * SECTION 2: DEVICE TYPES
 * ============================================================================ */

typedef enum nt_device {
    NT_DEV_CPU      = 0x00,     /**< Host CPU */
    NT_DEV_CUDA     = 0x10,     /**< NVIDIA CUDA GPU */
    NT_DEV_METAL    = 0x20,     /**< Apple Metal GPU */
    NT_DEV_VULKAN   = 0x30,     /**< Vulkan compute */
    NT_DEV_VTNPU    = 0x40,     /**< Virtual Tensor NPU (custom) */
    NT_DEV_FPGA     = 0x50,     /**< FPGA fabric */
    NT_DEV_OPENCL   = 0x60,     /**< OpenCL device */
    NT_DEV_REMOTE   = 0xF0,     /**< Remote device (network) */
} nt_device_t;

/* Device with index (e.g., cuda:0, cuda:1) */
typedef struct nt_device_id {
    nt_device_t type;
    uint16_t    index;
} nt_device_id_t;

#define NT_DEVICE_CPU       ((nt_device_id_t){NT_DEV_CPU, 0})
#define NT_DEVICE_CUDA(i)   ((nt_device_id_t){NT_DEV_CUDA, (i)})
#define NT_DEVICE_METAL(i)  ((nt_device_id_t){NT_DEV_METAL, (i)})
#define NT_DEVICE_VTNPU(i)  ((nt_device_id_t){NT_DEV_VTNPU, (i)})

/* Get human-readable name for device */
const char* nt_device_name(nt_device_t dev);

/* ============================================================================
 * SECTION 3: MEMORY LAYOUT TYPES
 * ============================================================================ */

typedef enum nt_layout {
    NT_LAYOUT_STRIDED       = 0,    /**< General strided (default) */
    NT_LAYOUT_CONTIGUOUS    = 1,    /**< Row-major contiguous (C order) */
    NT_LAYOUT_FORTRAN       = 2,    /**< Column-major contiguous (F order) */
    NT_LAYOUT_CHANNELS_LAST = 3,    /**< NHWC layout for images */
    NT_LAYOUT_BLOCKED       = 4,    /**< Block-sparse layout */
    NT_LAYOUT_JAGGED        = 5,    /**< Nested/ragged arrays */
    NT_LAYOUT_SPARSE_COO    = 6,    /**< Coordinate sparse format */
    NT_LAYOUT_SPARSE_CSR    = 7,    /**< Compressed Sparse Row */
    NT_LAYOUT_SPARSE_CSC    = 8,    /**< Compressed Sparse Column */
} nt_layout_t;

/* ============================================================================
 * SECTION 4: ERROR CODES
 * ============================================================================ */

typedef enum nt_status {
    NT_OK                   = 0,    /**< Success */
    NT_ERR_INVALID_ARG      = -1,   /**< Invalid argument */
    NT_ERR_OUT_OF_MEMORY    = -2,   /**< Memory allocation failed */
    NT_ERR_SHAPE_MISMATCH   = -3,   /**< Tensor shapes incompatible */
    NT_ERR_DTYPE_MISMATCH   = -4,   /**< Data types incompatible */
    NT_ERR_DEVICE_MISMATCH  = -5,   /**< Tensors on different devices */
    NT_ERR_NOT_IMPLEMENTED  = -6,   /**< Feature not implemented */
    NT_ERR_INVALID_STATE    = -7,   /**< Invalid object state */
    NT_ERR_IO               = -8,   /**< I/O error */
    NT_ERR_BACKEND          = -9,   /**< Backend-specific error */
    NT_ERR_TYPE_CHECK       = -10,  /**< Type system check failed */
    NT_ERR_GRAPH            = -11,  /**< Graph construction error */
    NT_ERR_OVERFLOW         = -12,  /**< Numeric overflow */
    NT_ERR_UNDERFLOW        = -13,  /**< Numeric underflow */
} nt_status_t;

/* Get human-readable error message */
const char* nt_status_str(nt_status_t status);

/* ============================================================================
 * SECTION 5: QUANTIZATION BLOCK STRUCTURES (GGML-compatible)
 * ============================================================================ */

/* Block size for quantization (GGML uses 32) */
#define NT_QK_K     256     /* K-quant super-block size */
#define NT_QK       32      /* Standard block size */

/* Q4_0: 4-bit quantization, block of 32 */
typedef struct NT_PACKED nt_block_q4_0 {
    uint16_t d;             /* Delta (FP16) */
    uint8_t  qs[NT_QK/2];   /* Quantized values (4 bits each) */
} nt_block_q4_0_t;

/* Q4_1: 4-bit quantization with min, block of 32 */
typedef struct NT_PACKED nt_block_q4_1 {
    uint16_t d;             /* Delta (FP16) */
    uint16_t m;             /* Min (FP16) */
    uint8_t  qs[NT_QK/2];   /* Quantized values */
} nt_block_q4_1_t;

/* Q8_0: 8-bit quantization, block of 32 */
typedef struct NT_PACKED nt_block_q8_0 {
    uint16_t d;             /* Delta (FP16) */
    int8_t   qs[NT_QK];     /* Quantized values */
} nt_block_q8_0_t;

/* Q8_1: 8-bit quantization with sum, block of 32 */
typedef struct NT_PACKED nt_block_q8_1 {
    float    d;             /* Delta */
    float    s;             /* Sum of quantized values */
    int8_t   qs[NT_QK];     /* Quantized values */
} nt_block_q8_1_t;

/* Q5_0: 5-bit quantization, block of 32 */
typedef struct NT_PACKED nt_block_q5_0 {
    uint16_t d;             /* Delta (FP16) */
    uint8_t  qh[4];         /* High bits */
    uint8_t  qs[NT_QK/2];   /* Low 4 bits */
} nt_block_q5_0_t;

/* Q5_1: 5-bit quantization with min, block of 32 */
typedef struct NT_PACKED nt_block_q5_1 {
    uint16_t d;             /* Delta (FP16) */
    uint16_t m;             /* Min (FP16) */
    uint8_t  qh[4];         /* High bits */
    uint8_t  qs[NT_QK/2];   /* Low 4 bits */
} nt_block_q5_1_t;

/* ============================================================================
 * SECTION 6: FORWARD DECLARATIONS
 * ============================================================================ */

/* Core structures (defined in other headers) */
typedef struct nt_tensor        nt_tensor_t;
typedef struct nt_storage       nt_storage_t;
typedef struct nt_context       nt_context_t;
typedef struct nt_graph         nt_graph_t;
typedef struct nt_edge          nt_edge_t;
typedef struct nt_reservoir     nt_reservoir_t;
typedef struct nt_type          nt_type_t;
typedef struct nt_backend       nt_backend_t;
typedef struct nt_allocator     nt_allocator_t;

/* ============================================================================
 * SECTION 7: COMMON CONSTANTS
 * ============================================================================ */

/* Maximum tensor dimensions (GGML uses 4, we support 8) */
#define NT_MAX_DIMS     8

/* Maximum source tensors for computation graph edges */
#define NT_MAX_SRC      4

/* Maximum name length for tensors */
#define NT_MAX_NAME     64

/* Default alignment for allocations */
#define NT_DEFAULT_ALIGN    64

/* Cache line size */
#define NT_CACHE_LINE       64

/* ============================================================================
 * SECTION 8: UTILITY MACROS
 * ============================================================================ */

#define NT_MIN(a, b)    ((a) < (b) ? (a) : (b))
#define NT_MAX(a, b)    ((a) > (b) ? (a) : (b))
#define NT_CLAMP(x, lo, hi) NT_MIN(NT_MAX(x, lo), hi)

#define NT_ALIGN_UP(x, align)   (((x) + (align) - 1) & ~((align) - 1))
#define NT_ALIGN_DOWN(x, align) ((x) & ~((align) - 1))

#define NT_ARRAY_SIZE(arr)      (sizeof(arr) / sizeof((arr)[0]))

#define NT_UNUSED_VAR(x)        ((void)(x))

/* Debug assertion */
#ifdef NT_DEBUG
    #include <assert.h>
    #define NT_ASSERT(cond) assert(cond)
#else
    #define NT_ASSERT(cond) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_TYPES_H */
