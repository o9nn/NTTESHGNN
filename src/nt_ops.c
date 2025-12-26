/**
 * @file nt_ops.c
 * @brief NTTESHGNN - Operations implementation
 */

#include "ntteshgnn/nt_ops.h"
#include "ntteshgnn/nt_context.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * OPERATION NAMES
 * ============================================================================ */

const char* nt_op_name(nt_op_t op) {
    switch (op) {
        case NT_OP_NONE:        return "none";
        case NT_OP_NEG:         return "neg";
        case NT_OP_ABS:         return "abs";
        case NT_OP_SIGN:        return "sign";
        case NT_OP_SQRT:        return "sqrt";
        case NT_OP_RSQRT:       return "rsqrt";
        case NT_OP_SQUARE:      return "square";
        case NT_OP_EXP:         return "exp";
        case NT_OP_LOG:         return "log";
        case NT_OP_SIN:         return "sin";
        case NT_OP_COS:         return "cos";
        case NT_OP_TANH:        return "tanh";
        case NT_OP_SIGMOID:     return "sigmoid";
        case NT_OP_RELU:        return "relu";
        case NT_OP_GELU:        return "gelu";
        case NT_OP_SILU:        return "silu";
        case NT_OP_ADD:         return "add";
        case NT_OP_SUB:         return "sub";
        case NT_OP_MUL:         return "mul";
        case NT_OP_DIV:         return "div";
        case NT_OP_POW:         return "pow";
        case NT_OP_SUM:         return "sum";
        case NT_OP_MEAN:        return "mean";
        case NT_OP_SOFTMAX:     return "softmax";
        case NT_OP_MATMUL:      return "matmul";
        case NT_OP_LAYER_NORM:  return "layer_norm";
        case NT_OP_RMS_NORM:    return "rms_norm";
        case NT_OP_ATTENTION:   return "attention";
        case NT_OP_ROPE:        return "rope";
        case NT_OP_EMBED:       return "embed";
        case NT_OP_QUANTIZE:    return "quantize";
        case NT_OP_DEQUANTIZE:  return "dequantize";
        default:                return "unknown";
    }
}

/* ============================================================================
 * CPU KERNELS - UNARY OPERATIONS
 * ============================================================================ */

static void cpu_neg_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = -src[i];
    }
}

static void cpu_abs_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = fabsf(src[i]);
    }
}

static void cpu_sqrt_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

static void cpu_exp_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = expf(src[i]);
    }
}

static void cpu_log_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = logf(src[i]);
    }
}

static void cpu_sin_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = sinf(src[i]);
    }
}

static void cpu_cos_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = cosf(src[i]);
    }
}

static void cpu_tanh_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = tanhf(src[i]);
    }
}

static void cpu_sigmoid_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
    }
}

static void cpu_relu_f32(const float* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

static void cpu_gelu_f32(const float* src, float* dst, int64_t n) {
    /* GELU(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) */
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    for (int64_t i = 0; i < n; i++) {
        float x = src[i];
        float x3 = x * x * x;
        dst[i] = 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + coeff * x3)));
    }
}

static void cpu_silu_f32(const float* src, float* dst, int64_t n) {
    /* SiLU(x) = x * sigmoid(x) */
    for (int64_t i = 0; i < n; i++) {
        float x = src[i];
        dst[i] = x / (1.0f + expf(-x));
    }
}

/* ============================================================================
 * CPU KERNELS - BINARY OPERATIONS
 * ============================================================================ */

static void cpu_add_f32(const float* a, const float* b, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

static void cpu_sub_f32(const float* a, const float* b, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = a[i] - b[i];
    }
}

static void cpu_mul_f32(const float* a, const float* b, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

static void cpu_div_f32(const float* a, const float* b, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = a[i] / b[i];
    }
}

/* ============================================================================
 * CPU KERNELS - REDUCTION OPERATIONS
 * ============================================================================ */

static float cpu_sum_f32(const float* src, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        sum += src[i];
    }
    return sum;
}

static float cpu_mean_f32(const float* src, int64_t n) {
    return cpu_sum_f32(src, n) / (float)n;
}

static void cpu_softmax_f32(const float* src, float* dst, int64_t n) {
    /* Find max for numerical stability */
    float max_val = src[0];
    for (int64_t i = 1; i < n; i++) {
        if (src[i] > max_val) max_val = src[i];
    }
    
    /* Compute exp and sum */
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = expf(src[i] - max_val);
        sum += dst[i];
    }
    
    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < n; i++) {
        dst[i] *= inv_sum;
    }
}

/* ============================================================================
 * CPU KERNELS - MATRIX OPERATIONS
 * ============================================================================ */

static void cpu_matmul_f32(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B[K,N] */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ============================================================================
 * CPU KERNELS - NORMALIZATION
 * ============================================================================ */

static void cpu_layer_norm_f32(const float* src, const float* weight, 
                                const float* bias, float* dst,
                                int n_features, int batch_size, float eps) {
    for (int b = 0; b < batch_size; b++) {
        const float* x = src + b * n_features;
        float* y = dst + b * n_features;
        
        /* Compute mean */
        float mean = 0.0f;
        for (int i = 0; i < n_features; i++) {
            mean += x[i];
        }
        mean /= (float)n_features;
        
        /* Compute variance */
        float var = 0.0f;
        for (int i = 0; i < n_features; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= (float)n_features;
        
        /* Normalize */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < n_features; i++) {
            float norm = (x[i] - mean) * inv_std;
            y[i] = norm * (weight ? weight[i] : 1.0f) + (bias ? bias[i] : 0.0f);
        }
    }
}

static void cpu_rms_norm_f32(const float* src, const float* weight,
                              float* dst, int n_features, int batch_size, float eps) {
    for (int b = 0; b < batch_size; b++) {
        const float* x = src + b * n_features;
        float* y = dst + b * n_features;
        
        /* Compute RMS */
        float sum_sq = 0.0f;
        for (int i = 0; i < n_features; i++) {
            sum_sq += x[i] * x[i];
        }
        float rms = sqrtf(sum_sq / (float)n_features + eps);
        float inv_rms = 1.0f / rms;
        
        /* Normalize */
        for (int i = 0; i < n_features; i++) {
            y[i] = x[i] * inv_rms * (weight ? weight[i] : 1.0f);
        }
    }
}

/* ============================================================================
 * OPERATION DISPATCH
 * ============================================================================ */

void nt_compute(nt_op_t op,
                const nt_tensor_t** inputs,
                int n_inputs,
                nt_tensor_t* output,
                const nt_op_params_t* params) {
    if (!output || !output->data) return;
    
    int64_t n = nt_tensor_numel(output);
    
    /* Dispatch based on operation and dtype */
    if (output->dtype == NT_F32) {
        float* dst = (float*)output->data;
        
        switch (op) {
            /* Unary operations */
            case NT_OP_NEG:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_neg_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_ABS:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_abs_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_SQRT:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_sqrt_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_EXP:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_exp_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_LOG:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_log_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_SIN:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_sin_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_COS:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_cos_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_TANH:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_tanh_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_SIGMOID:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_sigmoid_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_RELU:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_relu_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_GELU:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_gelu_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
            case NT_OP_SILU:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_silu_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
                
            /* Binary operations */
            case NT_OP_ADD:
                if (n_inputs >= 2 && inputs[0] && inputs[1]) {
                    cpu_add_f32((const float*)inputs[0]->data, 
                               (const float*)inputs[1]->data, dst, n);
                }
                break;
            case NT_OP_SUB:
                if (n_inputs >= 2 && inputs[0] && inputs[1]) {
                    cpu_sub_f32((const float*)inputs[0]->data,
                               (const float*)inputs[1]->data, dst, n);
                }
                break;
            case NT_OP_MUL:
                if (n_inputs >= 2 && inputs[0] && inputs[1]) {
                    cpu_mul_f32((const float*)inputs[0]->data,
                               (const float*)inputs[1]->data, dst, n);
                }
                break;
            case NT_OP_DIV:
                if (n_inputs >= 2 && inputs[0] && inputs[1]) {
                    cpu_div_f32((const float*)inputs[0]->data,
                               (const float*)inputs[1]->data, dst, n);
                }
                break;
                
            /* Reduction operations */
            case NT_OP_SOFTMAX:
                if (n_inputs >= 1 && inputs[0]) {
                    cpu_softmax_f32((const float*)inputs[0]->data, dst, n);
                }
                break;
                
            /* Matrix operations */
            case NT_OP_MATMUL:
                if (n_inputs >= 2 && inputs[0] && inputs[1]) {
                    int M = inputs[0]->ne[1];
                    int K = inputs[0]->ne[0];
                    int N = inputs[1]->ne[0];
                    cpu_matmul_f32((const float*)inputs[0]->data,
                                  (const float*)inputs[1]->data,
                                  dst, M, N, K);
                }
                break;
                
            /* Normalization */
            case NT_OP_LAYER_NORM:
                if (n_inputs >= 1 && inputs[0]) {
                    int n_features = inputs[0]->ne[0];
                    int batch_size = (int)(nt_tensor_numel(inputs[0]) / n_features);
                    const float* weight = (n_inputs >= 2 && inputs[1]) ? 
                                          (const float*)inputs[1]->data : NULL;
                    const float* bias = (n_inputs >= 3 && inputs[2]) ?
                                        (const float*)inputs[2]->data : NULL;
                    float eps = params ? params->epsilon : 1e-5f;
                    cpu_layer_norm_f32((const float*)inputs[0]->data,
                                      weight, bias, dst, n_features, batch_size, eps);
                }
                break;
                
            case NT_OP_RMS_NORM:
                if (n_inputs >= 1 && inputs[0]) {
                    int n_features = inputs[0]->ne[0];
                    int batch_size = (int)(nt_tensor_numel(inputs[0]) / n_features);
                    const float* weight = (n_inputs >= 2 && inputs[1]) ?
                                          (const float*)inputs[1]->data : NULL;
                    float eps = params ? params->epsilon : 1e-5f;
                    cpu_rms_norm_f32((const float*)inputs[0]->data,
                                    weight, dst, n_features, batch_size, eps);
                }
                break;
                
            default:
                fprintf(stderr, "Warning: Operation %s not implemented\n", nt_op_name(op));
                break;
        }
    }
}

/* ============================================================================
 * CONVENIENCE FUNCTIONS
 * ============================================================================ */

static nt_tensor_t* unary_op(nt_context_t* ctx, nt_tensor_t* x, nt_op_t op) {
    if (!x) return NULL;
    
    nt_tensor_t* result = nt_tensor_new(ctx, x->dtype, x->ndim, x->ne);
    if (!result) return NULL;
    
    const nt_tensor_t* inputs[] = {x};
    nt_compute(op, inputs, 1, result, NULL);
    
    return result;
}

static nt_tensor_t* binary_op(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b, nt_op_t op) {
    if (!a || !b) return NULL;
    
    /* TODO: Handle broadcasting */
    nt_tensor_t* result = nt_tensor_new(ctx, a->dtype, a->ndim, a->ne);
    if (!result) return NULL;
    
    const nt_tensor_t* inputs[] = {a, b};
    nt_compute(op, inputs, 2, result, NULL);
    
    return result;
}

nt_tensor_t* nt_neg(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_NEG);
}

nt_tensor_t* nt_abs(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_ABS);
}

nt_tensor_t* nt_sqrt(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_SQRT);
}

nt_tensor_t* nt_exp(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_EXP);
}

nt_tensor_t* nt_log(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_LOG);
}

nt_tensor_t* nt_sin(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_SIN);
}

nt_tensor_t* nt_cos(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_COS);
}

nt_tensor_t* nt_tanh(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_TANH);
}

nt_tensor_t* nt_sigmoid(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_SIGMOID);
}

nt_tensor_t* nt_relu(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_RELU);
}

nt_tensor_t* nt_gelu(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_GELU);
}

nt_tensor_t* nt_silu(nt_context_t* ctx, nt_tensor_t* x) {
    return unary_op(ctx, x, NT_OP_SILU);
}

nt_tensor_t* nt_add(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b) {
    return binary_op(ctx, a, b, NT_OP_ADD);
}

nt_tensor_t* nt_sub(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b) {
    return binary_op(ctx, a, b, NT_OP_SUB);
}

nt_tensor_t* nt_mul(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b) {
    return binary_op(ctx, a, b, NT_OP_MUL);
}

nt_tensor_t* nt_div(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b) {
    return binary_op(ctx, a, b, NT_OP_DIV);
}

nt_tensor_t* nt_matmul(nt_context_t* ctx, nt_tensor_t* a, nt_tensor_t* b) {
    if (!a || !b) return NULL;
    
    /* a: [M, K], b: [K, N] -> result: [M, N] */
    int32_t M = a->ne[1];
    int32_t N = b->ne[0];
    int32_t shape[] = {N, M};
    
    nt_tensor_t* result = nt_tensor_new(ctx, a->dtype, 2, shape);
    if (!result) return NULL;
    
    const nt_tensor_t* inputs[] = {a, b};
    nt_compute(NT_OP_MATMUL, inputs, 2, result, NULL);
    
    return result;
}

nt_tensor_t* nt_softmax(nt_context_t* ctx, nt_tensor_t* x, int axis) {
    NT_UNUSED_VAR(axis);  /* TODO: Implement axis parameter */
    return unary_op(ctx, x, NT_OP_SOFTMAX);
}

nt_tensor_t* nt_layer_norm(nt_context_t* ctx, nt_tensor_t* x,
                           nt_tensor_t* weight, nt_tensor_t* bias, float eps) {
    if (!x) return NULL;
    
    nt_tensor_t* result = nt_tensor_new(ctx, x->dtype, x->ndim, x->ne);
    if (!result) return NULL;
    
    const nt_tensor_t* inputs[] = {x, weight, bias};
    nt_op_params_t params = {0};
    params.epsilon = eps;
    
    nt_compute(NT_OP_LAYER_NORM, inputs, 3, result, &params);
    
    return result;
}

nt_tensor_t* nt_rms_norm(nt_context_t* ctx, nt_tensor_t* x,
                          nt_tensor_t* weight, float eps) {
    if (!x) return NULL;
    
    nt_tensor_t* result = nt_tensor_new(ctx, x->dtype, x->ndim, x->ne);
    if (!result) return NULL;
    
    const nt_tensor_t* inputs[] = {x, weight};
    nt_op_params_t params = {0};
    params.epsilon = eps;
    
    nt_compute(NT_OP_RMS_NORM, inputs, 2, result, &params);
    
    return result;
}

/* ============================================================================
 * BACKEND MANAGEMENT
 * ============================================================================ */

static nt_backend_t* g_backends[16] = {0};
static int g_n_backends = 0;

void nt_register_backend(nt_backend_t* be) {
    if (g_n_backends < 16) {
        g_backends[g_n_backends++] = be;
    }
}

nt_backend_t* nt_get_backend(nt_device_t device) {
    for (int i = 0; i < g_n_backends; i++) {
        if (g_backends[i] && g_backends[i]->device_type == device) {
            return g_backends[i];
        }
    }
    return NULL;
}

/* CPU backend (default) */
static int cpu_backend_init(nt_backend_t* be) {
    NT_UNUSED_VAR(be);
    return 0;
}

static void cpu_backend_shutdown(nt_backend_t* be) {
    NT_UNUSED_VAR(be);
}

static void* cpu_backend_alloc(nt_backend_t* be, size_t size, size_t align) {
    NT_UNUSED_VAR(be);
    void* ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void cpu_backend_free(nt_backend_t* be, void* ptr) {
    NT_UNUSED_VAR(be);
    free(ptr);
}

static void cpu_backend_compute(nt_backend_t* be, nt_op_t op,
                                const nt_tensor_t** inputs, int n_inputs,
                                nt_tensor_t* output, const nt_op_params_t* params) {
    NT_UNUSED_VAR(be);
    nt_compute(op, inputs, n_inputs, output, params);
}

static void cpu_backend_sync(nt_backend_t* be) {
    NT_UNUSED_VAR(be);
    /* CPU is synchronous */
}

static nt_backend_t g_cpu_backend = {
    .name = "cpu",
    .device_type = NT_DEV_CPU,
    .init = cpu_backend_init,
    .shutdown = cpu_backend_shutdown,
    .alloc = cpu_backend_alloc,
    .free = cpu_backend_free,
    .compute = cpu_backend_compute,
    .sync = cpu_backend_sync,
    .ctx = NULL,
};

nt_backend_t* nt_backend_cpu(void) {
    return &g_cpu_backend;
}
