/**
 * @file test_ops.c
 * @brief NTTESHGNN - Operations tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ntteshgnn/ntteshgnn.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

#define TEST_PASS(name) printf("PASS: %s\n", name)

#define EPSILON 1e-5f

static int float_eq(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

int test_unary_ops(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_tensor_t* x = nt_tensor_new_1d(ctx, NT_F32, 5);
    float* data = (float*)x->data;
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    data[4] = 5.0f;
    
    /* Test neg */
    nt_tensor_t* neg_x = nt_neg(ctx, x);
    TEST_ASSERT(neg_x != NULL, "neg operation");
    float* neg_data = (float*)neg_x->data;
    TEST_ASSERT(float_eq(neg_data[0], -1.0f), "neg[0]");
    TEST_ASSERT(float_eq(neg_data[4], -5.0f), "neg[4]");
    nt_tensor_release(neg_x);
    
    /* Test sqrt */
    nt_tensor_t* sqrt_x = nt_sqrt(ctx, x);
    TEST_ASSERT(sqrt_x != NULL, "sqrt operation");
    float* sqrt_data = (float*)sqrt_x->data;
    TEST_ASSERT(float_eq(sqrt_data[0], 1.0f), "sqrt[0]");
    TEST_ASSERT(float_eq(sqrt_data[3], 2.0f), "sqrt[3]");
    nt_tensor_release(sqrt_x);
    
    /* Test exp */
    nt_tensor_t* exp_x = nt_exp(ctx, x);
    TEST_ASSERT(exp_x != NULL, "exp operation");
    float* exp_data = (float*)exp_x->data;
    TEST_ASSERT(float_eq(exp_data[0], expf(1.0f)), "exp[0]");
    nt_tensor_release(exp_x);
    
    /* Test tanh */
    nt_tensor_t* tanh_x = nt_tanh(ctx, x);
    TEST_ASSERT(tanh_x != NULL, "tanh operation");
    float* tanh_data = (float*)tanh_x->data;
    TEST_ASSERT(float_eq(tanh_data[0], tanhf(1.0f)), "tanh[0]");
    nt_tensor_release(tanh_x);
    
    /* Test relu */
    data[0] = -1.0f;
    data[1] = 0.0f;
    data[2] = 1.0f;
    nt_tensor_t* relu_x = nt_relu(ctx, x);
    TEST_ASSERT(relu_x != NULL, "relu operation");
    float* relu_data = (float*)relu_x->data;
    TEST_ASSERT(float_eq(relu_data[0], 0.0f), "relu[-1] = 0");
    TEST_ASSERT(float_eq(relu_data[1], 0.0f), "relu[0] = 0");
    TEST_ASSERT(float_eq(relu_data[2], 1.0f), "relu[1] = 1");
    nt_tensor_release(relu_x);
    
    /* Test sigmoid */
    data[0] = 0.0f;
    nt_tensor_t* sig_x = nt_sigmoid(ctx, x);
    TEST_ASSERT(sig_x != NULL, "sigmoid operation");
    float* sig_data = (float*)sig_x->data;
    TEST_ASSERT(float_eq(sig_data[0], 0.5f), "sigmoid[0] = 0.5");
    nt_tensor_release(sig_x);
    
    nt_tensor_release(x);
    nt_context_free(ctx);
    
    TEST_PASS("unary_ops");
    return 0;
}

int test_binary_ops(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_tensor_t* a = nt_tensor_new_1d(ctx, NT_F32, 4);
    nt_tensor_t* b = nt_tensor_new_1d(ctx, NT_F32, 4);
    
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;
    b_data[0] = 5.0f; b_data[1] = 6.0f; b_data[2] = 7.0f; b_data[3] = 8.0f;
    
    /* Test add */
    nt_tensor_t* sum = nt_add(ctx, a, b);
    TEST_ASSERT(sum != NULL, "add operation");
    float* sum_data = (float*)sum->data;
    TEST_ASSERT(float_eq(sum_data[0], 6.0f), "add[0]");
    TEST_ASSERT(float_eq(sum_data[3], 12.0f), "add[3]");
    nt_tensor_release(sum);
    
    /* Test sub */
    nt_tensor_t* diff = nt_sub(ctx, a, b);
    TEST_ASSERT(diff != NULL, "sub operation");
    float* diff_data = (float*)diff->data;
    TEST_ASSERT(float_eq(diff_data[0], -4.0f), "sub[0]");
    TEST_ASSERT(float_eq(diff_data[3], -4.0f), "sub[3]");
    nt_tensor_release(diff);
    
    /* Test mul */
    nt_tensor_t* prod = nt_mul(ctx, a, b);
    TEST_ASSERT(prod != NULL, "mul operation");
    float* prod_data = (float*)prod->data;
    TEST_ASSERT(float_eq(prod_data[0], 5.0f), "mul[0]");
    TEST_ASSERT(float_eq(prod_data[3], 32.0f), "mul[3]");
    nt_tensor_release(prod);
    
    /* Test div */
    nt_tensor_t* quot = nt_div(ctx, a, b);
    TEST_ASSERT(quot != NULL, "div operation");
    float* quot_data = (float*)quot->data;
    TEST_ASSERT(float_eq(quot_data[0], 0.2f), "div[0]");
    TEST_ASSERT(float_eq(quot_data[3], 0.5f), "div[3]");
    nt_tensor_release(quot);
    
    nt_tensor_release(a);
    nt_tensor_release(b);
    nt_context_free(ctx);
    
    TEST_PASS("binary_ops");
    return 0;
}

int test_matmul(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    /* A: 2x3, B: 3x2 -> C: 2x2 */
    nt_tensor_t* A = nt_tensor_new_2d(ctx, NT_F32, 3, 2);
    nt_tensor_t* B = nt_tensor_new_2d(ctx, NT_F32, 2, 3);
    
    float* A_data = (float*)A->data;
    float* B_data = (float*)B->data;
    
    /* A = [[1, 2, 3],
     *      [4, 5, 6]] */
    A_data[0] = 1.0f; A_data[1] = 2.0f; A_data[2] = 3.0f;
    A_data[3] = 4.0f; A_data[4] = 5.0f; A_data[5] = 6.0f;
    
    /* B = [[7, 8],
     *      [9, 10],
     *      [11, 12]] */
    B_data[0] = 7.0f;  B_data[1] = 8.0f;
    B_data[2] = 9.0f;  B_data[3] = 10.0f;
    B_data[4] = 11.0f; B_data[5] = 12.0f;
    
    nt_tensor_t* C = nt_matmul(ctx, A, B);
    TEST_ASSERT(C != NULL, "matmul operation");
    TEST_ASSERT(C->ne[0] == 2, "matmul result shape[0]");
    TEST_ASSERT(C->ne[1] == 2, "matmul result shape[1]");
    
    float* C_data = (float*)C->data;
    /* C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
     *      [4*7+5*9+6*11, 4*8+5*10+6*12]]
     *   = [[58, 64],
     *      [139, 154]] */
    TEST_ASSERT(float_eq(C_data[0], 58.0f), "matmul[0,0]");
    TEST_ASSERT(float_eq(C_data[1], 64.0f), "matmul[0,1]");
    TEST_ASSERT(float_eq(C_data[2], 139.0f), "matmul[1,0]");
    TEST_ASSERT(float_eq(C_data[3], 154.0f), "matmul[1,1]");
    
    nt_tensor_release(A);
    nt_tensor_release(B);
    nt_tensor_release(C);
    nt_context_free(ctx);
    
    TEST_PASS("matmul");
    return 0;
}

int test_softmax(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_tensor_t* x = nt_tensor_new_1d(ctx, NT_F32, 4);
    float* data = (float*)x->data;
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    
    nt_tensor_t* sm = nt_softmax(ctx, x, -1);
    TEST_ASSERT(sm != NULL, "softmax operation");
    
    float* sm_data = (float*)sm->data;
    
    /* Check that softmax sums to 1 */
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += sm_data[i];
    }
    TEST_ASSERT(float_eq(sum, 1.0f), "softmax sums to 1");
    
    /* Check that values are in increasing order */
    TEST_ASSERT(sm_data[0] < sm_data[1], "softmax[0] < softmax[1]");
    TEST_ASSERT(sm_data[1] < sm_data[2], "softmax[1] < softmax[2]");
    TEST_ASSERT(sm_data[2] < sm_data[3], "softmax[2] < softmax[3]");
    
    /* Check all values are positive */
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(sm_data[i] > 0.0f, "softmax values positive");
    }
    
    nt_tensor_release(x);
    nt_tensor_release(sm);
    nt_context_free(ctx);
    
    TEST_PASS("softmax");
    return 0;
}

int test_layer_norm(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_tensor_t* x = nt_tensor_new_1d(ctx, NT_F32, 4);
    float* data = (float*)x->data;
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    
    nt_tensor_t* ln = nt_layer_norm(ctx, x, NULL, NULL, 1e-5f);
    TEST_ASSERT(ln != NULL, "layer_norm operation");
    
    float* ln_data = (float*)ln->data;
    
    /* Check that mean is approximately 0 */
    float mean = 0.0f;
    for (int i = 0; i < 4; i++) {
        mean += ln_data[i];
    }
    mean /= 4.0f;
    TEST_ASSERT(fabsf(mean) < 1e-4f, "layer_norm mean ≈ 0");
    
    /* Check that variance is approximately 1 */
    float var = 0.0f;
    for (int i = 0; i < 4; i++) {
        var += (ln_data[i] - mean) * (ln_data[i] - mean);
    }
    var /= 4.0f;
    TEST_ASSERT(fabsf(var - 1.0f) < 0.1f, "layer_norm var ≈ 1");
    
    nt_tensor_release(x);
    nt_tensor_release(ln);
    nt_context_free(ctx);
    
    TEST_PASS("layer_norm");
    return 0;
}

int test_rms_norm(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_tensor_t* x = nt_tensor_new_1d(ctx, NT_F32, 4);
    float* data = (float*)x->data;
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    
    nt_tensor_t* rn = nt_rms_norm(ctx, x, NULL, 1e-5f);
    TEST_ASSERT(rn != NULL, "rms_norm operation");
    
    float* rn_data = (float*)rn->data;
    
    /* Check that RMS is approximately 1 */
    float rms = 0.0f;
    for (int i = 0; i < 4; i++) {
        rms += rn_data[i] * rn_data[i];
    }
    rms = sqrtf(rms / 4.0f);
    TEST_ASSERT(fabsf(rms - 1.0f) < 0.1f, "rms_norm RMS ≈ 1");
    
    nt_tensor_release(x);
    nt_tensor_release(rn);
    nt_context_free(ctx);
    
    TEST_PASS("rms_norm");
    return 0;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("NTTESHGNN Operations Tests\n");
    printf("==========================\n\n");
    
    int failed = 0;
    
    failed += test_unary_ops();
    failed += test_binary_ops();
    failed += test_matmul();
    failed += test_softmax();
    failed += test_layer_norm();
    failed += test_rms_norm();
    
    printf("\n");
    if (failed == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed.\n", failed);
    }
    
    return failed;
}
