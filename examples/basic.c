/**
 * @file basic.c
 * @brief NTTESHGNN - Basic usage example
 * 
 * This example demonstrates the fundamental tensor operations
 * and memory management in NTTESHGNN.
 */

#include <stdio.h>
#include "ntteshgnn/ntteshgnn.h"

int main(void) {
    printf("NTTESHGNN Basic Example\n");
    printf("=======================\n\n");
    
    /* Initialize library */
    nt_init();
    
    /* Create a context for memory management */
    nt_context_t* ctx = nt_context_new();
    
    /* ========================================
     * 1. Tensor Creation
     * ======================================== */
    printf("1. Creating tensors...\n");
    
    /* Create a 2D tensor (matrix) */
    nt_tensor_t* A = nt_tensor_new_2d(ctx, NT_F32, 3, 4);
    nt_tensor_set_name(A, "A");
    
    /* Fill with sequential values */
    float* A_data = (float*)A->data;
    for (int i = 0; i < 12; i++) {
        A_data[i] = (float)i;
    }
    
    printf("Matrix A:\n");
    nt_tensor_print(A);
    
    /* Create another tensor */
    nt_tensor_t* B = nt_tensor_new_2d(ctx, NT_F32, 3, 4);
    nt_tensor_set_name(B, "B");
    nt_tensor_fill(B, 2.0f);
    
    printf("\nMatrix B (filled with 2.0):\n");
    nt_tensor_print(B);
    
    /* ========================================
     * 2. Element-wise Operations
     * ======================================== */
    printf("\n2. Element-wise operations...\n");
    
    /* Addition */
    nt_tensor_t* C = nt_add(ctx, A, B);
    nt_tensor_set_name(C, "C = A + B");
    printf("\nC = A + B:\n");
    nt_tensor_print(C);
    
    /* Multiplication */
    nt_tensor_t* D = nt_mul(ctx, A, B);
    nt_tensor_set_name(D, "D = A * B");
    printf("\nD = A * B:\n");
    nt_tensor_print(D);
    
    /* ========================================
     * 3. Activation Functions
     * ======================================== */
    printf("\n3. Activation functions...\n");
    
    /* Create input for activations */
    nt_tensor_t* x = nt_tensor_new_1d(ctx, NT_F32, 5);
    float* x_data = (float*)x->data;
    x_data[0] = -2.0f;
    x_data[1] = -1.0f;
    x_data[2] = 0.0f;
    x_data[3] = 1.0f;
    x_data[4] = 2.0f;
    
    printf("\nInput x:\n");
    nt_tensor_print(x);
    
    /* ReLU */
    nt_tensor_t* relu_x = nt_relu(ctx, x);
    printf("\nReLU(x):\n");
    nt_tensor_print(relu_x);
    
    /* Sigmoid */
    nt_tensor_t* sig_x = nt_sigmoid(ctx, x);
    printf("\nSigmoid(x):\n");
    nt_tensor_print(sig_x);
    
    /* Tanh */
    nt_tensor_t* tanh_x = nt_tanh(ctx, x);
    printf("\nTanh(x):\n");
    nt_tensor_print(tanh_x);
    
    /* GELU */
    nt_tensor_t* gelu_x = nt_gelu(ctx, x);
    printf("\nGELU(x):\n");
    nt_tensor_print(gelu_x);
    
    /* ========================================
     * 4. Matrix Multiplication
     * ======================================== */
    printf("\n4. Matrix multiplication...\n");
    
    /* Create matrices for matmul */
    nt_tensor_t* M1 = nt_tensor_new_2d(ctx, NT_F32, 3, 2);  /* 2x3 */
    nt_tensor_t* M2 = nt_tensor_new_2d(ctx, NT_F32, 4, 3);  /* 3x4 */
    
    float* M1_data = (float*)M1->data;
    float* M2_data = (float*)M2->data;
    
    /* M1 = [[1, 2, 3], [4, 5, 6]] */
    M1_data[0] = 1; M1_data[1] = 2; M1_data[2] = 3;
    M1_data[3] = 4; M1_data[4] = 5; M1_data[5] = 6;
    
    /* M2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] */
    for (int i = 0; i < 12; i++) {
        M2_data[i] = (float)(i + 1);
    }
    
    printf("\nM1 (2x3):\n");
    nt_tensor_print(M1);
    
    printf("\nM2 (3x4):\n");
    nt_tensor_print(M2);
    
    nt_tensor_t* M3 = nt_matmul(ctx, M1, M2);
    printf("\nM1 @ M2 (2x4):\n");
    nt_tensor_print(M3);
    
    /* ========================================
     * 5. Normalization
     * ======================================== */
    printf("\n5. Normalization...\n");
    
    nt_tensor_t* norm_input = nt_tensor_new_1d(ctx, NT_F32, 4);
    float* norm_data = (float*)norm_input->data;
    norm_data[0] = 1.0f;
    norm_data[1] = 2.0f;
    norm_data[2] = 3.0f;
    norm_data[3] = 4.0f;
    
    printf("\nInput:\n");
    nt_tensor_print(norm_input);
    
    nt_tensor_t* ln = nt_layer_norm(ctx, norm_input, NULL, NULL, 1e-5f);
    printf("\nLayer Norm:\n");
    nt_tensor_print(ln);
    
    nt_tensor_t* rn = nt_rms_norm(ctx, norm_input, NULL, 1e-5f);
    printf("\nRMS Norm:\n");
    nt_tensor_print(rn);
    
    /* ========================================
     * 6. Softmax
     * ======================================== */
    printf("\n6. Softmax...\n");
    
    nt_tensor_t* logits = nt_tensor_new_1d(ctx, NT_F32, 4);
    float* logits_data = (float*)logits->data;
    logits_data[0] = 1.0f;
    logits_data[1] = 2.0f;
    logits_data[2] = 3.0f;
    logits_data[3] = 4.0f;
    
    printf("\nLogits:\n");
    nt_tensor_print(logits);
    
    nt_tensor_t* probs = nt_softmax(ctx, logits, -1);
    printf("\nSoftmax probabilities:\n");
    nt_tensor_print(probs);
    
    /* ========================================
     * 7. Context Statistics
     * ======================================== */
    printf("\n7. Context statistics...\n");
    nt_context_print_stats(ctx);
    
    /* ========================================
     * Cleanup
     * ======================================== */
    printf("\nCleaning up...\n");
    
    /* Release tensors */
    nt_tensor_release(A);
    nt_tensor_release(B);
    nt_tensor_release(C);
    nt_tensor_release(D);
    nt_tensor_release(x);
    nt_tensor_release(relu_x);
    nt_tensor_release(sig_x);
    nt_tensor_release(tanh_x);
    nt_tensor_release(gelu_x);
    nt_tensor_release(M1);
    nt_tensor_release(M2);
    nt_tensor_release(M3);
    nt_tensor_release(norm_input);
    nt_tensor_release(ln);
    nt_tensor_release(rn);
    nt_tensor_release(logits);
    nt_tensor_release(probs);
    
    /* Free context */
    nt_context_free(ctx);
    
    /* Cleanup library */
    nt_cleanup();
    
    printf("\nDone!\n");
    return 0;
}
