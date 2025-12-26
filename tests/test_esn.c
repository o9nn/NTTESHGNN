/**
 * @file test_esn.c
 * @brief NTTESHGNN - Echo State Network tests
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

#define EPSILON 1e-4f

/* ============================================================================
 * TESTS
 * ============================================================================ */

int test_reservoir_creation(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, 10, 100, 5);
    TEST_ASSERT(res != NULL, "reservoir creation");
    TEST_ASSERT(res->input_size == 10, "input size");
    TEST_ASSERT(res->reservoir_size == 100, "reservoir size");
    TEST_ASSERT(res->output_size == 5, "output size");
    TEST_ASSERT(res->W_res != NULL, "W_res allocated");
    TEST_ASSERT(res->W_in != NULL, "W_in allocated");
    TEST_ASSERT(res->state != NULL, "state allocated");
    TEST_ASSERT(res->bias != NULL, "bias allocated");
    
    /* Check W_res shape */
    TEST_ASSERT(res->W_res->ne[0] == 100, "W_res shape[0]");
    TEST_ASSERT(res->W_res->ne[1] == 100, "W_res shape[1]");
    
    /* Check W_in shape */
    TEST_ASSERT(res->W_in->ne[0] == 100, "W_in shape[0]");
    TEST_ASSERT(res->W_in->ne[1] == 10, "W_in shape[1]");
    
    /* Check state shape */
    TEST_ASSERT(res->state->ne[0] == 100, "state shape");
    
    nt_reservoir_free(res);
    nt_context_free(ctx);
    
    TEST_PASS("reservoir_creation");
    return 0;
}

int test_reservoir_configuration(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, 5, 50, 1);
    TEST_ASSERT(res != NULL, "reservoir creation");
    
    /* Test configuration setters */
    nt_reservoir_set_spectral_radius(res, 0.95f);
    TEST_ASSERT(fabsf(res->spectral_radius - 0.95f) < EPSILON, "spectral radius set");
    
    nt_reservoir_set_input_scaling(res, 0.5f);
    TEST_ASSERT(fabsf(res->input_scaling - 0.5f) < EPSILON, "input scaling set");
    
    nt_reservoir_set_leaking_rate(res, 0.7f);
    TEST_ASSERT(fabsf(res->leaking_rate - 0.7f) < EPSILON, "leaking rate set");
    
    nt_reservoir_set_density(res, 0.2f);
    TEST_ASSERT(fabsf(res->density - 0.2f) < EPSILON, "density set");
    
    nt_reservoir_set_activation(res, NT_ESN_RELU);
    TEST_ASSERT(res->activation == NT_ESN_RELU, "activation set");
    
    nt_reservoir_set_noise(res, 0.01f);
    TEST_ASSERT(fabsf(res->noise_level - 0.01f) < EPSILON, "noise set");
    
    nt_reservoir_free(res);
    nt_context_free(ctx);
    
    TEST_PASS("reservoir_configuration");
    return 0;
}

int test_reservoir_initialization(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, 5, 50, 1);
    TEST_ASSERT(res != NULL, "reservoir creation");
    
    nt_reservoir_set_spectral_radius(res, 0.9f);
    nt_reservoir_set_density(res, 0.1f);
    
    /* Initialize */
    nt_reservoir_init(res, NT_ESN_INIT_RANDOM, 42);
    TEST_ASSERT(res->is_initialized, "reservoir initialized");
    
    /* Check that W_res has some non-zero values */
    float* W_res = (float*)res->W_res->data;
    int non_zero = 0;
    int total = res->reservoir_size * res->reservoir_size;
    for (int i = 0; i < total; i++) {
        if (W_res[i] != 0.0f) non_zero++;
    }
    
    printf("  Non-zero weights: %d / %d (%.1f%%)\n", 
           non_zero, total, 100.0f * non_zero / total);
    
    TEST_ASSERT(non_zero > 0, "W_res has non-zero values");
    TEST_ASSERT(non_zero < total, "W_res is sparse");
    
    nt_reservoir_free(res);
    nt_context_free(ctx);
    
    TEST_PASS("reservoir_initialization");
    return 0;
}

int test_reservoir_update(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, 3, 20, 1);
    TEST_ASSERT(res != NULL, "reservoir creation");
    
    nt_reservoir_set_leaking_rate(res, 0.5f);
    nt_reservoir_init(res, NT_ESN_INIT_RANDOM, 123);
    
    /* Create input */
    nt_tensor_t* input = nt_tensor_new_1d(ctx, NT_F32, 3);
    float* in_data = (float*)input->data;
    in_data[0] = 0.5f;
    in_data[1] = -0.3f;
    in_data[2] = 0.8f;
    
    /* Initial state should be zero */
    float* state = (float*)res->state->data;
    float sum = 0.0f;
    for (int32_t i = 0; i < res->reservoir_size; i++) {
        sum += fabsf(state[i]);
    }
    TEST_ASSERT(sum == 0.0f, "initial state is zero");
    
    /* Update reservoir */
    nt_tensor_t* new_state = nt_reservoir_step(res, input, NULL);
    TEST_ASSERT(new_state != NULL, "step returns state");
    
    /* State should now be non-zero */
    sum = 0.0f;
    for (int32_t i = 0; i < res->reservoir_size; i++) {
        sum += fabsf(state[i]);
    }
    TEST_ASSERT(sum > 0.0f, "state is non-zero after update");
    
    /* State values should be in [-1, 1] due to tanh */
    for (int32_t i = 0; i < res->reservoir_size; i++) {
        TEST_ASSERT(state[i] >= -1.0f && state[i] <= 1.0f, "state in [-1,1]");
    }
    
    /* Multiple updates should change state */
    float first_state_0 = state[0];
    nt_reservoir_step(res, input, NULL);
    TEST_ASSERT(state[0] != first_state_0, "state changes with updates");
    
    nt_tensor_release(input);
    nt_reservoir_free(res);
    nt_context_free(ctx);
    
    TEST_PASS("reservoir_update");
    return 0;
}

int test_reservoir_reset(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, 2, 10, 1);
    TEST_ASSERT(res != NULL, "reservoir creation");
    
    nt_reservoir_init(res, NT_ESN_INIT_RANDOM, 456);
    
    /* Create input and update */
    nt_tensor_t* input = nt_tensor_new_1d(ctx, NT_F32, 2);
    float* in_data = (float*)input->data;
    in_data[0] = 1.0f;
    in_data[1] = 1.0f;
    
    nt_reservoir_step(res, input, NULL);
    
    /* State should be non-zero */
    float* state = (float*)res->state->data;
    float sum = 0.0f;
    for (int32_t i = 0; i < res->reservoir_size; i++) {
        sum += fabsf(state[i]);
    }
    TEST_ASSERT(sum > 0.0f, "state non-zero before reset");
    
    /* Reset */
    nt_reservoir_reset(res);
    
    /* State should be zero again */
    sum = 0.0f;
    for (int32_t i = 0; i < res->reservoir_size; i++) {
        sum += fabsf(state[i]);
    }
    TEST_ASSERT(sum == 0.0f, "state zero after reset");
    
    nt_tensor_release(input);
    nt_reservoir_free(res);
    nt_context_free(ctx);
    
    TEST_PASS("reservoir_reset");
    return 0;
}

int test_reservoir_training(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, 1, 50, 1);
    TEST_ASSERT(res != NULL, "reservoir creation");
    
    res->warmup_steps = 10;
    nt_reservoir_init(res, NT_ESN_INIT_RANDOM, 789);
    
    /* Start state collection */
    nt_reservoir_start_collection(res, 100);
    TEST_ASSERT(res->is_training, "training mode enabled");
    TEST_ASSERT(res->collected_states != NULL, "collection buffer allocated");
    
    /* Generate simple sine wave and collect states */
    nt_tensor_t* input = nt_tensor_new_1d(ctx, NT_F32, 1);
    float* in_data = (float*)input->data;
    
    for (int t = 0; t < 110; t++) {
        in_data[0] = sinf(0.1f * t);
        nt_reservoir_step(res, input, NULL);
        nt_reservoir_collect_state(res);
    }
    
    printf("  Collected %u states\n", res->n_collected);
    TEST_ASSERT(res->n_collected == 100, "collected 100 states");
    
    /* Create targets (shifted sine) */
    int32_t target_shape[] = {1, 100};
    nt_tensor_t* targets = nt_tensor_new(ctx, NT_F32, 2, target_shape);
    float* tgt_data = (float*)targets->data;
    for (int t = 0; t < 100; t++) {
        tgt_data[t] = sinf(0.1f * (t + 11));  /* Shifted by warmup + 1 */
    }
    
    /* Train readout */
    nt_reservoir_train_readout(res, NULL, targets, 1e-6f);
    TEST_ASSERT(res->W_out != NULL, "W_out created");
    TEST_ASSERT(!res->is_training, "training mode disabled");
    
    nt_tensor_release(input);
    nt_tensor_release(targets);
    nt_reservoir_free(res);
    nt_context_free(ctx);
    
    TEST_PASS("reservoir_training");
    return 0;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("NTTESHGNN Echo State Network Tests\n");
    printf("===================================\n\n");
    
    int failed = 0;
    
    failed += test_reservoir_creation();
    failed += test_reservoir_configuration();
    failed += test_reservoir_initialization();
    failed += test_reservoir_update();
    failed += test_reservoir_reset();
    failed += test_reservoir_training();
    
    printf("\n");
    if (failed == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed.\n", failed);
    }
    
    return failed;
}
