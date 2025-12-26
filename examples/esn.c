/**
 * @file esn.c
 * @brief NTTESHGNN - Echo State Network example
 * 
 * This example demonstrates creating and using an Echo State Network
 * for time series processing.
 */

#include <stdio.h>
#include <math.h>
#include "ntteshgnn/ntteshgnn.h"

/* Generate a Mackey-Glass time series (chaotic) */
void generate_mackey_glass(float* data, int length, float tau, float beta, float gamma, float n) {
    /* Initialize with constant */
    int tau_int = (int)tau;
    for (int i = 0; i < tau_int + 1; i++) {
        data[i] = 1.2f;
    }
    
    /* Generate time series */
    float dt = 0.1f;
    for (int i = tau_int + 1; i < length; i++) {
        float x_tau = data[i - tau_int];
        float x = data[i - 1];
        float dx = beta * x_tau / (1.0f + powf(x_tau, n)) - gamma * x;
        data[i] = x + dt * dx;
    }
}

int main(void) {
    printf("NTTESHGNN Echo State Network Example\n");
    printf("====================================\n\n");
    
    /* Initialize library */
    nt_init();
    nt_context_t* ctx = nt_context_new();
    
    /* ========================================
     * 1. Create reservoir
     * ======================================== */
    printf("1. Creating reservoir...\n");
    
    int input_dim = 1;
    int reservoir_size = 100;
    int output_dim = 1;
    
    nt_reservoir_t* res = nt_reservoir_new(ctx, input_dim, reservoir_size, output_dim);
    if (!res) {
        printf("Error: Failed to create reservoir\n");
        return 1;
    }
    
    /* Configure reservoir */
    nt_reservoir_set_spectral_radius(res, 0.9f);
    nt_reservoir_set_density(res, 0.1f);
    nt_reservoir_set_leaking_rate(res, 0.3f);
    nt_reservoir_set_input_scaling(res, 1.0f);
    nt_reservoir_set_activation(res, NT_ESN_TANH);
    res->warmup_steps = 100;
    
    /* Initialize weights */
    nt_reservoir_init(res, NT_ESN_INIT_RANDOM, 42);
    
    printf("  Reservoir size: %d\n", res->reservoir_size);
    printf("  Spectral radius: %.4f\n", res->spectral_radius);
    printf("  Density: %.2f\n", res->density);
    printf("  Leaking rate: %.2f\n", res->leaking_rate);
    
    /* ========================================
     * 2. Generate training data
     * ======================================== */
    printf("\n2. Generating training data...\n");
    
    int seq_length = 500;
    int washout = 100;
    int train_length = seq_length - washout;
    
    /* Generate Mackey-Glass time series */
    float* signal = (float*)malloc(seq_length * sizeof(float));
    generate_mackey_glass(signal, seq_length, 17.0f, 0.2f, 0.1f, 10.0f);
    
    /* Normalize to [-1, 1] */
    float min_val = signal[0], max_val = signal[0];
    for (int i = 1; i < seq_length; i++) {
        if (signal[i] < min_val) min_val = signal[i];
        if (signal[i] > max_val) max_val = signal[i];
    }
    float range = max_val - min_val;
    for (int i = 0; i < seq_length; i++) {
        signal[i] = 2.0f * (signal[i] - min_val) / range - 1.0f;
    }
    
    printf("  Generated %d samples of Mackey-Glass time series\n", seq_length);
    printf("  Signal range: [%.3f, %.3f]\n", -1.0f, 1.0f);
    
    /* ========================================
     * 3. Drive reservoir and collect states
     * ======================================== */
    printf("\n3. Driving reservoir...\n");
    
    /* Start state collection */
    nt_reservoir_start_collection(res, train_length);
    
    /* Create input tensor */
    nt_tensor_t* input = nt_tensor_new_1d(ctx, NT_F32, 1);
    float* in_data = (float*)input->data;
    
    /* Drive reservoir through signal */
    for (int t = 0; t < seq_length; t++) {
        in_data[0] = signal[t];
        nt_reservoir_step(res, input, NULL);
        nt_reservoir_collect_state(res);
    }
    
    printf("  Collected %u reservoir states\n", res->n_collected);
    
    /* ========================================
     * 4. Analyze reservoir dynamics
     * ======================================== */
    printf("\n4. Analyzing reservoir dynamics...\n");
    
    float* state_data = (float*)res->collected_states->data;
    
    /* Compute mean activation per neuron */
    float avg_activation = 0.0f;
    float max_activation = 0.0f;
    int active_neurons = 0;
    
    for (int i = 0; i < reservoir_size; i++) {
        float mean = 0.0f;
        for (uint32_t t = 0; t < res->n_collected; t++) {
            mean += fabsf(state_data[t * reservoir_size + i]);
        }
        mean /= res->n_collected;
        
        avg_activation += mean;
        if (mean > max_activation) {
            max_activation = mean;
        }
        if (mean > 0.1f) {
            active_neurons++;
        }
    }
    avg_activation /= reservoir_size;
    
    printf("  Average activation: %.4f\n", avg_activation);
    printf("  Max activation: %.4f\n", max_activation);
    printf("  Active neurons (>0.1): %d / %d\n", active_neurons, reservoir_size);
    
    /* ========================================
     * 5. Train readout layer
     * ======================================== */
    printf("\n5. Training readout layer...\n");
    
    /* Create target tensor (predict next step) */
    int32_t target_shape[] = {1, train_length};
    nt_tensor_t* targets = nt_tensor_new(ctx, NT_F32, 2, target_shape);
    float* target_data = (float*)targets->data;
    
    /* Target is the signal shifted by 1 step */
    for (int i = 0; i < train_length; i++) {
        int src_idx = washout + i + 1;
        if (src_idx < seq_length) {
            target_data[i] = signal[src_idx];
        } else {
            target_data[i] = signal[seq_length - 1];
        }
    }
    
    /* Train readout */
    nt_reservoir_train_readout(res, NULL, targets, 1e-6f);
    
    printf("  Trained with %d samples (washout: %d)\n", train_length, washout);
    
    /* ========================================
     * 6. Test prediction
     * ======================================== */
    printf("\n6. Testing prediction...\n");
    
    /* Reset reservoir */
    nt_reservoir_reset(res);
    
    /* Run through warmup period */
    for (int t = 0; t < washout; t++) {
        in_data[0] = signal[t];
        nt_reservoir_step(res, input, NULL);
    }
    
    /* Predict and compute error */
    float mse = 0.0f;
    int n_test = seq_length - washout - 1;
    
    printf("\n  Sample predictions (actual -> predicted):\n");
    for (int t = washout; t < seq_length - 1; t++) {
        in_data[0] = signal[t];
        nt_reservoir_step(res, input, NULL);
        
        nt_tensor_t* pred = nt_reservoir_predict(res);
        float predicted = ((float*)pred->data)[0];
        float actual = signal[t + 1];
        
        float error = predicted - actual;
        mse += error * error;
        
        /* Print first few predictions */
        if (t < washout + 5 || t >= seq_length - 4) {
            printf("    t=%3d: %.4f -> %.4f (error: %.4f)\n", 
                   t, actual, predicted, error);
        } else if (t == washout + 5) {
            printf("    ...\n");
        }
        
        nt_tensor_release(pred);
    }
    
    mse /= n_test;
    float rmse = sqrtf(mse);
    
    printf("\n  Test MSE: %.6f\n", mse);
    printf("  Test RMSE: %.6f\n", rmse);
    
    /* ========================================
     * 7. Print reservoir summary
     * ======================================== */
    printf("\n7. Reservoir summary:\n");
    nt_reservoir_print(res);
    
    /* ========================================
     * Cleanup
     * ======================================== */
    printf("\nCleaning up...\n");
    
    free(signal);
    nt_tensor_release(input);
    nt_tensor_release(targets);
    nt_reservoir_free(res);
    nt_context_free(ctx);
    nt_cleanup();
    
    printf("Done!\n");
    return 0;
}
