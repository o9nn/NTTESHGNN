/**
 * @file nt_echostate.c
 * @brief NTTESHGNN - Echo State Network implementation
 */

#include "ntteshgnn/nt_echostate.h"
#include "ntteshgnn/nt_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

/* Simple xorshift128+ RNG */
static uint64_t xorshift128plus(uint64_t* state) {
    uint64_t s1 = state[0];
    uint64_t s0 = state[1];
    state[0] = s0;
    s1 ^= s1 << 23;
    state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return state[1] + s0;
}

static float rand_uniform(uint64_t* state) {
    return (float)(xorshift128plus(state) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
}

static float rand_uniform_range(uint64_t* state, float min, float max) {
    return min + (max - min) * rand_uniform(state);
}

/* ============================================================================
 * RESERVOIR LIFECYCLE
 * ============================================================================ */

nt_reservoir_t* nt_reservoir_new(nt_context_t* ctx,
                                  int32_t input_size,
                                  int32_t reservoir_size,
                                  int32_t output_size) {
    nt_reservoir_t* res = (nt_reservoir_t*)calloc(1, sizeof(nt_reservoir_t));
    if (!res) return NULL;
    
    res->input_size = input_size;
    res->reservoir_size = reservoir_size;
    res->output_size = output_size;
    res->ctx = ctx;
    
    /* Create state tensors */
    int32_t state_shape[] = {reservoir_size};
    res->state = nt_tensor_new(ctx, NT_F32, 1, state_shape);
    res->prev_state = nt_tensor_new(ctx, NT_F32, 1, state_shape);
    
    if (!res->state || !res->prev_state) {
        nt_reservoir_free(res);
        return NULL;
    }
    
    nt_tensor_zero(res->state);
    nt_tensor_zero(res->prev_state);
    
    /* Create weight matrices */
    int32_t W_in_shape[] = {reservoir_size, input_size};
    res->W_in = nt_tensor_new(ctx, NT_F32, 2, W_in_shape);
    
    int32_t W_res_shape[] = {reservoir_size, reservoir_size};
    res->W_res = nt_tensor_new(ctx, NT_F32, 2, W_res_shape);
    
    int32_t bias_shape[] = {reservoir_size};
    res->bias = nt_tensor_new(ctx, NT_F32, 1, bias_shape);
    
    if (!res->W_in || !res->W_res || !res->bias) {
        nt_reservoir_free(res);
        return NULL;
    }
    
    /* Default hyperparameters */
    res->spectral_radius = 0.9f;
    res->input_scaling = 1.0f;
    res->feedback_scaling = 0.0f;
    res->leaking_rate = 0.3f;
    res->density = 0.1f;
    res->noise_level = 0.0f;
    
    res->activation = NT_ESN_TANH;
    res->init_type = NT_ESN_INIT_RANDOM;
    res->use_feedback = false;
    res->use_bias = true;
    res->warmup_steps = 100;
    
    res->is_initialized = false;
    res->is_training = false;
    res->step_count = 0;
    
    /* Initialize RNG state */
    res->rng_state[0] = 12345;
    res->rng_state[1] = 67890;
    
    return res;
}

void nt_reservoir_free(nt_reservoir_t* res) {
    if (!res) return;
    
    if (res->state) nt_tensor_release(res->state);
    if (res->prev_state) nt_tensor_release(res->prev_state);
    if (res->W_in) nt_tensor_release(res->W_in);
    if (res->W_res) nt_tensor_release(res->W_res);
    if (res->W_fb) nt_tensor_release(res->W_fb);
    if (res->bias) nt_tensor_release(res->bias);
    if (res->W_out) nt_tensor_release(res->W_out);
    if (res->b_out) nt_tensor_release(res->b_out);
    if (res->collected_states) nt_tensor_release(res->collected_states);
    
    free(res);
}

void nt_reservoir_reset(nt_reservoir_t* res) {
    if (!res) return;
    
    if (res->state) nt_tensor_zero(res->state);
    if (res->prev_state) nt_tensor_zero(res->prev_state);
    res->step_count = 0;
}

/* ============================================================================
 * RESERVOIR CONFIGURATION
 * ============================================================================ */

void nt_reservoir_set_spectral_radius(nt_reservoir_t* res, float sr) {
    if (res) res->spectral_radius = sr;
}

void nt_reservoir_set_input_scaling(nt_reservoir_t* res, float scale) {
    if (res) res->input_scaling = scale;
}

void nt_reservoir_set_leaking_rate(nt_reservoir_t* res, float alpha) {
    if (res) res->leaking_rate = alpha;
}

void nt_reservoir_set_density(nt_reservoir_t* res, float density) {
    if (res) res->density = density;
}

void nt_reservoir_set_activation(nt_reservoir_t* res, nt_esn_activation_t act) {
    if (res) res->activation = act;
}

void nt_reservoir_set_feedback(nt_reservoir_t* res, bool enable, float scaling) {
    if (!res) return;
    res->use_feedback = enable;
    res->feedback_scaling = scaling;
    
    if (enable && !res->W_fb) {
        int32_t W_fb_shape[] = {res->reservoir_size, res->output_size};
        res->W_fb = nt_tensor_new(res->ctx, NT_F32, 2, W_fb_shape);
    }
}

void nt_reservoir_set_noise(nt_reservoir_t* res, float noise) {
    if (res) res->noise_level = noise;
}

/* ============================================================================
 * RESERVOIR INITIALIZATION
 * ============================================================================ */

/* Compute spectral radius using power iteration */
static float compute_spectral_radius(const float* W, int32_t n) {
    float* v = (float*)malloc(n * sizeof(float));
    float* v_new = (float*)malloc(n * sizeof(float));
    
    if (!v || !v_new) {
        if (v) free(v);
        if (v_new) free(v_new);
        return 0.0f;
    }
    
    /* Initialize with random vector */
    uint64_t rng[2] = {12345, 67890};
    for (int32_t i = 0; i < n; i++) {
        v[i] = rand_uniform(rng);
    }
    
    /* Normalize */
    float norm = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrtf(norm);
    for (int32_t i = 0; i < n; i++) {
        v[i] /= norm;
    }
    
    /* Power iteration */
    float eigenvalue = 0.0f;
    for (int iter = 0; iter < 100; iter++) {
        /* v_new = W @ v */
        for (int32_t i = 0; i < n; i++) {
            v_new[i] = 0.0f;
            for (int32_t j = 0; j < n; j++) {
                v_new[i] += W[i * n + j] * v[j];
            }
        }
        
        /* Compute norm (eigenvalue estimate) */
        norm = 0.0f;
        for (int32_t i = 0; i < n; i++) {
            norm += v_new[i] * v_new[i];
        }
        norm = sqrtf(norm);
        eigenvalue = norm;
        
        /* Normalize */
        if (norm > 1e-10f) {
            for (int32_t i = 0; i < n; i++) {
                v[i] = v_new[i] / norm;
            }
        }
    }
    
    free(v);
    free(v_new);
    
    return eigenvalue;
}

void nt_reservoir_init_random(nt_reservoir_t* res, uint64_t seed) {
    if (!res) return;
    
    res->rng_state[0] = seed;
    res->rng_state[1] = seed ^ 0xDEADBEEF;
    
    int32_t n = res->reservoir_size;
    int32_t m = res->input_size;
    
    /* Initialize W_in */
    float* W_in = (float*)res->W_in->data;
    for (int32_t i = 0; i < n * m; i++) {
        W_in[i] = rand_uniform_range(res->rng_state, -1.0f, 1.0f) * res->input_scaling;
    }
    
    /* Initialize W_res with sparsity */
    float* W_res = (float*)res->W_res->data;
    for (int32_t i = 0; i < n * n; i++) {
        if (rand_uniform(res->rng_state) < res->density) {
            W_res[i] = rand_uniform_range(res->rng_state, -1.0f, 1.0f);
        } else {
            W_res[i] = 0.0f;
        }
    }
    
    /* Scale to desired spectral radius */
    float current_sr = compute_spectral_radius(W_res, n);
    if (current_sr > 1e-6f) {
        float scale = res->spectral_radius / current_sr;
        for (int32_t i = 0; i < n * n; i++) {
            W_res[i] *= scale;
        }
    }
    
    /* Initialize bias */
    if (res->use_bias && res->bias) {
        float* bias = (float*)res->bias->data;
        for (int32_t i = 0; i < n; i++) {
            bias[i] = rand_uniform_range(res->rng_state, -0.1f, 0.1f);
        }
    }
    
    /* Initialize feedback weights if used */
    if (res->use_feedback && res->W_fb) {
        float* W_fb = (float*)res->W_fb->data;
        for (int32_t i = 0; i < n * res->output_size; i++) {
            W_fb[i] = rand_uniform_range(res->rng_state, -1.0f, 1.0f) * res->feedback_scaling;
        }
    }
    
    res->is_initialized = true;
}

void nt_reservoir_init(nt_reservoir_t* res, nt_esn_init_t init_type, uint64_t seed) {
    if (!res) return;
    
    res->init_type = init_type;
    
    switch (init_type) {
        case NT_ESN_INIT_RANDOM:
        default:
            nt_reservoir_init_random(res, seed);
            break;
    }
}

/* ============================================================================
 * RESERVOIR UPDATE
 * ============================================================================ */

static void apply_activation(float* data, int32_t n, nt_esn_activation_t act) {
    switch (act) {
        case NT_ESN_TANH:
            for (int32_t i = 0; i < n; i++) {
                data[i] = tanhf(data[i]);
            }
            break;
        case NT_ESN_RELU:
            for (int32_t i = 0; i < n; i++) {
                data[i] = data[i] > 0.0f ? data[i] : 0.0f;
            }
            break;
        case NT_ESN_SIGMOID:
            for (int32_t i = 0; i < n; i++) {
                data[i] = 1.0f / (1.0f + expf(-data[i]));
            }
            break;
        case NT_ESN_LEAKY_RELU:
            for (int32_t i = 0; i < n; i++) {
                data[i] = data[i] > 0.0f ? data[i] : 0.01f * data[i];
            }
            break;
        case NT_ESN_IDENTITY:
        default:
            break;
    }
}

nt_tensor_t* nt_reservoir_step(nt_reservoir_t* res, const nt_tensor_t* input,
                                const nt_tensor_t* feedback) {
    if (!res || !input || !res->is_initialized) return NULL;
    
    int32_t n = res->reservoir_size;
    int32_t m = res->input_size;
    
    float* state = (float*)res->state->data;
    float* prev_state = (float*)res->prev_state->data;
    float* W_in = (float*)res->W_in->data;
    float* W_res = (float*)res->W_res->data;
    const float* x = (const float*)input->data;
    
    /* Save previous state */
    memcpy(prev_state, state, n * sizeof(float));
    
    /* Compute new state: s = act(W_in @ x + W_res @ s_prev + bias) */
    float* new_state = (float*)malloc(n * sizeof(float));
    if (!new_state) return NULL;
    
    for (int32_t i = 0; i < n; i++) {
        float sum = 0.0f;
        
        /* W_in @ x */
        for (int32_t j = 0; j < m; j++) {
            sum += W_in[i * m + j] * x[j];
        }
        
        /* W_res @ s_prev */
        for (int32_t j = 0; j < n; j++) {
            sum += W_res[i * n + j] * prev_state[j];
        }
        
        /* Feedback */
        if (res->use_feedback && feedback && res->W_fb) {
            float* W_fb = (float*)res->W_fb->data;
            const float* fb = (const float*)feedback->data;
            for (int32_t j = 0; j < res->output_size; j++) {
                sum += W_fb[i * res->output_size + j] * fb[j];
            }
        }
        
        /* Bias */
        if (res->use_bias && res->bias) {
            sum += ((float*)res->bias->data)[i];
        }
        
        /* Noise */
        if (res->noise_level > 0.0f) {
            sum += res->noise_level * rand_uniform_range(res->rng_state, -1.0f, 1.0f);
        }
        
        new_state[i] = sum;
    }
    
    /* Apply activation */
    apply_activation(new_state, n, res->activation);
    
    /* Leaky integration */
    float alpha = res->leaking_rate;
    for (int32_t i = 0; i < n; i++) {
        state[i] = (1.0f - alpha) * prev_state[i] + alpha * new_state[i];
    }
    
    free(new_state);
    res->step_count++;
    
    return res->state;
}

/* ============================================================================
 * RESERVOIR TRAINING
 * ============================================================================ */

void nt_reservoir_start_collection(nt_reservoir_t* res, uint32_t capacity) {
    if (!res) return;
    
    if (res->collected_states) {
        nt_tensor_release(res->collected_states);
    }
    
    int32_t shape[] = {res->reservoir_size, (int32_t)capacity};
    res->collected_states = nt_tensor_new(res->ctx, NT_F32, 2, shape);
    res->n_collected = 0;
    res->collection_capacity = capacity;
    res->is_training = true;
}

void nt_reservoir_collect_state(nt_reservoir_t* res) {
    if (!res || !res->is_training || !res->collected_states) return;
    if (res->n_collected >= res->collection_capacity) return;
    if (res->step_count < res->warmup_steps) return;
    
    float* states = (float*)res->collected_states->data;
    float* state = (float*)res->state->data;
    
    memcpy(states + res->n_collected * res->reservoir_size,
           state, res->reservoir_size * sizeof(float));
    
    res->n_collected++;
}

void nt_reservoir_train_readout(nt_reservoir_t* res, 
                                 const nt_tensor_t* inputs,
                                 const nt_tensor_t* targets,
                                 float regularization) {
    (void)inputs;  /* Currently unused - states already collected */
    if (!res || !targets || !res->collected_states) return;
    if (res->n_collected == 0) return;
    
    int32_t n = res->reservoir_size;
    int32_t p = res->output_size;
    uint32_t T = res->n_collected;
    
    /* Create output weights if not exist */
    if (!res->W_out) {
        int32_t W_out_shape[] = {p, n};
        res->W_out = nt_tensor_new(res->ctx, NT_F32, 2, W_out_shape);
    }
    
    /* Simple ridge regression: W_out = targets @ states^T @ (states @ states^T + λI)^-1 */
    /* For simplicity, use pseudo-inverse approximation */
    
    float* states = (float*)res->collected_states->data;
    const float* tgt = (const float*)targets->data;
    float* W_out = (float*)res->W_out->data;
    
    /* Compute states @ states^T + λI */
    float* StS = (float*)calloc(n * n, sizeof(float));
    if (!StS) return;
    
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (uint32_t t = 0; t < T; t++) {
                sum += states[t * n + i] * states[t * n + j];
            }
            StS[i * n + j] = sum;
            if (i == j) {
                StS[i * n + j] += regularization;
            }
        }
    }
    
    /* Compute targets @ states^T */
    float* TtS = (float*)calloc(p * n, sizeof(float));
    if (!TtS) {
        free(StS);
        return;
    }
    
    for (int32_t i = 0; i < p; i++) {
        for (int32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (uint32_t t = 0; t < T; t++) {
                sum += tgt[t * p + i] * states[t * n + j];
            }
            TtS[i * n + j] = sum;
        }
    }
    
    /* Copy TtS as approximation (should use proper solver) */
    memcpy(W_out, TtS, p * n * sizeof(float));
    
    free(StS);
    free(TtS);
    
    res->is_training = false;
}

/* ============================================================================
 * RESERVOIR PREDICTION
 * ============================================================================ */

nt_tensor_t* nt_reservoir_predict(nt_reservoir_t* res) {
    if (!res || !res->W_out) return NULL;
    
    int32_t n = res->reservoir_size;
    int32_t p = res->output_size;
    
    int32_t out_shape[] = {p};
    nt_tensor_t* output = nt_tensor_new(res->ctx, NT_F32, 1, out_shape);
    if (!output) return NULL;
    
    float* y = (float*)output->data;
    float* W_out = (float*)res->W_out->data;
    float* state = (float*)res->state->data;
    
    for (int32_t i = 0; i < p; i++) {
        y[i] = 0.0f;
        for (int32_t j = 0; j < n; j++) {
            y[i] += W_out[i * n + j] * state[j];
        }
        if (res->b_out) {
            y[i] += ((float*)res->b_out->data)[i];
        }
    }
    
    return output;
}

/* ============================================================================
 * PRINTING
 * ============================================================================ */

void nt_reservoir_print(const nt_reservoir_t* res) {
    if (!res) {
        printf("Reservoir: NULL\n");
        return;
    }
    
    printf("Reservoir:\n");
    printf("  Input size: %d\n", res->input_size);
    printf("  Reservoir size: %d\n", res->reservoir_size);
    printf("  Output size: %d\n", res->output_size);
    printf("  Spectral radius: %.4f\n", res->spectral_radius);
    printf("  Density: %.4f\n", res->density);
    printf("  Leaking rate: %.4f\n", res->leaking_rate);
    printf("  Input scaling: %.4f\n", res->input_scaling);
    printf("  Initialized: %s\n", res->is_initialized ? "yes" : "no");
    printf("  Steps: %lu\n", res->step_count);
}
