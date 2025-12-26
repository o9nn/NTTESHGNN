/**
 * @file test_tensor.c
 * @brief NTTESHGNN - Tensor tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ntteshgnn/ntteshgnn.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

#define TEST_PASS(name) printf("PASS: %s\n", name)

/* ============================================================================
 * TESTS
 * ============================================================================ */

int test_tensor_creation(void) {
    /* Test 1D tensor */
    nt_tensor_t* t1 = nt_tensor_new_1d(NULL, NT_F32, 10);
    TEST_ASSERT(t1 != NULL, "1D tensor creation");
    TEST_ASSERT(t1->ndim == 1, "1D tensor ndim");
    TEST_ASSERT(t1->ne[0] == 10, "1D tensor shape");
    TEST_ASSERT(nt_tensor_numel(t1) == 10, "1D tensor numel");
    nt_tensor_release(t1);
    
    /* Test 2D tensor */
    nt_tensor_t* t2 = nt_tensor_new_2d(NULL, NT_F32, 4, 5);
    TEST_ASSERT(t2 != NULL, "2D tensor creation");
    TEST_ASSERT(t2->ndim == 2, "2D tensor ndim");
    TEST_ASSERT(t2->ne[0] == 4, "2D tensor shape[0]");
    TEST_ASSERT(t2->ne[1] == 5, "2D tensor shape[1]");
    TEST_ASSERT(nt_tensor_numel(t2) == 20, "2D tensor numel");
    nt_tensor_release(t2);
    
    /* Test 3D tensor */
    nt_tensor_t* t3 = nt_tensor_new_3d(NULL, NT_F32, 2, 3, 4);
    TEST_ASSERT(t3 != NULL, "3D tensor creation");
    TEST_ASSERT(t3->ndim == 3, "3D tensor ndim");
    TEST_ASSERT(nt_tensor_numel(t3) == 24, "3D tensor numel");
    nt_tensor_release(t3);
    
    /* Test 4D tensor */
    nt_tensor_t* t4 = nt_tensor_new_4d(NULL, NT_F32, 2, 3, 4, 5);
    TEST_ASSERT(t4 != NULL, "4D tensor creation");
    TEST_ASSERT(t4->ndim == 4, "4D tensor ndim");
    TEST_ASSERT(nt_tensor_numel(t4) == 120, "4D tensor numel");
    nt_tensor_release(t4);
    
    TEST_PASS("tensor_creation");
    return 0;
}

int test_tensor_initialization(void) {
    nt_tensor_t* t = nt_tensor_new_2d(NULL, NT_F32, 3, 4);
    TEST_ASSERT(t != NULL, "tensor creation");
    
    /* Test zero */
    nt_tensor_zero(t);
    float* data = (float*)t->data;
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT(data[i] == 0.0f, "zero initialization");
    }
    
    /* Test ones */
    nt_tensor_ones(t);
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT(data[i] == 1.0f, "ones initialization");
    }
    
    /* Test fill */
    nt_tensor_fill(t, 3.14f);
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT(fabsf(data[i] - 3.14f) < 1e-6f, "fill initialization");
    }
    
    /* Test random */
    uint64_t seed = 12345;
    nt_tensor_rand(t, &seed);
    int all_same = 1;
    for (int i = 1; i < 12; i++) {
        if (data[i] != data[0]) {
            all_same = 0;
            break;
        }
    }
    TEST_ASSERT(!all_same, "random initialization produces different values");
    
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT(data[i] >= 0.0f && data[i] <= 1.0f, "random values in [0,1]");
    }
    
    nt_tensor_release(t);
    
    TEST_PASS("tensor_initialization");
    return 0;
}

int test_tensor_views(void) {
    nt_tensor_t* t = nt_tensor_new_2d(NULL, NT_F32, 4, 6);
    TEST_ASSERT(t != NULL, "tensor creation");
    
    /* Fill with sequential values */
    float* data = (float*)t->data;
    for (int i = 0; i < 24; i++) {
        data[i] = (float)i;
    }
    
    /* Test view */
    nt_tensor_t* v = nt_tensor_view(t);
    TEST_ASSERT(v != NULL, "view creation");
    TEST_ASSERT(v->data == t->data, "view shares data");
    TEST_ASSERT(v->flags & NT_FLAG_VIEW, "view flag set");
    nt_tensor_release(v);
    
    /* Test reshape */
    int32_t new_shape[] = {2, 12};
    nt_tensor_t* r = nt_tensor_reshape(t, 2, new_shape);
    TEST_ASSERT(r != NULL, "reshape creation");
    TEST_ASSERT(r->ne[0] == 2, "reshape shape[0]");
    TEST_ASSERT(r->ne[1] == 12, "reshape shape[1]");
    TEST_ASSERT(nt_tensor_numel(r) == 24, "reshape preserves numel");
    nt_tensor_release(r);
    
    /* Test transpose */
    nt_tensor_t* tr = nt_tensor_transpose(t, 0, 1);
    TEST_ASSERT(tr != NULL, "transpose creation");
    TEST_ASSERT(tr->ne[0] == 6, "transpose shape[0]");
    TEST_ASSERT(tr->ne[1] == 4, "transpose shape[1]");
    TEST_ASSERT(!nt_tensor_is_contiguous(tr), "transpose is not contiguous");
    nt_tensor_release(tr);
    
    /* Test flatten */
    nt_tensor_t* f = nt_tensor_flatten(t);
    TEST_ASSERT(f != NULL, "flatten creation");
    TEST_ASSERT(f->ndim == 1, "flatten ndim");
    TEST_ASSERT(f->ne[0] == 24, "flatten shape");
    nt_tensor_release(f);
    
    nt_tensor_release(t);
    
    TEST_PASS("tensor_views");
    return 0;
}

int test_tensor_clone(void) {
    nt_tensor_t* t = nt_tensor_new_2d(NULL, NT_F32, 3, 4);
    TEST_ASSERT(t != NULL, "tensor creation");
    
    /* Fill with values */
    float* data = (float*)t->data;
    for (int i = 0; i < 12; i++) {
        data[i] = (float)i * 0.5f;
    }
    
    /* Clone */
    nt_tensor_t* c = nt_tensor_clone(t);
    TEST_ASSERT(c != NULL, "clone creation");
    TEST_ASSERT(c->data != t->data, "clone has separate data");
    TEST_ASSERT(c->ndim == t->ndim, "clone has same ndim");
    TEST_ASSERT(c->ne[0] == t->ne[0], "clone has same shape[0]");
    TEST_ASSERT(c->ne[1] == t->ne[1], "clone has same shape[1]");
    
    float* clone_data = (float*)c->data;
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT(clone_data[i] == data[i], "clone has same values");
    }
    
    /* Modify original, clone should be unchanged */
    data[0] = 999.0f;
    TEST_ASSERT(clone_data[0] != 999.0f, "clone is independent");
    
    nt_tensor_release(c);
    nt_tensor_release(t);
    
    TEST_PASS("tensor_clone");
    return 0;
}

int test_tensor_refcount(void) {
    nt_tensor_t* t = nt_tensor_new_1d(NULL, NT_F32, 10);
    TEST_ASSERT(t != NULL, "tensor creation");
    TEST_ASSERT(nt_tensor_refcount(t) == 1, "initial refcount is 1");
    
    /* Retain */
    nt_tensor_retain(t);
    TEST_ASSERT(nt_tensor_refcount(t) == 2, "refcount after retain");
    
    /* Release */
    nt_tensor_release(t);
    TEST_ASSERT(nt_tensor_refcount(t) == 1, "refcount after release");
    
    /* Final release */
    nt_tensor_release(t);
    /* t is now freed, don't access it */
    
    TEST_PASS("tensor_refcount");
    return 0;
}

int test_tensor_metadata(void) {
    nt_tensor_t* t = nt_tensor_new_1d(NULL, NT_F32, 10);
    TEST_ASSERT(t != NULL, "tensor creation");
    
    /* Set name */
    nt_tensor_set_name(t, "test_tensor");
    const char* name = nt_tensor_get_name(t);
    TEST_ASSERT(strcmp(name, "test_tensor") == 0, "tensor name");
    
    /* Set requires_grad */
    nt_tensor_set_requires_grad(t, true);
    TEST_ASSERT(t->flags & NT_FLAG_REQUIRES_GRAD, "requires_grad flag");
    
    nt_tensor_release(t);
    
    TEST_PASS("tensor_metadata");
    return 0;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("NTTESHGNN Tensor Tests\n");
    printf("======================\n\n");
    
    int failed = 0;
    
    failed += test_tensor_creation();
    failed += test_tensor_initialization();
    failed += test_tensor_views();
    failed += test_tensor_clone();
    failed += test_tensor_refcount();
    failed += test_tensor_metadata();
    
    printf("\n");
    if (failed == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed.\n", failed);
    }
    
    return failed;
}
