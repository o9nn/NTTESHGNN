#include "ntteshgnn/ntteshgnn.h"
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)

static int transpose_slice_case(void) {
    nt_tensor_t *base = nt_tensor_new_2d(NULL, NT_F32, 2, 3);
    CHECK(base);
    for (int i = 0; i < 6; ++i) ((float*)base->data)[i] = (float)(i + 1);
    nt_tensor_t *view = nt_tensor_transpose(base, 0, 1);
    CHECK(view && view->storage == base->storage && view->ne[0] == 3 && view->ne[1] == 2);
    CHECK(!nt_tensor_is_contiguous(view));
    nt_tensor_t *copy = nt_tensor_contiguous(view);
    CHECK(copy && copy->storage != base->storage && nt_tensor_is_contiguous(copy));
    const float expected[] = {1, 3, 5, 2, 4, 6};
    for (int i = 0; i < 6; ++i) CHECK(((float*)copy->data)[i] == expected[i]);
    CHECK(((float*)base->data)[1] == 2); /* no mutation through copy */

    nt_tensor_t *sub = nt_tensor_slice(view, 0, 1, 3);
    CHECK(sub && sub->storage == view->storage && sub->storage_offset == 8);
    nt_tensor_t *materialized = nt_tensor_clone(sub);
    const float sub_expected[] = {3, 5, 4, 6};
    CHECK(materialized && materialized->storage != base->storage);
    for (int i = 0; i < 4; ++i) CHECK(((float*)materialized->data)[i] == sub_expected[i]);
    nt_tensor_release(base); /* storage retained by both views */
    CHECK(*(float*)((uint8_t*)sub->data + sub->nb[0]) == 5);
    nt_tensor_release(sub);
    nt_tensor_release(view);
    nt_tensor_release(copy);
    nt_tensor_release(materialized);
    return 0;
}

static int permute_rank4_case(void) {
    nt_tensor_t *base = nt_tensor_new_4d(NULL, NT_F32, 2, 3, 2, 2);
    CHECK(base);
    for (int i = 0; i < 24; ++i) ((float*)base->data)[i] = (float)(i + 1);
    const int order[] = {2, 0, 3, 1};
    nt_tensor_t *view = nt_tensor_permute(base, order);
    CHECK(view && view->storage == base->storage);
    CHECK(view->ne[0] == 2 && view->ne[1] == 2 && view->ne[2] == 2 && view->ne[3] == 3);
    CHECK(view->nb[0] == base->nb[2] && view->nb[1] == base->nb[0] &&
          view->nb[2] == base->nb[3] && view->nb[3] == base->nb[1]);
    nt_tensor_t *copy = nt_tensor_contiguous(view);
    CHECK(copy && copy != view && copy->storage != view->storage);
    for (int i3=0; i3<3; ++i3) for (int i2=0; i2<2; ++i2)
    for (int i1=0; i1<2; ++i1) for (int i0=0; i0<2; ++i0) {
        int64_t linear = i0 + 2*(i1 + 2*(i2 + 2*i3));
        int64_t source = i1 + 2*(i3 + 3*(i0 + 2*i2));
        CHECK(((float*)copy->data)[linear] == ((float*)base->data)[source]);
    }
    const int inverse[] = {1, 3, 0, 2};
    nt_tensor_t *restore = nt_tensor_permute(view, inverse);
    nt_tensor_t *back = nt_tensor_clone(restore);
    CHECK(back && back->ndim == base->ndim);
    for (int i=0; i<24; ++i) CHECK(((float*)back->data)[i] == ((float*)base->data)[i]);
    nt_tensor_release(base); nt_tensor_release(view); nt_tensor_release(copy);
    nt_tensor_release(restore); nt_tensor_release(back);
    return 0;
}

static int invalid_layout_case(void) {
    nt_tensor_t *base = nt_tensor_new_2d(NULL, NT_F32, 2, 3);
    CHECK(base);
    CHECK(!nt_tensor_new_2d(NULL, NT_F32, -1, 3));
    CHECK(!nt_tensor_new_2d(NULL, NT_F32, INT32_MAX, INT32_MAX));
    const int32_t bad_shape[] = {0, 3};
    CHECK(!nt_tensor_reshape(base, 2, bad_shape));
    const int duplicate[] = {0,0}, invalid[] = {0,2}, identity[] = {0,1};
    CHECK(!nt_tensor_permute(base, duplicate));
    CHECK(!nt_tensor_permute(base, invalid));
    CHECK(!nt_tensor_permute(base, NULL));
    CHECK(!nt_tensor_slice(base, 0, 1, 1));
    CHECK(!nt_tensor_slice(base, 1, 2, 4));
    nt_tensor_t *same_view = nt_tensor_permute(base, identity);
    CHECK(same_view && same_view->storage == base->storage && nt_tensor_is_contiguous(same_view));
    nt_tensor_t *same = nt_tensor_contiguous(same_view);
    CHECK(same == same_view);
    nt_tensor_release(same); nt_tensor_release(same_view);
    int32_t dims[] = {2, 2}, strides[] = {4, 16};
    CHECK(!nt_tensor_from_storage(base->storage, 8, NT_F32, 2, dims, strides));
    CHECK(!nt_tensor_from_ptr(base->data, NT_F32, 2, base->ne, strides));
    nt_tensor_release(base);
    return 0;
}

int main(void) {
    if (transpose_slice_case() || permute_rank4_case() || invalid_layout_case()) return 1;
    puts("PASS: transposed 2x3, offset slice, rank-4 permute/inverse, invalid-layout guards");
    return 0;
}
