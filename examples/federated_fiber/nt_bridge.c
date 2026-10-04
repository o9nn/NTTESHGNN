#include "nt_bridge.h"
#include "ntteshgnn/ntteshgnn.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

int nt_materialize_role_first_f32(const float *source, size_t width,
                                  float *destination, size_t dst_count) {
    if (!source || !destination || !width || width > INT32_MAX / 3 ||
        width > SIZE_MAX / width / 3 || dst_count != 3 * width * width)
        return -1;
    /* Torch's [out,in] row-major bytes are NTT/GGML ne=[in,out]. */
    int32_t shape[2] = {(int32_t)width, (int32_t)(3 * width)};
    nt_tensor_t *w = nt_tensor_from_ptr((void*)source, NT_F32, 2, shape, NULL);
    if (!w) return -2;
    const int32_t split[3] = {(int32_t)width, (int32_t)width, 3};
    nt_tensor_t *r = nt_tensor_reshape(w, 3, split);
    const int order[3] = {2, 0, 1}; /* output axis -> input axis */
    nt_tensor_t *v = r ? nt_tensor_permute(r, order) : NULL;
    nt_tensor_t *c = v ? nt_tensor_contiguous(v) : NULL;
    int result = -3;
    if (c && c->ne[0] == 3 && c->ne[1] == (int32_t)width &&
        c->ne[2] == (int32_t)width && nt_tensor_is_contiguous(c)) {
        memcpy(destination, c->data, dst_count * sizeof(float));
        result = 0;
    }
    nt_tensor_release(c);
    nt_tensor_release(v);
    nt_tensor_release(r);
    nt_tensor_release(w);
    return result;
}
