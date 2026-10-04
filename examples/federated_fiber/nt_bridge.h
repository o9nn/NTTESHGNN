#ifndef NT_FEDERATED_QKV_BRIDGE_H
#define NT_FEDERATED_QKV_BRIDGE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Torch7 rows [3D,D] -> scalar fibers [role,in,out], role fastest.
 * dst_count must equal 3*D*D. Returns 0 on success, nonzero on rejection. */
int nt_materialize_role_first_f32(const float *source, size_t width,
                                  float *destination, size_t dst_count);
#ifdef __cplusplus
}
#endif
#endif
