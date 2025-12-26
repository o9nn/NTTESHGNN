/**
 * @file nt_types.c
 * @brief NTTESHGNN - Type utilities implementation
 */

#include "ntteshgnn/nt_types.h"
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * DATA TYPE NAMES
 * ============================================================================ */

const char* nt_dtype_name(nt_dtype_t dt) {
    switch (dt) {
        case NT_F64:        return "f64";
        case NT_F32:        return "f32";
        case NT_F16:        return "f16";
        case NT_BF16:       return "bf16";
        case NT_F8E4M3:     return "f8e4m3";
        case NT_F8E5M2:     return "f8e5m2";
        case NT_I64:        return "i64";
        case NT_I32:        return "i32";
        case NT_I16:        return "i16";
        case NT_I8:         return "i8";
        case NT_I4:         return "i4";
        case NT_I2:         return "i2";
        case NT_I1:         return "i1";
        case NT_U64:        return "u64";
        case NT_U32:        return "u32";
        case NT_U16:        return "u16";
        case NT_U8:         return "u8";
        case NT_U4:         return "u4";
        case NT_Q8_0:       return "q8_0";
        case NT_Q8_1:       return "q8_1";
        case NT_Q5_0:       return "q5_0";
        case NT_Q5_1:       return "q5_1";
        case NT_Q4_0:       return "q4_0";
        case NT_Q4_1:       return "q4_1";
        case NT_Q4_K:       return "q4_k";
        case NT_Q6_K:       return "q6_k";
        case NT_Q2_K:       return "q2_k";
        case NT_IQ4_NL:     return "iq4_nl";
        case NT_IQ3_XXS:    return "iq3_xxs";
        case NT_IQ2_XXS:    return "iq2_xxs";
        case NT_C64:        return "c64";
        case NT_C32:        return "c32";
        case NT_C16:        return "c16";
        case NT_BOOL:       return "bool";
        case NT_NESTED:     return "nested";
        case NT_EDGE:       return "edge";
        case NT_PTR:        return "ptr";
        case NT_VOID:       return "void";
        default:            return "unknown";
    }
}

/* ============================================================================
 * DEVICE NAMES
 * ============================================================================ */

const char* nt_device_name(nt_device_t dev) {
    switch (dev) {
        case NT_DEV_CPU:    return "cpu";
        case NT_DEV_CUDA:   return "cuda";
        case NT_DEV_METAL:  return "metal";
        case NT_DEV_VULKAN: return "vulkan";
        case NT_DEV_VTNPU:  return "vtnpu";
        case NT_DEV_FPGA:   return "fpga";
        case NT_DEV_OPENCL: return "opencl";
        case NT_DEV_REMOTE: return "remote";
        default:            return "unknown";
    }
}

/* ============================================================================
 * STATUS MESSAGES
 * ============================================================================ */

const char* nt_status_str(nt_status_t status) {
    switch (status) {
        case NT_OK:                 return "success";
        case NT_ERR_INVALID_ARG:    return "invalid argument";
        case NT_ERR_OUT_OF_MEMORY:  return "out of memory";
        case NT_ERR_SHAPE_MISMATCH: return "shape mismatch";
        case NT_ERR_DTYPE_MISMATCH: return "dtype mismatch";
        case NT_ERR_DEVICE_MISMATCH:return "device mismatch";
        case NT_ERR_NOT_IMPLEMENTED:return "not implemented";
        case NT_ERR_INVALID_STATE:  return "invalid state";
        case NT_ERR_IO:             return "I/O error";
        case NT_ERR_BACKEND:        return "backend error";
        case NT_ERR_TYPE_CHECK:     return "type check failed";
        case NT_ERR_GRAPH:          return "graph error";
        case NT_ERR_OVERFLOW:       return "numeric overflow";
        case NT_ERR_UNDERFLOW:      return "numeric underflow";
        default:                    return "unknown error";
    }
}
