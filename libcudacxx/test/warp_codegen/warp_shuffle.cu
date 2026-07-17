//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint> // IWYU pragma: keep
#include <cuda/warp> // IWYU pragma: keep

#define DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(type, suffix)                     \
  extern "C" __device__ __noinline__ type wrapper_idx_##suffix(type value)  \
  {                                                                         \
    return cuda::device::warp_shuffle_idx(value, 1).data;                   \
  }                                                                         \
  extern "C" __device__ __noinline__ type wrapper_up_##suffix(type value)   \
  {                                                                         \
    return cuda::device::warp_shuffle_up(value, 1).data;                    \
  }                                                                         \
  extern "C" __device__ __noinline__ type wrapper_down_##suffix(type value) \
  {                                                                         \
    return cuda::device::warp_shuffle_down(value, 1).data;                  \
  }                                                                         \
  extern "C" __device__ __noinline__ type wrapper_xor_##suffix(type value)  \
  {                                                                         \
    return cuda::device::warp_shuffle_xor(value, 1).data;                   \
  }

DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint8_t, u8)
DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint16_t, u16)
DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint32_t, u32)
DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint64_t, u64)

#undef DEFINE_WARP_SHUFFLE_CODEGEN_TESTS

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_xor_u64
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-4: {{.*SHFL.BFLY.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_down_u64
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-4: {{.*SHFL.DOWN.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_up_u64
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-4: {{.*SHFL.UP.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_idx_u64
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-4: {{.*SHFL.IDX.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_xor_u32
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.BFLY.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_down_u32
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.DOWN.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_up_u32
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.UP.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_idx_u32
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.IDX.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_xor_u16
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.BFLY.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_down_u16
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.DOWN.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_up_u16
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.UP.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_idx_u16
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.IDX.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_xor_u8
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.BFLY.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_down_u8
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.DOWN.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_up_u8
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.UP.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : wrapper_idx_u8
; SMXX-COUNT-1: {{.*PRMT.*}}
; SMXX-NOT: {{.*PRMT.*}}
; SMXX-COUNT-2: {{.*SHFL.IDX.*}}
; SMXX-NOT: {{.*(PRMT|SHFL).*$}}
; SMXX: {{.*RET.*}}

*/
