//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint>
#include <cuda/warp>

#define DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(type, suffix)                     \
  extern "C" __device__ __noinline__ type wrapper_idx_##suffix(type value)  \
  {                                                                         \
    return cuda::device::warp_shuffle_idx(value, 1).data;                   \
  }                                                                         \
  extern "C" __device__ __noinline__ type native_idx_##suffix(type value)   \
  {                                                                         \
    return __shfl_sync(0xFFFFFFFFu, value, 1);                              \
  }                                                                         \
  extern "C" __device__ __noinline__ type wrapper_up_##suffix(type value)   \
  {                                                                         \
    return cuda::device::warp_shuffle_up(value, 1).data;                    \
  }                                                                         \
  extern "C" __device__ __noinline__ type native_up_##suffix(type value)    \
  {                                                                         \
    return __shfl_up_sync(0xFFFFFFFFu, value, 1);                           \
  }                                                                         \
  extern "C" __device__ __noinline__ type wrapper_down_##suffix(type value) \
  {                                                                         \
    return cuda::device::warp_shuffle_down(value, 1).data;                  \
  }                                                                         \
  extern "C" __device__ __noinline__ type native_down_##suffix(type value)  \
  {                                                                         \
    return __shfl_down_sync(0xFFFFFFFFu, value, 1);                         \
  }                                                                         \
  extern "C" __device__ __noinline__ type wrapper_xor_##suffix(type value)  \
  {                                                                         \
    return cuda::device::warp_shuffle_xor(value, 1).data;                   \
  }                                                                         \
  extern "C" __device__ __noinline__ type native_xor_##suffix(type value)   \
  {                                                                         \
    return __shfl_xor_sync(0xFFFFFFFFu, value, 1);                          \
  }

DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint8_t, u8)
DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint16_t, u16)
DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint32_t, u32)
DEFINE_WARP_SHUFFLE_CODEGEN_TESTS(uint64_t, u64)

#undef DEFINE_WARP_SHUFFLE_CODEGEN_TESTS
