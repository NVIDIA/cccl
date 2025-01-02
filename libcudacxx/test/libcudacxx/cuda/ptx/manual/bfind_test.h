//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: clang && !nvcc

// <cuda/ptx>

template <typename T, typename F>
__device__ auto to_void(T, F x)
{
  return reinterpret_cast<void*>(static_cast<uint32_t (*)(T, cuda::ptx::bfind_shift_amount)>(x));
}

__global__ void test_bfind(void** fn_ptr)
{
#if __cccl_ptx_isa >= 200
  // auto to_void = [] __device__(auto type, auto x) {
  //   using T = decltype(type);
  //   return reinterpret_cast<void*>(static_cast<uint32_t (*)(T, cuda::ptx::bfind_shift_amount)>(x));
  // };
  //  mov.u32 sreg_value, %%tid.x;
  *fn_ptr++ = to_void(uint32_t{}, cuda::ptx::bfind<uint32_t>);
  *fn_ptr++ = to_void(uint64_t{}, cuda::ptx::bfind<uint64_t>);
  *fn_ptr++ = to_void(int32_t{}, cuda::ptx::bfind<int32_t>);
  *fn_ptr++ = to_void(int64_t{}, cuda::ptx::bfind<int64_t>);
#endif // __cccl_ptx_isa >= 200
}
