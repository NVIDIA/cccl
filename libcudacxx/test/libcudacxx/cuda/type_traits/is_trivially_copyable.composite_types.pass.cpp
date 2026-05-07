//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/__complex_>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_copyable()
{
  static_assert(cuda::is_trivially_copyable<T>::value);
  static_assert(cuda::is_trivially_copyable<const T>::value);
  static_assert(cuda::is_trivially_copyable_v<T>);
  static_assert(cuda::is_trivially_copyable_v<const T>);
}

template <class T>
struct TriviallyCopyableWrapper
{
  T x;
};

struct TrivialPod
{
  int x;
  float y;
};

class NonTriviallyCopyable
{
public:
  __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {} // NOLINT
};

template <class T>
__host__ __device__ void test_is_trivially_copyable_compositions()
{
  test_is_trivially_copyable<T[4]>();
  test_is_trivially_copyable<cuda::std::array<T, 4>>();
  test_is_trivially_copyable<cuda::std::pair<T, T>>();
  test_is_trivially_copyable<cuda::std::tuple<T, T>>();
  test_is_trivially_copyable<cuda::std::complex<T>>();
  test_is_trivially_copyable<cuda::complex<T>>();
  test_is_trivially_copyable<TriviallyCopyableWrapper<T>>();
}

__host__ __device__ void test_composite_types()
{
  test_is_trivially_copyable<int[4]>();

  test_is_trivially_copyable<TrivialPod>();
  test_is_trivially_copyable<TrivialPod[2]>();

  // cuda::std::array, pair, tuple, complex, and aggregate wrappers of trivially copyable types
  test_is_trivially_copyable_compositions<int>();
  test_is_trivially_copyable_compositions<float>();
  test_is_trivially_copyable<cuda::std::tuple<>>();

  // non-trivially copyable types
  static_assert(!cuda::is_trivially_copyable_v<NonTriviallyCopyable>);
}

__host__ __device__ void test_extended_fp_types()
{
#if _CCCL_HAS_NVFP16()
  test_is_trivially_copyable_compositions<__half>();
  test_is_trivially_copyable_compositions<__half2>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test_is_trivially_copyable_compositions<__nv_bfloat16>();
  test_is_trivially_copyable_compositions<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
}

int main(int, char**)
{
  test_composite_types();
  test_extended_fp_types();
  return 0;
}
