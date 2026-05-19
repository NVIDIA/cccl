//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.math], classification functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec = simd::rebind_t<int, Vec>;
  using Mask   = typename Vec::mask_type;
  Vec vec(math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fpclassify(vec)), IntVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isfinite(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isinf(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isnan(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isnormal(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::signbit(vec)), Mask>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::fpclassify(vec)), IntVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isfinite(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isinf(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isnan(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::isnormal(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::signbit(vec)), Mask>);

  static_assert(noexcept(cuda::std::simd::fpclassify(vec)));
  static_assert(noexcept(cuda::std::simd::isfinite(vec)));
  static_assert(noexcept(cuda::std::simd::isinf(vec)));
  static_assert(noexcept(cuda::std::simd::isnan(vec)));
  static_assert(noexcept(cuda::std::simd::isnormal(vec)));
  static_assert(noexcept(cuda::std::simd::signbit(vec)));

  IntVec classes = cuda::std::simd::fpclassify(vec);
  Mask finite    = cuda::std::simd::isfinite(vec);
  Mask inf       = cuda::std::simd::isinf(vec);
  Mask nan       = cuda::std::simd::isnan(vec);
  Mask normal    = cuda::std::simd::isnormal(vec);
  Mask signs     = cuda::std::simd::signbit(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(classes[i] == cuda::std::fpclassify(vec[i]));
    assert(finite[i] == cuda::std::isfinite(vec[i]));
    assert(inf[i] == cuda::std::isinf(vec[i]));
    assert(nan[i] == cuda::std::isnan(vec[i]));
    assert(normal[i] == cuda::std::isnormal(vec[i]));
    assert(signs[i] == cuda::std::signbit(vec[i]));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()
DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
