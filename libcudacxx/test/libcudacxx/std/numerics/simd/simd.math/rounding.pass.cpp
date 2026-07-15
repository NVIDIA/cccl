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

// [simd.math], rounding functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
  using LongVec  = simd::rebind_t<long, Vec>;
  using LLongVec = simd::rebind_t<long long, Vec>;

  Vec vec(math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ceil(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::floor(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nearbyint(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::rint(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::round(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::trunc(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lrint(vec)), LongVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::llrint(vec)), LLongVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lround(vec)), LongVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::llround(vec)), LLongVec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::ceil(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::floor(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::nearbyint(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::rint(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::round(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::trunc(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::lrint(vec)), LongVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::llrint(vec)), LLongVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::lround(vec)), LongVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::llround(vec)), LLongVec>);

  static_assert(noexcept(cuda::std::simd::ceil(vec)));
  static_assert(noexcept(cuda::std::simd::floor(vec)));
  static_assert(noexcept(cuda::std::simd::nearbyint(vec)));
  static_assert(noexcept(cuda::std::simd::rint(vec)));
  static_assert(noexcept(cuda::std::simd::round(vec)));
  static_assert(noexcept(cuda::std::simd::trunc(vec)));
  static_assert(noexcept(cuda::std::simd::lrint(vec)));
  static_assert(noexcept(cuda::std::simd::llrint(vec)));
  static_assert(noexcept(cuda::std::simd::lround(vec)));
  static_assert(noexcept(cuda::std::simd::llround(vec)));

  Vec ceil_result         = cuda::std::simd::ceil(vec);
  Vec floor_result        = cuda::std::simd::floor(vec);
  Vec nearbyint_result    = cuda::std::simd::nearbyint(vec);
  Vec rint_result         = cuda::std::simd::rint(vec);
  Vec round_result        = cuda::std::simd::round(vec);
  Vec trunc_result        = cuda::std::simd::trunc(vec);
  LongVec lrint_result    = cuda::std::simd::lrint(vec);
  LLongVec llrint_result  = cuda::std::simd::llrint(vec);
  LongVec lround_result   = cuda::std::simd::lround(vec);
  LLongVec llround_result = cuda::std::simd::llround(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(ceil_result[i] == cuda::std::ceil(vec[i]));
    assert(floor_result[i] == cuda::std::floor(vec[i]));
    assert(nearbyint_result[i] == cuda::std::nearbyint(vec[i]));
    assert(rint_result[i] == cuda::std::rint(vec[i]));
    assert(round_result[i] == cuda::std::round(vec[i]));
    assert(trunc_result[i] == cuda::std::trunc(vec[i]));
    assert(lrint_result[i] == cuda::std::lrint(vec[i]));
    assert(llrint_result[i] == cuda::std::llrint(vec[i]));
    assert(lround_result[i] == cuda::std::lround(vec[i]));
    assert(llround_result[i] == cuda::std::llround(vec[i]));
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
