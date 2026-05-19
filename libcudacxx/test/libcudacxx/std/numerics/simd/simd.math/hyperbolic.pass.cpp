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

// [simd.math], hyperbolic functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec vec(positive_math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::acosh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::asinh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atanh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::cosh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::sinh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::tanh(vec)), Vec>);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::acosh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::asinh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::atanh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cosh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::sinh(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::tanh(vec)), Vec>);

  static_assert(noexcept(cuda::std::simd::acosh(vec)));
  static_assert(noexcept(cuda::std::simd::asinh(vec)));
  static_assert(noexcept(cuda::std::simd::atanh(vec)));
  static_assert(noexcept(cuda::std::simd::cosh(vec)));
  static_assert(noexcept(cuda::std::simd::sinh(vec)));
  static_assert(noexcept(cuda::std::simd::tanh(vec)));

  Vec acosh_arg(T{2});
  Vec atanh_arg(math_values<T>{});
  Vec acosh_result = cuda::std::simd::acosh(acosh_arg);
  Vec asinh_result = cuda::std::simd::asinh(vec);
  Vec atanh_result = cuda::std::simd::atanh(atanh_arg);
  Vec cosh_result  = cuda::std::simd::cosh(vec);
  Vec sinh_result  = cuda::std::simd::sinh(vec);
  Vec tanh_result  = cuda::std::simd::tanh(vec);
  T tolerance      = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(acosh_result[i], cuda::std::acosh(acosh_arg[i]), tolerance));
    assert(almost_equal(asinh_result[i], cuda::std::asinh(vec[i]), tolerance));
    assert(almost_equal(atanh_result[i], cuda::std::atanh(atanh_arg[i]), tolerance));
    assert(almost_equal(cosh_result[i], cuda::std::cosh(vec[i]), tolerance));
    assert(almost_equal(sinh_result[i], cuda::std::sinh(vec[i]), tolerance));
    assert(almost_equal(tanh_result[i], cuda::std::tanh(vec[i]), tolerance));
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
