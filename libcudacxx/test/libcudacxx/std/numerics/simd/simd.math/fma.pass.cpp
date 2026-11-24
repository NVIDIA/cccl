//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// <cuda/std/__simd_>

// [simd.math], fma

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "math_test_utils.h"

// NVCC/MSVC C++17 misresolves dependent qualified math calls. We use a local using-declaration and ADL as workaround
namespace test_detail
{
using simd::fma;

struct fma_function
{
  template <typename X, typename Y, typename Z>
  TEST_HOST_DEVICE_FUNC auto operator()(const X& x, const Y& y, const Z& z) const -> decltype(fma(x, y, z));
};
} // namespace test_detail

using test_detail::fma_function;

template <int N>
TEST_HOST_DEVICE_FUNC void test_mixed_arguments()
{
  // Overload participation.
  using F              = simd::vec<float, 3>;
  using D              = simd::vec<double, 3>;
  using Other          = simd::vec<double, 4>;
  using math_wrapper_t = explicit_math_wrapper<D>;

  static_assert(cuda::std::is_invocable_v<fma_function, math_wrapper_t, F, D>);
  static_assert(cuda::std::is_invocable_v<fma_function, F, D, F>);
  static_assert(!cuda::std::is_invocable_v<fma_function, F, D, Other>);
  static_assert(!cuda::std::is_invocable_v<fma_function, F, Other, D>);
  static_assert(!cuda::std::is_invocable_v<fma_function, Other, F, D>);
  static_assert(!cuda::std::is_invocable_v<fma_function, F, F, double>);
  static_assert(!cuda::std::is_invocable_v<fma_function, double, double, double>);

  // Mixed argument types.
  const F x(4.0f);
  const D y(2.0);
  const F z(2.0f);
  const D fused0 = simd::fma(y, x, z);
  const D fused1 = simd::fma(x, y, z);
  check_math_values(fused0, 10.0);
  check_math_values(fused1, 10.0);
  // Mixed wrapper and vector arguments.
  const math_wrapper<D> y1{D(2.0)};
  check_math_values(simd::fma(x, y1, y1), 10.0);
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_wrapper()
{
  using V = simd::vec<T, N>;
  const explicit_math_wrapper<V> x{V(T{4})};
  const explicit_math_wrapper<V> y{V(T{2})};
  static_assert(noexcept(simd::fma(x, y, y)));

  check_math_values(simd::fma(x, y, y), T{10});
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_non_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec x(positive_math_values<T>{});
  Vec y(T{2});
  Vec z(T{0.25});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::fma(x, y, z)), Vec>);
  static_assert(noexcept(cuda::std::simd::fma(x, y, z)));

  Vec fma_result = cuda::std::simd::fma(x, y, z);
  for (int i = 0; i < N; ++i)
  {
    assert(fma_result[i] == cuda::std::fma(x[i], y[i], z[i]));
  }
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_scalar()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec x(positive_math_values<T>{});
  Vec y(T{2});
  Vec z(T{0.25});
  T scalar_y{2};
  T scalar_z{0.25};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(scalar_y, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, scalar_y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, y, scalar_z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(scalar_y, scalar_z, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(scalar_y, y, scalar_z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, scalar_y, scalar_z)), Vec>);

  static_assert(noexcept(cuda::std::simd::fma(scalar_y, y, z)));
  static_assert(noexcept(cuda::std::simd::fma(x, scalar_y, z)));
  static_assert(noexcept(cuda::std::simd::fma(x, y, scalar_z)));
  static_assert(noexcept(cuda::std::simd::fma(scalar_y, scalar_z, z)));
  static_assert(noexcept(cuda::std::simd::fma(scalar_y, y, scalar_z)));
  static_assert(noexcept(cuda::std::simd::fma(x, scalar_y, scalar_z)));

  Vec fma_mixed1 = cuda::std::simd::fma(scalar_y, y, z);
  Vec fma_mixed2 = cuda::std::simd::fma(x, scalar_y, z);
  Vec fma_mixed3 = cuda::std::simd::fma(x, y, scalar_z);
  Vec fma_mixed4 = cuda::std::simd::fma(scalar_y, scalar_z, z);
  Vec fma_mixed5 = cuda::std::simd::fma(scalar_y, y, scalar_z);
  Vec fma_mixed6 = cuda::std::simd::fma(x, scalar_y, scalar_z);
  for (int i = 0; i < N; ++i)
  {
    assert(fma_mixed1[i] == cuda::std::fma(scalar_y, y[i], z[i]));
    assert(fma_mixed2[i] == cuda::std::fma(x[i], scalar_y, z[i]));
    assert(fma_mixed3[i] == cuda::std::fma(x[i], y[i], scalar_z));
    assert(fma_mixed4[i] == cuda::std::fma(scalar_y, scalar_z, z[i]));
    assert(fma_mixed5[i] == cuda::std::fma(scalar_y, y[i], scalar_z));
    assert(fma_mixed6[i] == cuda::std::fma(x[i], scalar_y, scalar_z));
  }
}

template <typename T, int N>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_mixed_arguments<N>();
  test_wrapper<T, N>();
  test_non_scalar<T, N>();
  test_scalar<T, N>();
}

DEFINE_SIMD_MATH_FLOATING_TEST()
DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
