//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA Complex++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

// <cuda/std/__simd_>

// [simd.complex.math] trigonometric and hyperbolic functions:
// sin, asin, cos, acos, tan, atan, sinh, asinh, cosh, acosh, tanh, atanh

// The execution gets stuck in nvcc 12.0 with msvc.
// UNSUPPORTED: msvc && nvcc-12.0

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// sin, cos, tan, asin, acos, atan

template <typename T, int N>
TEST_FUNC void test_trig()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_diverse_generator<T>{});

  static_assert(cuda::std::is_same_v<decltype(simd::sin(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::cos(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::tan(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::asin(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::acos(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::atan(vec)), ComplexVec>);
  static_assert(!noexcept(simd::sin(vec)));
  static_assert(!noexcept(simd::cos(vec)));
  static_assert(!noexcept(simd::tan(vec)));
  static_assert(!noexcept(simd::asin(vec)));
  static_assert(!noexcept(simd::acos(vec)));
  static_assert(!noexcept(simd::atan(vec)));
  ComplexVec vec_sin  = simd::sin(vec);
  ComplexVec vec_cos  = simd::cos(vec);
  ComplexVec vec_tan  = simd::tan(vec);
  ComplexVec vec_asin = simd::asin(vec);
  ComplexVec vec_acos = simd::acos(vec);
  ComplexVec vec_atan = simd::atan(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_sin[i], cuda::std::sin(vec[i]));
    is_fp_close(vec_cos[i], cuda::std::cos(vec[i]));
    is_fp_close(vec_tan[i], cuda::std::tan(vec[i]));
    is_fp_close(vec_asin[i], cuda::std::asin(vec[i]));
    is_fp_close(vec_acos[i], cuda::std::acos(vec[i]));
    is_fp_close(vec_atan[i], cuda::std::atan(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sinh, cosh, tanh, asinh, acosh, atanh

template <typename T, int N>
TEST_FUNC void test_hyperbolic()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_diverse_generator<T>{});

  static_assert(cuda::std::is_same_v<decltype(simd::sinh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::cosh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::tanh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::asinh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::acosh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::atanh(vec)), ComplexVec>);
  static_assert(!noexcept(simd::sinh(vec)));
  static_assert(!noexcept(simd::cosh(vec)));
  static_assert(!noexcept(simd::tanh(vec)));
  static_assert(!noexcept(simd::asinh(vec)));
  static_assert(!noexcept(simd::acosh(vec)));
  static_assert(!noexcept(simd::atanh(vec)));

  ComplexVec vec_sinh                   = simd::sinh(vec);
  ComplexVec vec_cosh                   = simd::cosh(vec);
  ComplexVec vec_tanh                   = simd::tanh(vec);
  ComplexVec vec_asinh                  = simd::asinh(vec);
  ComplexVec vec_acosh                  = simd::acosh(vec);
  [[maybe_unused]] ComplexVec vec_atanh = simd::atanh(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_sinh[i], cuda::std::sinh(vec[i]));
    is_fp_close(vec_cosh[i], cuda::std::cosh(vec[i]));
    is_fp_close(vec_tanh[i], cuda::std::tanh(vec[i]));
    is_fp_close(vec_asinh[i], cuda::std::asinh(vec[i]));
    is_fp_close(vec_acosh[i], cuda::std::acosh(vec[i]));
    // cicc seg faults on atanh with clang 14 and nvcc 12.0
#if !_CCCL_COMPILER(CLANG, <=, 14) && !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0)
    is_fp_close(vec_atanh[i], cuda::std::atanh(vec[i]));
#endif
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC void test_type()
{
  test_trig<T, N>();
  test_hyperbolic<T, N>();
}

TEST_FUNC bool test()
{
  test_type<float, 1>();
  test_type<float, 4>();
#if _CCCL_HAS_INT128()
  test_type<double, 1>();
  test_type<double, 4>();
#endif // _CCCL_HAS_INT128()
  return true;
}

TEST_FUNC bool test_runtime()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half, 1>();
  test_type<__half, 4>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16, 1>();
  test_type<__nv_bfloat16, 4>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  return true;
}

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
