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

// [simd.ctor] complex constructor, [simd.complex.access] complex accessors: real(), imag()

#include <cuda/__complex_>
#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <complex>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "../simd_test_utils.h"
#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// complex constructor

template <typename T, typename Complex, int N>
TEST_FUNC constexpr void test_complex_ctor()
{
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<typename ComplexVec::mask_type, simd::mask<Complex, N>>);

  RealVec reals(offset_generator<T, 1>{});
  RealVec imags(offset_generator<T, 10>{});

  static_assert(noexcept(ComplexVec(reals, imags)));
  static_assert(noexcept(ComplexVec(reals)));

  ComplexVec vec(reals, imags);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i].real() == static_cast<T>(i + 1));
    assert(vec[i].imag() == static_cast<T>(i + 10));
  }

  ComplexVec vec_real_only(reals);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_real_only[i].real() == static_cast<T>(i + 1));
    assert(vec_real_only[i].imag() == T(0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// real() / imag() getters

template <typename T, typename Complex, int N>
TEST_FUNC constexpr void test_getters()
{
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 0, 10>{});

  auto reals = vec.real();
  auto imags = vec.imag();

  static_assert(cuda::std::is_same_v<decltype(reals), simd::basic_vec<T, simd::fixed_size<N>>>);
  static_assert(cuda::std::is_same_v<decltype(imags), simd::basic_vec<T, simd::fixed_size<N>>>);
  static_assert(noexcept(vec.real()));
  static_assert(noexcept(vec.imag()));

  for (int i = 0; i < N; ++i)
  {
    assert(reals[i] == static_cast<T>(i));
    assert(imags[i] == static_cast<T>(i + 10));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// real(vec) / imag(vec) setters

template <typename T, typename Complex, int N>
TEST_FUNC constexpr void test_setters()
{
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(1), T(2)));

  static_assert(noexcept(vec.real(cuda::std::declval<const RealVec&>())));
  static_assert(noexcept(vec.imag(cuda::std::declval<const RealVec&>())));

  RealVec new_reals(offset_generator<T, 100>{});
  vec.real(new_reals);

  for (int i = 0; i < N; ++i)
  {
    assert(vec[i].real() == static_cast<T>(i + 100));
    assert(vec[i].imag() == T(2));
  }

  RealVec new_imags(offset_generator<T, 200>{});
  vec.imag(new_imags);

  for (int i = 0; i < N; ++i)
  {
    assert(vec[i].real() == static_cast<T>(i + 100));
    assert(vec[i].imag() == static_cast<T>(i + 200));
  }
}

//----------------------------------------------------------------------------------------------------------------------

// cuda::complex<T> does not support extended fp types
template <typename T, int N>
TEST_FUNC constexpr void test_cuda_complex_type()
{
  using CudaComplex = ::cuda::complex<T>;
  test_complex_ctor<T, CudaComplex, N>();
  test_getters<T, CudaComplex, N>();
  test_setters<T, CudaComplex, N>();
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  using CudaStdComplex = cuda::std::complex<T>;
  test_complex_ctor<T, CudaStdComplex, N>();
  test_getters<T, CudaStdComplex, N>();
  test_setters<T, CudaStdComplex, N>();

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST,
               (test_complex_ctor<T, ::std::complex<T>, N>(); test_getters<T, ::std::complex<T>, N>();
                test_setters<T, ::std::complex<T>, N>();))
#endif // _CCCL_HAS_HOST_STD_LIB()
}

TEST_FUNC constexpr bool test()
{
  test_type<float, 1>();
  test_type<float, 4>();
  test_cuda_complex_type<float, 1>();
  test_cuda_complex_type<float, 4>();
#if _CCCL_HAS_INT128()
  test_type<double, 1>();
  test_type<double, 4>();
  test_cuda_complex_type<double, 1>();
  test_cuda_complex_type<double, 4>();
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
  static_assert(test());
  assert(test_runtime());
  return 0;
}
