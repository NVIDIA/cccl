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

// [simd.traits], member availability and disabled specializations

#include <cuda/__complex_>
#include <cuda/std/__simd_>
#include <cuda/std/complex>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/type_traits>

#include <nv/target>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <complex>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, typename = void>
inline constexpr bool has_type_field = false;

template <typename T>
inline constexpr bool has_type_field<T, cuda::std::void_t<typename T::type>> = true;

template <typename T, typename = void>
inline constexpr bool has_value_field = false;

template <typename T>
inline constexpr bool has_value_field<T, cuda::std::void_t<decltype(T::value)>> = true;

template <typename T, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_enabled()
{
  using V = simd::vec<T, N>;
  using M = simd::mask<T, N>;

  static_assert(V::size() == N);
  static_assert(M::size() == N);
  static_assert(cuda::is_trivially_copyable_v<V>);
  static_assert(cuda::is_trivially_copyable_v<M>);
  static_assert(has_value_field<simd::alignment<V>>);
  static_assert(simd::__simd_size_v<T, simd::fixed_size<N>> == N);

  // invalid resize and rebind
  static_assert(!has_type_field<simd::resize<0, V>>);
  static_assert(!has_type_field<simd::resize<-1, V>>);
  static_assert(!has_type_field<simd::resize<65, V>>);
  static_assert(!has_type_field<simd::resize<0, M>>);
  static_assert(!has_type_field<simd::resize<-1, M>>);
  static_assert(!has_type_field<simd::resize<65, M>>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  test_enabled<T, 1>();
  test_enabled<T, 3>();
}

TEST_HOST_DEVICE_FUNC constexpr void test_types()
{
  test_type<int>();
  test_type<unsigned>();
  test_type<char>();
  test_type<char16_t>();
  test_type<char32_t>();
  test_type<wchar_t>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_type<float>();
  test_type<double>();
  test_type<cuda::std::complex<float>>();
  test_type<cuda::complex<float>>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<cuda::std::complex<double>>();
  test_type<cuda::complex<double>>();
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test_type<::std::complex<float>>();))
#  if _CCCL_HAS_INT128()
  NV_IF_TARGET(NV_IS_HOST, (test_type<::std::complex<double>>();))
#  endif // _CCCL_HAS_INT128()
#endif // _CCCL_HAS_HOST_STD_LIB()
#if _CCCL_HAS_NVFP16()
  test_type<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
}

TEST_HOST_DEVICE_FUNC constexpr void test_other_traits()
{
  using V = simd::vec<int, 4>;

  using RefV = simd::basic_vec<int&, simd::fixed_size<4>>;
  static_assert(cuda::std::is_same_v<typename RefV::mask_type, simd::mask<int, 4>>);

  // test traits for invalid/disabled mask cases
  static_assert(simd::__mask_size_v<3, simd::fixed_size<4>> == 0);
  static_assert(simd::__mask_size_v<4, simd::fixed_size<0>> == 0);

  // test traits for invalid/disabled vector complex cases
  static_assert(!simd::__is_vectorizable_v<cuda::std::complex<const float>>);
  static_assert(!simd::__is_vectorizable_v<cuda::std::complex<volatile float>>);
#if _CCCL_HAS_HOST_STD_LIB()
  static_assert(simd::__is_vectorizable_v<::std::complex<float>>);
  static_assert(!simd::__is_vectorizable_v<::std::complex<int>>);
#endif // _CCCL_HAS_HOST_STD_LIB()

  static_assert(!has_value_field<simd::alignment<int, int>>);
  static_assert(!has_value_field<simd::alignment<simd::mask<int, 4>, int>>);
  static_assert(has_value_field<simd::alignment<simd::vec<int, 0>, int>>);

  static_assert(!has_type_field<simd::rebind<int, int>>);
  static_assert(!has_type_field<simd::rebind<int, const V>>);
  static_assert(!has_type_field<simd::resize<4, int>>);
  static_assert(!has_type_field<simd::resize<4, const V>>);
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_types();
  test_other_traits();
  return true;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
