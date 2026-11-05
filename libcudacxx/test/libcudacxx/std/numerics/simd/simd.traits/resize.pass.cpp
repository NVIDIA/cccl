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

// template<simd-size-type N, class V> struct resize;
// template<simd-size-type N, class V> using resize_t = typename resize<N, V>::type;

#include <cuda/__complex_>
#include <cuda/std/__simd_>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include <nv/target>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <complex>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// resize with basic_vec

template <typename T, int OldN, int NewN>
TEST_HOST_DEVICE_FUNC void test_resize_vec()
{
  using OldVec   = simd::basic_vec<T, simd::fixed_size<OldN>>;
  using Result   = simd::resize_t<NewN, OldVec>;
  using Expected = simd::basic_vec<T, simd::fixed_size<NewN>>;
  static_assert(cuda::std::is_same_v<typename Result::value_type, T>);
  static_assert(Result::size() == NewN);
  static_assert(cuda::std::is_same_v<Result, Expected>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_resize_vec_all()
{
  test_resize_vec<T, 4, 1>();
  test_resize_vec<T, 4, 3>();
  test_resize_vec<T, 4, 4>();
  test_resize_vec<T, 4, 2>();
  test_resize_vec<T, 2, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// resize with basic_mask

template <typename T, int OldN, int NewN>
TEST_HOST_DEVICE_FUNC void test_resize_mask()
{
  using OldMask  = simd::mask<T, OldN>;
  using Result   = simd::resize_t<NewN, OldMask>;
  using Expected = simd::mask<T, NewN>;
  static_assert(Result::size() == NewN);
  static_assert(cuda::std::is_same_v<Result, Expected>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_resize_mask_all()
{
  test_resize_mask<T, 4, 1>();
  test_resize_mask<T, 4, 3>();
  test_resize_mask<T, 4, 4>();
  test_resize_mask<T, 4, 2>();
  test_resize_mask<T, 2, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// resize_t matches resize::type

template <int N, typename V>
TEST_HOST_DEVICE_FUNC void test_resize_t_alias()
{
  static_assert(cuda::std::is_same_v<simd::resize_t<N, V>, typename simd::resize<N, V>::type>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_resize_vec_all<T>();
  test_resize_mask_all<T>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  // resize basic_vec and basic_mask
  test_type<int>();
  test_type<unsigned>();
  test_type<char>();
  test_type<short>();
  test_type<long long>();
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

  // resize_t alias matches resize::type
  test_resize_t_alias<8, simd::vec<int, 4>>();
  test_resize_t_alias<2, simd::vec<float, 8>>();
}

int main(int, char**)
{
  test();
  return 0;
}
