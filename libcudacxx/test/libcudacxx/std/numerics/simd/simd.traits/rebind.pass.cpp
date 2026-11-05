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

// template<class T, class V> struct rebind;
// template<class T, class V> using rebind_t = typename rebind<T, V>::type;

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
// rebind with basic_vec

template <typename NewT, typename OldT, int N>
TEST_HOST_DEVICE_FUNC void test_rebind_vec()
{
  using OldVec = simd::basic_vec<OldT, simd::fixed_size<N>>;
  using Result = simd::rebind_t<NewT, OldVec>;
  static_assert(cuda::std::is_same_v<Result, simd::vec<NewT, N>>);
  static_assert(cuda::std::is_same_v<typename Result::value_type, NewT>);
  static_assert(Result::size() == N);
}

template <typename NewT, typename OldT>
TEST_HOST_DEVICE_FUNC void test_rebind_vec_sizes()
{
  test_rebind_vec<NewT, OldT, 1>();
  test_rebind_vec<NewT, OldT, 2>();
  test_rebind_vec<NewT, OldT, 3>();
  test_rebind_vec<NewT, OldT, 4>();
  test_rebind_vec<NewT, OldT, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// rebind with basic_mask

template <typename NewT, typename OldT, int N>
TEST_HOST_DEVICE_FUNC void test_rebind_mask()
{
  using OldMask = simd::mask<OldT, N>;
  using Result  = simd::rebind_t<NewT, OldMask>;
  static_assert(cuda::std::is_same_v<Result, simd::mask<NewT, N>>);
  static_assert(Result::size() == N);
}

template <typename NewT, typename OldT>
TEST_HOST_DEVICE_FUNC void test_rebind_mask_sizes()
{
  test_rebind_mask<NewT, OldT, 1>();
  test_rebind_mask<NewT, OldT, 2>();
  test_rebind_mask<NewT, OldT, 3>();
  test_rebind_mask<NewT, OldT, 4>();
  test_rebind_mask<NewT, OldT, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// rebind_t matches rebind::type

template <typename NewT, typename V>
TEST_HOST_DEVICE_FUNC void test_rebind_t_alias()
{
  // Check the nested type directly to verify the alias definition.
  // NOLINTNEXTLINE(cccl-modernize-type-traits)
  static_assert(cuda::std::is_same_v<simd::rebind_t<NewT, V>, typename simd::rebind<NewT, V>::type>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_rebind_vec_sizes<float, T>();
  test_rebind_mask_sizes<float, T>();
  test_rebind_vec_sizes<T, int>();
  test_rebind_mask_sizes<T, int>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  // rebind basic_vec and basic_mask
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

  // Additional source/destination pairs.
  test_rebind_vec_sizes<int, float>();
  test_rebind_vec_sizes<int, char>();
  test_rebind_vec_sizes<int, double>();
  test_rebind_vec_sizes<short, long long>();
  test_rebind_mask_sizes<short, long long>();

  // rebind_t alias matches rebind::type
  test_rebind_t_alias<float, simd::vec<int, 4>>();
  test_rebind_t_alias<int, simd::vec<double, 2>>();
  test_rebind_t_alias<double, simd::mask<int, 4>>();
}

int main(int, char**)
{
  test();
  return 0;
}
