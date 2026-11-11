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

// [simd.overview], [simd.mask.overview], disabled specializations

#include <cuda/std/__simd_>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, typename = void>
inline constexpr bool has_type_field = false;

template <typename T>
inline constexpr bool has_type_field<T, cuda::std::void_t<typename T::type>> = true;

template <typename T, typename = void>
inline constexpr bool has_size_field = false;

template <typename T>
inline constexpr bool has_size_field<T, cuda::std::void_t<decltype(T::size)>> = true;

struct unknown_abi
{};

struct unrelated_type
{};

struct incomplete_type;

template <typename T, typename Abi = simd::fixed_size<4>>
TEST_HOST_DEVICE_FUNC constexpr void test_disabled_vec()
{
  using V = simd::basic_vec<T, Abi>;

  static_assert(!has_size_field<V>);
  static_assert(cuda::std::is_same_v<typename V::value_type, T>);
  static_assert(cuda::std::is_same_v<typename V::abi_type, Abi>);

  static_cast<void>(sizeof(typename V::mask_type)); // mask_type is complete type
  static_assert(simd::__simd_size_v<T, Abi> == 0);
}

template <size_t Bytes, typename Abi = simd::native<unsigned char>>
TEST_HOST_DEVICE_FUNC constexpr void test_disabled_mask()
{
  using M = simd::basic_mask<Bytes, Abi>;

  static_assert(!has_size_field<M>);
  static_assert(cuda::std::is_same_v<typename M::value_type, bool>);
  static_assert(cuda::std::is_same_v<typename M::abi_type, Abi>);
}

TEST_HOST_DEVICE_FUNC constexpr bool test_disabled_types()
{
  test_disabled_vec<bool>();
  test_disabled_vec<const int>();
  test_disabled_vec<volatile int>();
  test_disabled_vec<void>();
  test_disabled_vec<int&>();
  test_disabled_vec<void>();
  test_disabled_vec<int[2]>();
  test_disabled_vec<cuda::std::complex<int>>();

  test_disabled_vec<unrelated_type>();
  test_disabled_vec<incomplete_type>();
  test_disabled_vec<int, unknown_abi>();

  test_disabled_vec<int, simd::fixed_size<0>>();
  test_disabled_vec<int, simd::fixed_size<-1>>();
  test_disabled_vec<int, simd::fixed_size<65>>();

  // mask only supports 1, 2, 4, 8, (16)
  test_disabled_mask<0>();
  test_disabled_mask<3>();
  test_disabled_mask<32>();
  test_disabled_mask<4, unknown_abi>();
  test_disabled_mask<4, simd::fixed_size<0>>();
  test_disabled_mask<4, simd::fixed_size<-1>>();
  test_disabled_mask<4, simd::fixed_size<65>>();
  return true;
}

int main(int, char**)
{
  static_assert(test_disabled_types());
  return 0;
}
