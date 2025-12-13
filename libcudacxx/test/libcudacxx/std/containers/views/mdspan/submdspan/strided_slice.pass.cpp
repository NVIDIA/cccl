//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <size_t Val>
struct integral_like : cuda::std::integral_constant<size_t, Val>
{};
static_assert(cuda::std::__integral_constant_like<integral_like<42>>, "");

template <size_t Val>
struct not_integral_like : cuda::std::integral_constant<size_t, Val>
{
  __host__ __device__ constexpr not_integral_like(int) noexcept {}
};
static_assert(!cuda::std::__integral_constant_like<not_integral_like<42>>, "");

template <class OffsetType, class ExtentType, class StrideType>
_CCCL_CONCEPT can_strided_slice = _CCCL_REQUIRES_EXPR((OffsetType, ExtentType, StrideType))(
  (cuda::std::strided_slice<OffsetType, ExtentType, StrideType>{}));

static_assert(can_strided_slice<int, short, size_t>, "");
static_assert(can_strided_slice<integral_like<42>, int, int>, "");
static_assert(can_strided_slice<int, integral_like<42>, int>, "");
static_assert(can_strided_slice<int, int, integral_like<42>>, "");

// We cannot check mandates with the current setup
// static_assert(!can_strided_slice<int, void, integral_like<42>>, "");
// static_assert(!can_strided_slice<not_integral_like<42>, int, int>, "");
// static_assert(!can_strided_slice<int, not_integral_like<42>, int>, "");
// static_assert(!can_strided_slice<int, int, not_integral_like<42>>, "");

template <class T>
__host__ __device__ constexpr T construct_from_int([[maybe_unused]] int val) noexcept
{
  if constexpr (cuda::std::__integral_constant_like<T>)
  {
    return T{};
  }
  else
  {
    return static_cast<T>(val);
  }
}

template <class OffsetType, class ExtentType, class StrideType>
__host__ __device__ constexpr void test()
{
  using strided_slice = cuda::std::strided_slice<OffsetType, ExtentType, StrideType>;
  // Ensure we are trivially copy/move constructible
  static_assert(cuda::std::is_trivially_copy_constructible<strided_slice>::value, "");
  static_assert(cuda::std::is_trivially_move_constructible<strided_slice>::value, "");

  // Ensure we properly do not store compile time sizes
#if _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() && !TEST_COMPILER(MSVC)
  static_assert(sizeof(strided_slice) == sizeof(cuda::std::tuple<OffsetType, ExtentType, StrideType>), "");
#endif // _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() && !TEST_COMPILER(MSVC)

  // Ensure we have the right alias types
  static_assert(cuda::std::is_same<typename strided_slice::offset_type, OffsetType>::value, "");
  static_assert(cuda::std::is_same<typename strided_slice::extent_type, ExtentType>::value, "");
  static_assert(cuda::std::is_same<typename strided_slice::stride_type, StrideType>::value, "");

  // Ensure we have the right members with the right types
  static_assert(cuda::std::is_same<decltype(strided_slice{}.offset), OffsetType>::value, "");
  static_assert(cuda::std::is_same<decltype(strided_slice{}.extent), ExtentType>::value, "");
  static_assert(cuda::std::is_same<decltype(strided_slice{}.stride), StrideType>::value, "");

  {
    strided_slice zero_initialized;
    assert(zero_initialized.offset == (cuda::std::__integral_constant_like<OffsetType> ? 42 : 0));
    assert(zero_initialized.extent == (cuda::std::__integral_constant_like<ExtentType> ? 42 : 0));
    assert(zero_initialized.stride == (cuda::std::__integral_constant_like<StrideType> ? 42 : 0));
  }
  {
    strided_slice value_initialized{};
    assert(value_initialized.offset == (cuda::std::__integral_constant_like<OffsetType> ? 42 : 0));
    assert(value_initialized.extent == (cuda::std::__integral_constant_like<ExtentType> ? 42 : 0));
    assert(value_initialized.stride == (cuda::std::__integral_constant_like<StrideType> ? 42 : 0));
  }
  { // The const here is load bearing because clang-cuda does not properly initialize otherwise
    const strided_slice list_initialized = {
      construct_from_int<OffsetType>(1), construct_from_int<ExtentType>(2), construct_from_int<StrideType>(3)};
    assert(list_initialized.offset == (cuda::std::__integral_constant_like<OffsetType> ? 42 : 1));
    assert(list_initialized.extent == (cuda::std::__integral_constant_like<ExtentType> ? 42 : 2));
    assert(list_initialized.stride == (cuda::std::__integral_constant_like<StrideType> ? 42 : 3));
  }
#if TEST_STD_VER >= 2020
  { // The const here is load bearing because clang-cuda does not properly initialize otherwise
    const strided_slice aggregate_initialized = {
      .offset = construct_from_int<OffsetType>(1),
      .extent = construct_from_int<ExtentType>(2),
      .stride = construct_from_int<StrideType>(3)};
    assert(aggregate_initialized.offset == (cuda::std::__integral_constant_like<OffsetType> ? 42 : 1));
    assert(aggregate_initialized.extent == (cuda::std::__integral_constant_like<ExtentType> ? 42 : 2));
    assert(aggregate_initialized.stride == (cuda::std::__integral_constant_like<StrideType> ? 42 : 3));
  }
#endif // TEST_STD_VER >= 2020
}

__host__ __device__ constexpr bool test()
{
  test<int, short, size_t>();
  test<integral_like<42>, int, int>();
  test<int, integral_like<42>, int>();
  test<int, int, integral_like<42>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
