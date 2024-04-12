//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc && c++14, msvc && c++17

// TODO: Diagnose static assertion failures in NVRTC
// XFAIL: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "../mdspan.extents.util/extents_util.hpp"
#include "../my_int.hpp"

// TYPED_TEST(TestExtents, array_ctor)
template <class T>
__host__ __device__ void test_array_con()
{
  using TestFixture = TestExtents<T>;
  TestFixture t;

  auto e = typename TestFixture::extents_type(t.dyn_sizes);
  assert(e == t.exts);
}

template <class T, class IndexType, size_t N, class = void>
struct is_array_cons_avail : cuda::std::false_type
{};

template <class T, class IndexType, size_t N>
struct is_array_cons_avail<
  T,
  IndexType,
  N,
  cuda::std::enable_if_t<cuda::std::is_same<decltype(T{cuda::std::declval<cuda::std::array<IndexType, N>>()}), T>::value>>
    : cuda::std::true_type
{};

template <class T, class IndexType, size_t N>
constexpr bool is_array_cons_avail_v = is_array_cons_avail<T, IndexType, N>::value;

int main(int, char**)
{
  test_array_con<cuda::std::tuple_element_t<0, extents_test_types>>();
  test_array_con<cuda::std::tuple_element_t<1, extents_test_types>>();
  test_array_con<cuda::std::tuple_element_t<2, extents_test_types>>();
  test_array_con<cuda::std::tuple_element_t<3, extents_test_types>>();
  test_array_con<cuda::std::tuple_element_t<4, extents_test_types>>();
  test_array_con<cuda::std::tuple_element_t<5, extents_test_types>>();

  static_assert(is_array_cons_avail_v<cuda::std::dextents<int, 2>, int, 2> == true, "");

  static_assert(is_array_cons_avail_v<cuda::std::dextents<int, 2>, my_int, 2> == true, "");

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  // Constraint: rank consistency
  static_assert(is_array_cons_avail_v<cuda::std::dextents<int, 1>, int, 2> == false, "");

  // Constraint: convertibility
  static_assert(is_array_cons_avail_v<cuda::std::dextents<my_int, 1>, my_int_non_convertible, 1> == false, "");

  // Constraint: nonthrow-constructibility
#  ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  static_assert(is_array_cons_avail_v<cuda::std::dextents<int, 1>, my_int_non_nothrow_constructible, 1> == false, "");
#  endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
#endif // !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
