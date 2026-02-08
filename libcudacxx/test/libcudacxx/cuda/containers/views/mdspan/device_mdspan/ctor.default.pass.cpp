//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr mdspan();
// Constraints:
//   - rank_dynamic() > 0 is true.
//   - is_default_constructible_v<data_handle_type> is true.
//   - is_default_constructible_v<mapping_type> is true.
//   - is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, map_.required_span_size()) is an accessible range of ptr_
//               and acc_ for the values of map_ and acc_ after the invocation of this constructor.
//
// Effects: Value-initializes ptr_, map_, and acc_.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <bool hc,
          bool mc,
          bool ac,
          class H,
          class M,
          class A,
          cuda::std::enable_if_t<(M::extents_type::rank_dynamic() > 0) && hc && mc && ac, int> = 0>
__host__ __device__ constexpr void test_mdspan_types(const H&, const M&, const A&)
{
  using MDS = cuda::device_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(hc == cuda::std::is_default_constructible<H>::value, "");
  static_assert(mc == cuda::std::is_default_constructible<M>::value, "");
  static_assert(ac == cuda::std::is_default_constructible<A>::value, "");

  MDS m;
#if !TEST_COMPILER(GCC)
  static_assert(noexcept(MDS()) == (noexcept(H()) && noexcept(M()) && noexcept(A())), "");
#endif // !TEST_COMPILER(GCC)
  assert(m.extents() == typename MDS::extents_type());
  test_equality_handle(m, H{});
  test_equality_mapping(m, M{});
  test_equality_accessor(m, A{});
}
template <bool hc,
          bool mc,
          bool ac,
          class H,
          class M,
          class A,
          cuda::std::enable_if_t<!((M::extents_type::rank_dynamic() > 0) && hc && mc && ac), int> = 0>
__host__ __device__ constexpr void test_mdspan_types(const H&, const M&, const A&)
{
  using MDS = cuda::device_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(hc == cuda::std::is_default_constructible<H>::value, "");
  static_assert(mc == cuda::std::is_default_constructible<M>::value, "");
  static_assert(ac == cuda::std::is_default_constructible<A>::value, "");
  static_assert(!cuda::std::is_default_constructible<MDS>::value, "");
}

template <bool hc, bool mc, bool ac, class H, class L, class A>
__host__ __device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types<hc, mc, ac>(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types<hc, mc, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_types<hc, mc, ac>(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types<hc, mc, ac>(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types<hc, mc, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types<hc, mc, ac>(
    handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <bool hc, bool ac, class H, class A>
__host__ __device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents<hc, true, ac>(handle, cuda::std::layout_left(), acc);
  mixin_extents<hc, true, ac>(handle, cuda::std::layout_right(), acc);

  // Use weird layout, make sure it has the properties we want to test
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  static_assert(!cuda::std::is_default_constructible<
                  typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<char, D>>>::value,
                "");
  mixin_extents<hc, false, ac>(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout<true, true>(elements.data(), cuda::std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is not default constructible except for const double, where it is not noexcept
  static_assert(
    cuda::std::is_default_constructible<checked_accessor<T>>::value == cuda::std::is_same<T, const double>::value, "");
  // checked_accessor's data handle type is not default constructible for double
  static_assert(cuda::std::is_default_constructible<typename checked_accessor<T>::data_handle_type>::value
                  != cuda::std::is_same<T, double>::value,
                "");
  mixin_layout<!cuda::std::is_same<T, double>::value, cuda::std::is_same<T, const double>::value>(
    typename checked_accessor<T>::data_handle_type(elements.data()), checked_accessor<T>(1024));
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX20 void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout<true, true>(elements.get_ptr(), cuda::std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is not default constructible except for const double, where it is not noexcept
  static_assert(
    cuda::std::is_default_constructible<checked_accessor<T>>::value == cuda::std::is_same<T, const double>::value, "");
  // checked_accessor's data handle type is not default constructible for double
  static_assert(cuda::std::is_default_constructible<typename checked_accessor<T>::data_handle_type>::value
                  != cuda::std::is_same<T, double>::value,
                "");
  mixin_layout<!cuda::std::is_same<T, double>::value, cuda::std::is_same<T, const double>::value>(
    typename checked_accessor<T>::data_handle_type(elements.get_ptr()), checked_accessor<T>(1024));
}

__host__ __device__ constexpr bool test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
  test_evil();

#if TEST_STD_VER >= 2020
  static_assert(test(), "");
  static_assert(test_evil(), "");
#endif // TEST_STD_VER >= 2020
  return 0;
}
