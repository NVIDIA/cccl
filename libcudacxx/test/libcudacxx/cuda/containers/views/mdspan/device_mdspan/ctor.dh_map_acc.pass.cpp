//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr mdspan(data_handle_type p, const mapping_type& m, const accessor_type& a);
//
// Preconditions: [0, m.required_span_size()) is an accessible range of p and a.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with cuda::std::move(p),
//   - direct-non-list-initializes map_ with m, and
//   - direct-non-list-initializes acc_ with a.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <class H, class M, class A>
__host__ __device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc)
{
  using MDS = cuda::device_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    move_counted_handle<typename MDS::element_type>::move_counter() = 0;
  }
  // use formulation of constructor which tests that it is not explicit
  MDS m = {handle, map, acc};
  test_move_counter<MDS, H>();

  static_assert(!noexcept(MDS(handle, map, acc)), "");
  assert(m.extents() == map.extents());
  test_equality_handle(m, handle);
  test_equality_mapping(m, map);
  test_equality_accessor(m, acc);
}

template <class H, class L, class A>
__host__ __device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <class H, class A>
__host__ __device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents(handle, cuda::std::layout_left(), acc);
  mixin_extents(handle, cuda::std::layout_right(), acc);
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout(elements.data(), cuda::std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is not default constructible except for const double, where it is not noexcept
  static_assert(
    cuda::std::is_default_constructible<checked_accessor<T>>::value == cuda::std::is_same<T, const double>::value, "");
  // checked_accessor's data handle type is not default constructible for double
  static_assert(cuda::std::is_default_constructible<typename checked_accessor<T>::data_handle_type>::value
                  != cuda::std::is_same<T, double>::value,
                "");
  mixin_layout(typename checked_accessor<T>::data_handle_type(elements.data()), checked_accessor<T>(1024));
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX20 void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout(elements.get_ptr(), cuda::std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is not default constructible except for const double, where it is not noexcept
  static_assert(
    cuda::std::is_default_constructible<checked_accessor<T>>::value == cuda::std::is_same<T, const double>::value, "");
  // checked_accessor's data handle type is not default constructible for double
  static_assert(cuda::std::is_default_constructible<typename checked_accessor<T>::data_handle_type>::value
                  != cuda::std::is_same<T, double>::value,
                "");
  mixin_layout(typename checked_accessor<T>::data_handle_type(elements.get_ptr()), checked_accessor<T>(1024));
}

template <class E>
using mapping_t = typename cuda::std::layout_right::template mapping<E>;

__host__ __device__ constexpr bool test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();

  // test non-constructibility from wrong args
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  using mds_t                         = cuda::device_mdspan<float, cuda::std::extents<int, 3, D, D>>;
  using acc_t                         = cuda::std::default_accessor<float>;

  // sanity check
  static_assert(cuda::std::is_constructible<mds_t, float*, mapping_t<cuda::std::extents<int, 3, D, D>>, acc_t>::value,
                "");

  // test non-constructibility from wrong accessor
  static_assert(!cuda::std::is_constructible<mds_t,
                                             float*,
                                             mapping_t<cuda::std::extents<int, 3, D, D>>,
                                             cuda::std::default_accessor<const float>>::value,
                "");

  // test non-constructibility from wrong mapping type
  // wrong rank
  static_assert(!cuda::std::is_constructible<mds_t, float*, mapping_t<cuda::std::extents<int, D, D>>, acc_t>::value,
                "");
  static_assert(
    !cuda::std::is_constructible<mds_t, float*, mapping_t<cuda::std::extents<int, D, D, D, D>>, acc_t>::value, "");
  // wrong type in general: note the map constructor does NOT convert, since it takes by const&
  static_assert(!cuda::std::is_constructible<mds_t, float*, mapping_t<cuda::std::extents<int, D, D, D>>, acc_t>::value,
                "");
  static_assert(
    !cuda::std::is_constructible<mds_t, float*, mapping_t<cuda::std::extents<unsigned, 3, D, D>>, acc_t>::value, "");

  // test non-constructibility from wrong handle_type
  static_assert(
    !cuda::std::is_constructible<mds_t, const float*, mapping_t<cuda::std::extents<int, 3, D, D>>, acc_t>::value, "");

  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
  return true;
}

__host__ __device__ void test_host()
{
  test();
  test_evil();

#if TEST_STD_VER >= 2020
  static_assert(test(), "");
  static_assert(test_evil(), "");
#endif // TEST_STD_VER >= 2020
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test_host();)

  return 0;
}
