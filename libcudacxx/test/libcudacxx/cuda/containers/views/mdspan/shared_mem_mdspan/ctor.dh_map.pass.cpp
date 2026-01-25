//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr mdspan(data_handle_type p, const mapping_type& m);
//
// Constraints: is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, m.required_span_size()) is an accessible range of p and acc_
//                for the value of acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with cuda::std::move(p),
//   - direct-non-list-initializes map_ with m, and
//   - value-initializes acc_.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <bool ac, class H, class M, class A, cuda::std::enable_if_t<ac, int> = 0>
__device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A&)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    move_counted_handle<typename MDS::element_type>::move_counter() = 0;
  }
  // use formulation of constructor which tests that it is not explicit
  MDS m = {handle, map};
  test_move_counter<MDS, H>();

  static_assert(!noexcept(MDS(handle, map)));
  assert(m.extents() == map.extents());
  test_equality_handle(m, handle);
  test_equality_mapping(m, map);
  test_equality_accessor(m, A{});
}
template <bool ac, class H, class M, class A, cuda::std::enable_if_t<!ac, int> = 0>
__device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A&)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  static_assert(!cuda::std::is_constructible_v<MDS, const H&, const M&>);
}

template <bool ac, class H, class L, class A>
__device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types<ac>(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types<ac>(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_types<ac>(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types<ac>(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types<ac>(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types<ac>(
    handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <bool ac, class H, class A>
__device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents<ac>(handle, cuda::std::layout_left(), acc);
  mixin_extents<ac>(handle, cuda::std::layout_right(), acc);
  mixin_extents<ac>(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout<true>(elements.data(), cuda::std::default_accessor<T>());
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout<true>(elements.get_ptr(), cuda::std::default_accessor<T>());
}

template <class E>
using mapping_t = typename cuda::std::layout_right::template mapping<E>;

__device__ void test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();

  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  using mds_t                         = cuda::shared_memory_mdspan<float, cuda::std::extents<int, 3, D, D>>;

  // sanity check
  static_assert(cuda::std::is_constructible_v<mds_t, float*, mapping_t<cuda::std::extents<int, 3, D, D>>>);

  // test non-constructibility from wrong mapping type
  // wrong rank
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, mapping_t<cuda::std::extents<int, D, D>>>);
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, mapping_t<cuda::std::extents<int, D, D, D, D>>>);
  // wrong type in general: note the map constructor does NOT convert, since it takes by const&
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, mapping_t<cuda::std::extents<int, D, D, D>>>);
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, mapping_t<cuda::std::extents<unsigned, 3, D, D>>>);

  // test non-constructibility from wrong handle_type
  static_assert(!cuda::std::is_constructible_v<mds_t, const float*, mapping_t<cuda::std::extents<int, 3, D, D>>>);
}

__device__ void test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test(); test_evil();))
  return 0;
}
