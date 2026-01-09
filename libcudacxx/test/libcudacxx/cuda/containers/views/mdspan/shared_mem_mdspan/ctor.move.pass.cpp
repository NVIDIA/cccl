//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr mdspan(mdspan&&) = default;
//
// A specialization of mdspan is a trivially copyable type if its accessor_type, mapping_type, and data_handle_type are
// trivially copyable types.
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
__device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  MDS m_org(handle, map, acc);
  MDS m(cuda::std::move(m_org));
  static_assert(cuda::std::is_trivially_move_constructible_v<MDS>
                == (cuda::std::is_trivially_move_constructible_v<H> && cuda::std::is_trivially_move_constructible_v<M>
                    && cuda::std::is_trivially_move_constructible_v<A>) );
  assert(m.extents() == map.extents());
  test_equality_handle(m, handle);
  test_equality_mapping(m, map);
  test_equality_accessor(m, acc);
}

template <class H, class L, class A>
__device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
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
__device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  // make sure we test a trivially copyable mapping
  static_assert(cuda::std::is_trivially_move_constructible_v<
                typename cuda::std::layout_left::template mapping<cuda::std::extents<int>>>);
  mixin_extents(handle, cuda::std::layout_left(), acc);
  mixin_extents(handle, cuda::std::layout_right(), acc);
  // make sure we test a not trivially copyable mapping
  static_assert(!cuda::std::is_trivially_move_constructible_v<
                typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>>);
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  // make sure we test trivially constructible accessor and data_handle
  static_assert(cuda::std::is_trivially_move_constructible_v<cuda::std::default_accessor<T>>);
  static_assert(
    cuda::std::is_trivially_move_constructible_v<typename cuda::std::default_accessor<T>::data_handle_type>);
  mixin_layout(elements.data(), cuda::std::default_accessor<T>());
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  // make sure we test trivially constructible accessor and data_handle
  static_assert(cuda::std::is_trivially_move_constructible_v<cuda::std::default_accessor<T>>);
  static_assert(
    cuda::std::is_trivially_move_constructible_v<typename cuda::std::default_accessor<T>::data_handle_type>);
  mixin_layout(elements.get_ptr(), cuda::std::default_accessor<T>());
}

__device__ void test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
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
