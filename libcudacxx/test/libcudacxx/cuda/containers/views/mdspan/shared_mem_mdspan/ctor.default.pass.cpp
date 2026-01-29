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
__device__ constexpr void test_mdspan_types(const H&, const M&, const A&)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(hc == cuda::std::is_default_constructible_v<H>);
  static_assert(mc == cuda::std::is_default_constructible_v<M>);
  static_assert(ac == cuda::std::is_default_constructible_v<A>);

  MDS m;
#if !TEST_COMPILER(GCC)
  static_assert(noexcept(MDS()) == (noexcept(H()) && noexcept(M()) && noexcept(A())));
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
__device__ constexpr void test_mdspan_types(const H&, const M&, const A&)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(hc == cuda::std::is_default_constructible_v<H>);
  static_assert(mc == cuda::std::is_default_constructible_v<M>);
  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  static_assert(!cuda::std::is_default_constructible_v<MDS>);
}

template <bool hc, bool mc, bool ac, class H, class L, class A>
__device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
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
__device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents<hc, true, ac>(handle, cuda::std::layout_left(), acc);
  mixin_extents<hc, true, ac>(handle, cuda::std::layout_right(), acc);

  // Use weird layout, make sure it has the properties we want to test
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  static_assert(!cuda::std::is_default_constructible_v<
                typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<char, D>>>);
  mixin_extents<hc, false, ac>(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout<true, true>(elements.data(), cuda::std::default_accessor<T>());
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout<true, true>(elements.get_ptr(), cuda::std::default_accessor<T>());
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
