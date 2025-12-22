//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class... OtherIndexTypes>
//   constexpr explicit mdspan(data_handle_type p, OtherIndexTypes... exts);
//
// Let N be sizeof...(OtherIndexTypes).
//
// Constraints:
//   - (is_convertible_v<OtherIndexTypes, index_type> && ...) is true,
//   - (is_nothrow_constructible<index_type, OtherIndexTypes> && ...) is true,
//   - N == rank() || N == rank_dynamic() is true,
//   - is_constructible_v<mapping_type, extents_type> is true, and
//   - is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, map_.required_span_size()) is an accessible range of p and acc_
//                for the values of map_ and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with cuda::std::move(p),
//   - direct-non-list-initializes map_ with extents_type(static_cast<index_type>(cuda::std::move(exts))...), and
//   - value-initializes acc_.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

#if _CCCL_CUDA_COMPILER(NVRTC)
#  define _CCCL_NVRTC_CONCEPT_DECORATOR() __device__
#else
#  define _CCCL_NVRTC_CONCEPT_DECORATOR()
#endif // _CCCL_CUDA_COMPILER(NVRTC)

template <class MDS, class... Args>
_CCCL_CONCEPT check_mdspan_ctor_implicit2 =
  _CCCL_REQUIRES_EXPR((MDS, variadic Args), MDS m, Args... args)((m = {args...}));

template <class MDS>
__device__ void check_implicit_construction(MDS);

template <class MDS, class... Args>
_CCCL_NVRTC_CONCEPT_DECORATOR()
constexpr bool check_implicit_construction_impl(...)
{
  return false;
}
template <class MDS, class... Args>
_CCCL_NVRTC_CONCEPT_DECORATOR()
constexpr auto check_implicit_construction_impl(int)
  -> decltype(check_implicit_construction<MDS>({cuda::std::declval<Args>()...}), true)
{
  return true;
}

template <class MDS, class... Args>
_CCCL_CONCEPT check_mdspan_ctor_implicit = check_implicit_construction_impl<MDS, Args...>(0);

template <bool mec, bool ac, class H, class M, class A, class... Idxs, cuda::std::enable_if_t<mec && ac, int> = 0>
__device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A&, Idxs... idxs)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(mec == cuda::std::is_constructible_v<M, typename M::extents_type>);
  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    move_counted_handle<typename MDS::element_type>::move_counter() = 0;
  }
  MDS m(handle, idxs...);
  test_move_counter<MDS, H>();

  // sanity check that concept works
  static_assert(check_mdspan_ctor_implicit<MDS, H, cuda::std::array<typename MDS::index_type, MDS::rank_dynamic()>>);
  // check that the constructor from integral is explicit
  static_assert(!check_mdspan_ctor_implicit<MDS, H, Idxs...>);

  assert(m.extents() == map.extents());
  test_equality_handle(m, handle);
  test_equality_mapping(m, map);
  test_equality_accessor(m, A{});
}
template <bool mec, bool ac, class H, class M, class A, class... Idxs, cuda::std::enable_if_t<!(mec && ac), int> = 0>
__device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A&, Idxs... idxs)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(mec == cuda::std::is_constructible_v<M, typename M::extents_type>);
  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  static_assert(!cuda::std::is_constructible_v<MDS, const H&, Idxs...>);
}

template <bool mec, bool ac, class H, class L, class A>
__device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  // construct from just dynamic extents
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc, 7);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc, 2, 3);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc, 0, 3);
  test_mdspan_types<mec, ac>(
    handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc, 1, 2, 3, 2);

  // construct from all extents
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc, 7);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc, 2, 4, 3);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc, 0, 7, 3);
  test_mdspan_types<mec, ac>(
    handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc, 1, 7, 2, 4, 3, 2);
}

template <bool ac, class H, class A>
__device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents<true, ac>(handle, cuda::std::layout_left(), acc);
  mixin_extents<true, ac>(handle, cuda::std::layout_right(), acc);

  // Use weird layout, make sure it has the properties we want to test
  // Sanity check that this layouts mapping is constructible from extents (via its move constructor)
  static_assert(
    cuda::std::is_constructible_v<typename layout_wrapping_integral<8>::template mapping<cuda::std::extents<int>>,
                                  cuda::std::extents<int>>);
  static_assert(
    !cuda::std::is_constructible_v<typename layout_wrapping_integral<8>::template mapping<cuda::std::extents<int>>,
                                   const cuda::std::extents<int>&>);
  mixin_extents<true, ac>(handle, layout_wrapping_integral<8>(), acc);
  // Sanity check that this layouts mapping is not constructible from extents
  static_assert(
    !cuda::std::is_constructible_v<typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>,
                                   cuda::std::extents<int>>);
  static_assert(
    !cuda::std::is_constructible_v<typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>,
                                   const cuda::std::extents<int>&>);
  mixin_extents<false, ac>(handle, layout_wrapping_integral<4>(), acc);
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

__device__ void test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();

  // test non-constructibility from wrong integer types
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  using mds_t                         = cuda::shared_memory_mdspan<float, cuda::std::extents<int, 3, D, D>>;
  // sanity check
  static_assert(cuda::std::is_constructible_v<mds_t, float*, int, int, int>);
  static_assert(cuda::std::is_constructible_v<mds_t, float*, int, int>);
  // wrong number of arguments
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, int>);
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, int, int, int, int>);
  // not convertible to int
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, int, int, cuda::std::dextents<int, 1>>);

  // test non-constructibility from wrong handle_type
  static_assert(!cuda::std::is_constructible_v<mds_t, const float*, int, int>);
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
