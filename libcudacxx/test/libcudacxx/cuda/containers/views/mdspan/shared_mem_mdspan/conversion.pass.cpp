//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class OtherElementType, class OtherExtents,
//         class OtherLayoutPolicy, class OtherAccessor>
//  constexpr explicit(see below)
//    mdspan(const mdspan<OtherElementType, OtherExtents,
//                        OtherLayoutPolicy, OtherAccessor>& other);
//
// Constraints:
//   - is_constructible_v<mapping_type, const OtherLayoutPolicy::template mapping<OtherExtents>&> is true, and
//   - is_constructible_v<accessor_type, const OtherAccessor&> is true.
// Mandates:
//   - is_constructible_v<data_handle_type, const OtherAccessor::data_handle_type&> is
//   - is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions:
//   - For each rank index r of extents_type, static_extent(r) == dynamic_extent || static_extent(r) == other.extent(r)
//   is true.
//   - [0, map_.required_span_size()) is an accessible range of ptr_ and acc_ for values of ptr_, map_, and acc_ after
//   the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with other.ptr_,
//   - direct-non-list-initializes map_ with other.map_, and
//   - direct-non-list-initializes acc_ with other.acc_.
//
// Remarks: The expression inside explicit is equivalent to:
//   !is_convertible_v<const OtherLayoutPolicy::template mapping<OtherExtents>&, mapping_type>
//   || !is_convertible_v<const OtherAccessor&, accessor_type>
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <class ToMDS, class FromMDS>
__device__ constexpr void test_implicit_conversion(ToMDS to_mds, FromMDS from_mds)
{
  assert(to_mds.extents() == from_mds.extents());
  test_equality_with_handle(to_mds, from_mds);
  test_equality_with_mapping(to_mds, from_mds);
  test_equality_with_accessor(to_mds, from_mds);
}

template <class M>
constexpr bool mapping_requirements =
  cuda::std::copyable<M> && cuda::std::equality_comparable<M> && cuda::std::is_nothrow_move_constructible_v<M>
  && cuda::std::is_nothrow_move_assignable_v<M> && cuda::std::is_nothrow_swappable_v<M>;

template <class ToMDS,
          class FromMDS,
          bool constructible,
          bool convertible,
          bool passes_mandates,
          cuda::std::enable_if_t<!constructible, int> = 0>
__device__ constexpr void test_conversion_impl(FromMDS)
{
  static_assert(!cuda::std::is_constructible_v<ToMDS, FromMDS>);
}
template <class ToMDS,
          class FromMDS,
          bool constructible,
          bool convertible,
          bool passes_mandates,
          cuda::std::enable_if_t<constructible, int>    = 0,
          cuda::std::enable_if_t<!passes_mandates, int> = 0>
__device__ constexpr void test_conversion_impl(FromMDS)
{}
template <class ToMDS,
          class FromMDS,
          bool constructible,
          bool convertible,
          bool passes_mandates,
          cuda::std::enable_if_t<constructible, int>   = 0,
          cuda::std::enable_if_t<passes_mandates, int> = 0,
          cuda::std::enable_if_t<convertible, int>     = 0>
__device__ constexpr void test_conversion_impl(FromMDS from_mds)
{
  ToMDS to_mds(from_mds);
  assert(to_mds.extents() == from_mds.extents());
  test_equality_with_handle(to_mds, from_mds);
  test_equality_with_mapping(to_mds, from_mds);
  test_equality_with_accessor(to_mds, from_mds);
  test_implicit_conversion(from_mds, from_mds);
}
template <class ToMDS,
          class FromMDS,
          bool constructible,
          bool convertible,
          bool passes_mandates,
          cuda::std::enable_if_t<constructible, int>   = 0,
          cuda::std::enable_if_t<passes_mandates, int> = 0,
          cuda::std::enable_if_t<!convertible, int>    = 0>
__device__ constexpr void test_conversion_impl(FromMDS from_mds)
{
  ToMDS to_mds(from_mds);
  assert(to_mds.extents() == from_mds.extents());
  test_equality_with_handle(to_mds, from_mds);
  test_equality_with_mapping(to_mds, from_mds);
  test_equality_with_accessor(to_mds, from_mds);
  static_assert(!cuda::std::is_convertible_v<FromMDS, ToMDS>);
}

template <class ToMDS, class FromMDS>
__device__ constexpr void test_conversion(FromMDS from_mds)
{
  // check some requirements, to see we didn't screw up our test layouts/accessors
  static_assert(cuda::std::copyable<typename ToMDS::mapping_type>);
  static_assert(cuda::std::equality_comparable<typename ToMDS::mapping_type>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<typename ToMDS::mapping_type>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<typename ToMDS::mapping_type>);
  static_assert(cuda::std::is_nothrow_swappable_v<typename ToMDS::mapping_type>);
  static_assert(mapping_requirements<typename ToMDS::mapping_type>);
  static_assert(mapping_requirements<typename FromMDS::mapping_type>);

  constexpr bool constructible =
    cuda::std::is_constructible_v<typename ToMDS::mapping_type, const typename FromMDS::mapping_type&>
    && cuda::std::is_constructible_v<typename ToMDS::accessor_type, const typename FromMDS::accessor_type&>;
  constexpr bool convertible =
    cuda::std::is_convertible_v<const typename FromMDS::mapping_type&, typename ToMDS::mapping_type>
    && cuda::std::is_convertible_v<const typename FromMDS::accessor_type&, typename ToMDS::accessor_type>;
  constexpr bool passes_mandates =
    cuda::std::is_constructible_v<typename ToMDS::data_handle_type, const typename FromMDS::data_handle_type&>
    && cuda::std::is_constructible_v<typename ToMDS::extents_type, typename FromMDS::extents_type>;

  test_conversion_impl<ToMDS, FromMDS, constructible, convertible, passes_mandates>(from_mds);
}

template <class ToL, class ToExt, class ToA, class FromH, class FromL, class FromExt, class FromA>
__device__ constexpr void
construct_from_mds(const FromH& handle, const FromL& layout, const FromExt& exts, const FromA& acc)
{
  using ToMDS   = cuda::shared_memory_mdspan<typename ToA::element_type, ToExt, ToL, ToA>;
  using FromMDS = cuda::shared_memory_mdspan<typename FromA::element_type, FromExt, FromL, FromA>;
  test_conversion<ToMDS>(FromMDS(handle, construct_mapping(layout, exts), acc));
}

template <class ToL, class ToA, class FromH, class FromL, class FromA>
__device__ constexpr void mixin_extents(const FromH& handle, const FromL& layout, const FromA& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  // constructible and convertible
  construct_from_mds<ToL, cuda::std::dextents<int, 0>, ToA>(handle, layout, cuda::std::dextents<int, 0>(), acc);
  construct_from_mds<ToL, cuda::std::dextents<int, 1>, ToA>(handle, layout, cuda::std::dextents<int, 1>(4), acc);
  construct_from_mds<ToL, cuda::std::dextents<int, 1>, ToA>(handle, layout, cuda::std::extents<int, 4>(), acc);
  construct_from_mds<ToL, cuda::std::dextents<int, 2>, ToA>(handle, layout, cuda::std::dextents<int, 2>(4, 5), acc);
  construct_from_mds<ToL, cuda::std::dextents<unsigned, 2>, ToA>(handle, layout, cuda::std::dextents<int, 2>(4, 5), acc);
  construct_from_mds<ToL, cuda::std::dextents<unsigned, 2>, ToA>(handle, layout, cuda::std::extents<int, D, 5>(4), acc);
  construct_from_mds<ToL, cuda::std::extents<int, D, 5>, ToA>(handle, layout, cuda::std::extents<int, D, 5>(4), acc);
  construct_from_mds<ToL, cuda::std::extents<int, D, 5>, ToA>(handle, layout, cuda::std::extents<int, D, 5>(4), acc);
  construct_from_mds<ToL, cuda::std::extents<int, D, 5, D, 7>, ToA>(
    handle, layout, cuda::std::extents<int, D, 5, D, 7>(4, 6), acc);

  // not convertible
  construct_from_mds<ToL, cuda::std::dextents<int, 1>, ToA>(handle, layout, cuda::std::dextents<unsigned, 1>(4), acc);
  construct_from_mds<ToL, cuda::std::extents<int, D, 5, D, 7>, ToA>(
    handle, layout, cuda::std::extents<int, D, 5, D, D>(4, 6, 7), acc);

  // not constructible
  construct_from_mds<ToL, cuda::std::dextents<int, 1>, ToA>(handle, layout, cuda::std::dextents<int, 2>(4, 5), acc);
  construct_from_mds<ToL, cuda::std::extents<int, D, 5, D, 8>, ToA>(
    handle, layout, cuda::std::extents<int, D, 5, D, 7>(4, 6), acc);
}

template <class ToA, class FromH, class FromA>
__device__ constexpr void mixin_layout(const FromH& handle, const FromA& acc)
{
  mixin_extents<cuda::std::layout_left, ToA>(handle, cuda::std::layout_left(), acc);
  mixin_extents<cuda::std::layout_right, ToA>(handle, cuda::std::layout_right(), acc);
  // Check layout policy conversion
  // different layout policies, but constructible and convertible
  static_assert(cuda::std::is_constructible_v<cuda::std::layout_left::mapping<cuda::std::dextents<int, 1>>,
                                              const cuda::std::layout_right::mapping<cuda::std::dextents<int, 1>>&>);
  static_assert(cuda::std::is_convertible_v<const cuda::std::layout_right::mapping<cuda::std::dextents<int, 1>>&,
                                            cuda::std::layout_left::mapping<cuda::std::dextents<int, 1>>>);
  // different layout policies, not constructible
  static_assert(!cuda::std::is_constructible_v<cuda::std::layout_left::mapping<cuda::std::dextents<int, 2>>,
                                               const cuda::std::layout_right::mapping<cuda::std::dextents<int, 2>>&>);
  // different layout policies, constructible and not convertible
  static_assert(cuda::std::is_constructible_v<cuda::std::layout_left::mapping<cuda::std::dextents<int, 1>>,
                                              const cuda::std::layout_right::mapping<cuda::std::dextents<size_t, 1>>&>);
  static_assert(!cuda::std::is_convertible_v<const cuda::std::layout_right::mapping<cuda::std::dextents<size_t, 1>>&,
                                             cuda::std::layout_left::mapping<cuda::std::dextents<int, 1>>>);

  mixin_extents<cuda::std::layout_left, ToA>(handle, cuda::std::layout_right(), acc);
  mixin_extents<layout_wrapping_integral<4>, ToA>(handle, layout_wrapping_integral<4>(), acc);
  // different layout policies, constructible and not convertible
  static_assert(
    !cuda::std::is_constructible_v<layout_wrapping_integral<8>::mapping<cuda::std::dextents<unsigned, 2>>,
                                   const layout_wrapping_integral<8>::mapping<cuda::std::dextents<int, 2>>&>);
  static_assert(cuda::std::is_constructible_v<layout_wrapping_integral<8>::mapping<cuda::std::dextents<unsigned, 2>>,
                                              layout_wrapping_integral<8>::mapping<cuda::std::dextents<int, 2>>>);
  mixin_extents<layout_wrapping_integral<8>, ToA>(handle, layout_wrapping_integral<8>(), acc);
}

// check that we cover all corners with respect to constructibility and convertibility
template <class ToA,
          class FromA,
          cuda::std::enable_if_t<!cuda::std::is_same_v<typename FromA::element_type, MinimalElementType>
                                   && !cuda::std::is_same_v<typename ToA::element_type, MinimalElementType>,
                                 int> = 0>
__device__ constexpr void test_impl(FromA from_acc)
{
  cuda::std::array<typename FromA::element_type, 1024> elements = {42};
  mixin_layout<ToA>(typename FromA::data_handle_type(elements.data()), from_acc);
}

template <class ToA,
          class FromA,
          cuda::std::enable_if_t<cuda::std::is_same_v<typename FromA::element_type, MinimalElementType>
                                   || cuda::std::is_same_v<typename ToA::element_type, MinimalElementType>,
                                 int> = 0>
__device__ void test_impl(FromA from_acc)
{
  ElementPool<typename FromA::element_type, 1024> elements;
  mixin_layout<ToA>(typename FromA::data_handle_type(elements.get_ptr()), from_acc);
}

template <bool constructible_constref_acc,
          bool convertible_constref_acc,
          bool constructible_nonconst_acc,
          bool convertible_nonconst_acc,
          bool constructible_constref_handle,
          bool convertible_constref_handle,
          bool constructible_nonconst_handle,
          bool convertible_nonconst_handle,
          class ToA,
          class FromA>
__device__ void test(FromA from_acc)
{
  static_assert(cuda::std::copyable<ToA>);
  static_assert(cuda::std::copyable<FromA>);
  static_assert(cuda::std::is_constructible_v<ToA, const FromA&> == constructible_constref_acc);
  static_assert(cuda::std::is_constructible_v<ToA, FromA> == constructible_nonconst_acc);
  static_assert(cuda::std::is_constructible_v<typename ToA::data_handle_type, const typename FromA::data_handle_type&>
                == constructible_constref_handle);
  static_assert(cuda::std::is_constructible_v<typename ToA::data_handle_type, typename FromA::data_handle_type>
                == constructible_nonconst_handle);
  static_assert(cuda::std::is_convertible_v<const FromA&, ToA> == convertible_constref_acc);
  static_assert(cuda::std::is_convertible_v<FromA, ToA> == convertible_nonconst_acc);
  static_assert(cuda::std::is_convertible_v<const typename FromA::data_handle_type&, typename ToA::data_handle_type>
                == convertible_constref_handle);
  static_assert(cuda::std::is_convertible_v<typename FromA::data_handle_type, typename ToA::data_handle_type>
                == convertible_nonconst_handle);

  test_impl<ToA>(from_acc);
}

__device__ void run_conversion_tests()
{
  // using shorthands here: t and o for better visual distinguishability
  constexpr bool t = true;
  constexpr bool o = false;

  // possibility matrix for constructibility and convertibility https://godbolt.org/z/98KGo8Wbc
  // you can't have convertibility without constructibility
  // and if you take const T& then you also can take T
  // this leaves 7 combinations
  // const_ref_ctor, const_ref_conv, nonconst_ctor, nonconst_conv, tested
  // o o o o X
  // o o t o X
  // o o t t X
  // t o t o X
  // t o t t X
  // t t t o X
  // t t t t X

  // conv_test_accessor_c/nc is an accessor pair which has configurable conversion properties, but plain ptr as data
  // handle accessor constructible
  test<t, t, t, t, t, t, t, t, cuda::std::default_accessor<float>>(cuda::std::default_accessor<float>());
  test<t, t, t, t, t, t, t, t, cuda::std::default_accessor<const float>>(cuda::std::default_accessor<float>());
  test<t, t, t, t, t, t, t, t, cuda::std::default_accessor<MinimalElementType>>(
    cuda::std::default_accessor<MinimalElementType>());
  test<t, t, t, t, t, t, t, t, cuda::std::default_accessor<const MinimalElementType>>(
    cuda::std::default_accessor<MinimalElementType>());
  test<t, o, t, o, t, t, t, t, conv_test_accessor_c<int, t, t, t, t>>(conv_test_accessor_nc<int, t, t>());
  test<t, o, t, t, t, t, t, t, conv_test_accessor_c<int, t, t, o, o>>(conv_test_accessor_nc<int, t, o>());
// FIXME: these tests trigger what appears to be a compiler bug on MINGW32 with --target=x86_64-w64-windows-gnu
// https://godbolt.org/z/KK8aj5bs7
// Bug report: https://github.com/llvm/llvm-project/issues/64077
#if !TEST_COMPILER(MSVC)
  test<t, t, t, o, t, t, t, t, conv_test_accessor_c<int, o, t, t, t>>(conv_test_accessor_nc<int, t, t>());
  test<t, t, t, t, t, t, t, t, conv_test_accessor_c<int, o, o, o, o>>(conv_test_accessor_nc<int, t, o>());
#endif // !TEST_COMPILER(MSVC)

  // ElementType convertible, but accessor not constructible
  test<o, o, o, o, o, o, o, o, cuda::std::default_accessor<float>>(cuda::std::default_accessor<int>());
  test<o, o, o, o, t, t, t, t, conv_test_accessor_c<int, o, o, t, t>>(conv_test_accessor_nc<int, o, o>());
  test<o, o, t, o, t, t, t, t, conv_test_accessor_c<int, o, t, o, o>>(conv_test_accessor_nc<int, o, t>());
  test<o, o, t, t, t, t, t, t, conv_test_accessor_c<int, o, o, t, t>>(conv_test_accessor_nc<int, o, t>());
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (run_conversion_tests();))
  return 0;
}
