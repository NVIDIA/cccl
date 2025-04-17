//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>
//
//  template<class ElementType, class Extents, class LayoutPolicy = layout_right,
//           class AccessorPolicy = default_accessor<ElementType>>
//  class mdspan {
//  public:
//    using extents_type = Extents;
//    using layout_type = LayoutPolicy;
//    using accessor_type = AccessorPolicy;
//    using mapping_type = typename layout_type::template mapping<extents_type>;
//    using element_type = ElementType;
//    using value_type = remove_cv_t<element_type>;
//    using index_type = typename extents_type::index_type;
//    using size_type = typename extents_type::size_type;
//    using rank_type = typename extents_type::rank_type;
//    using data_handle_type = typename accessor_type::data_handle_type;
//    using reference = typename accessor_type::reference;
//    ...
//  };

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "CustomTestAccessors.h"
#include "test_macros.h"

// Calculated expected size of an mdspan
// Note this expects that only default_accessor is empty
template <class MDS>
__host__ __device__ constexpr size_t expected_size()
{
  size_t sizeof_dht = sizeof(typename MDS::data_handle_type);
  size_t result     = sizeof_dht;
  if (MDS::rank_dynamic() > 0)
  {
    size_t alignof_idx = alignof(typename MDS::index_type);
    size_t sizeof_idx  = sizeof(typename MDS::index_type);
    // add alignment if necessary
    result += sizeof_dht % alignof_idx == 0 ? 0 : alignof_idx - (sizeof_dht % alignof_idx);
    // add sizeof stored extents
    result += MDS::rank_dynamic() * sizeof_idx;
  }
  using A = typename MDS::accessor_type;
  if (!cuda::std::is_same<A, cuda::std::default_accessor<typename MDS::element_type>>::value)
  {
    size_t alignof_acc = alignof(A);
    size_t sizeof_acc  = sizeof(A);
    // add alignment if necessary
    result += result % alignof_acc == 0 ? 0 : alignof_acc - (result % alignof_acc);
    // add sizeof stored accessor
    result += sizeof_acc;
  }
  // add alignment of the mdspan itself
  result += result % alignof(MDS) == 0 ? 0 : alignof(MDS) - (result % alignof(MDS));
  return result;
}

// check triviality
template <class T>
constexpr bool trv_df_ctor = cuda::std::is_trivially_default_constructible<T>::value;
template <class T>
constexpr bool trv_cp_ctor = cuda::std::is_trivially_copy_constructible<T>::value;
template <class T>
constexpr bool trv_mv_ctor = cuda::std::is_trivially_move_constructible<T>::value;
template <class T>
constexpr bool trv_dstruct = cuda::std::is_trivially_destructible<T>::value;
template <class T>
constexpr bool trv_cp_asgn = cuda::std::is_trivially_copy_assignable<T>::value;
template <class T>
constexpr bool trv_mv_asgn = cuda::std::is_trivially_move_assignable<T>::value;

template <class MDS, bool default_ctor, bool copy_ctor, bool move_ctor, bool destr, bool copy_assign, bool move_assign>
__host__ __device__ constexpr void check_triviality()
{
  static_assert(trv_df_ctor<MDS> == default_ctor, "");
  static_assert(trv_cp_ctor<MDS> == copy_ctor, "");
  static_assert(trv_mv_ctor<MDS> == move_ctor, "");
  static_assert(trv_dstruct<MDS> == destr, "");
  static_assert(trv_cp_asgn<MDS> == copy_assign, "");
  static_assert(trv_mv_asgn<MDS> == move_assign, "");
}

// Standard extension
template <class L,
          class MDS,
          cuda::std::enable_if_t<cuda::std::is_same<L, cuda::std::layout_left>::value
                                   || cuda::std::is_same<L, cuda::std::layout_right>::value,
                                 int> = 0>
__host__ __device__ constexpr void test_mdspan_no_unique_address()
{
#if _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS()
  static_assert(sizeof(MDS) == expected_size<MDS>(), "");
#endif // _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS()
}

template <class L,
          class MDS,
          cuda::std::enable_if_t<!cuda::std::is_same<L, cuda::std::layout_left>::value
                                   && !cuda::std::is_same<L, cuda::std::layout_right>::value,
                                 int> = 0>
__host__ __device__ constexpr void test_mdspan_no_unique_address()
{}

template <class T, class E, class L, class A>
__host__ __device__ constexpr void test_mdspan_types()
{
  using MDS = cuda::std::mdspan<T, E, L, A>;

  static_assert(cuda::std::is_same_v<typename MDS::extents_type, E>);
  static_assert(cuda::std::is_same_v<typename MDS::layout_type, L>);
  static_assert(cuda::std::is_same_v<typename MDS::accessor_type, A>);
  static_assert(cuda::std::is_same_v<typename MDS::mapping_type, typename L::template mapping<E>>);
  static_assert(cuda::std::is_same_v<typename MDS::element_type, T>);
  static_assert(cuda::std::is_same_v<typename MDS::value_type, cuda::std::remove_cv_t<T>>);
  static_assert(cuda::std::is_same_v<typename MDS::index_type, typename E::index_type>);
  static_assert(cuda::std::is_same_v<typename MDS::size_type, typename E::size_type>);
  static_assert(cuda::std::is_same_v<typename MDS::rank_type, typename E::rank_type>);
  static_assert(cuda::std::is_same_v<typename MDS::data_handle_type, typename A::data_handle_type>);
  static_assert(cuda::std::is_same_v<typename MDS::reference, typename A::reference>);

  // This miserably failed with clang-cl - likely because it doesn't honor/enable
  // no-unique-address fully by default
  test_mdspan_no_unique_address<L, MDS>();

  // check default template parameters:
  static_assert(cuda::std::is_same_v<cuda::std::mdspan<T, E>,
                                     cuda::std::mdspan<T, E, cuda::std::layout_right, cuda::std::default_accessor<T>>>);
  static_assert(
    cuda::std::is_same_v<cuda::std::mdspan<T, E, L>, cuda::std::mdspan<T, E, L, cuda::std::default_accessor<T>>>);

  // check triviality
  using DH = typename MDS::data_handle_type;
  using MP = typename MDS::mapping_type;

  check_triviality<MDS,
                   false, // mdspan is never trivially constructible right now
                   trv_cp_ctor<DH> && trv_cp_ctor<MP> && trv_cp_ctor<A>,
                   trv_mv_ctor<DH> && trv_mv_ctor<MP> && trv_mv_ctor<A>,
                   trv_dstruct<DH> && trv_dstruct<MP> && trv_dstruct<A>,
                   trv_cp_asgn<DH> && trv_cp_asgn<MP> && trv_cp_asgn<A>,
                   trv_mv_asgn<DH> && trv_mv_asgn<MP> && trv_mv_asgn<A>>();
}

template <class T, class L, class A>
__host__ __device__ constexpr void mixin_extents()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types<T, cuda::std::extents<int>, L, A>();
  test_mdspan_types<T, cuda::std::extents<char, D>, L, A>();
  test_mdspan_types<T, cuda::std::dextents<char, 7>, L, A>();
  test_mdspan_types<T, cuda::std::dextents<char, 9>, L, A>();
  test_mdspan_types<T, cuda::std::extents<unsigned, 7>, L, A>();
  test_mdspan_types<T, cuda::std::extents<unsigned, D, D, D>, L, A>();
  test_mdspan_types<T, cuda::std::extents<size_t, D, 7, D>, L, A>();
  test_mdspan_types<T, cuda::std::extents<int64_t, D, 7, D, 4, D, D>, L, A>();
}

template <class T, class A>
__host__ __device__ constexpr void mixin_layout()
{
  mixin_extents<T, cuda::std::layout_left, A>();
  mixin_extents<T, cuda::std::layout_right, A>();
  mixin_extents<T, layout_wrapping_integral<4>, A>();
}

template <class T>
__host__ __device__ constexpr void mixin_accessor()
{
  mixin_layout<T, cuda::std::default_accessor<T>>();
  mixin_layout<T, checked_accessor<T>>();
}

__host__ __device__ constexpr bool test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();

  // sanity checks for triviality
  check_triviality<cuda::std::mdspan<int, cuda::std::extents<int>>, false, true, true, true, true, true>();
  check_triviality<cuda::std::mdspan<int, cuda::std::dextents<int, 1>>, false, true, true, true, true, true>();
  check_triviality<cuda::std::mdspan<int, cuda::std::dextents<int, 1>, cuda::std::layout_right, checked_accessor<int>>,
                   false,
                   true,
                   false,
                   true,
                   true,
                   true>();

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

  static_assert(test(), "");
#if TEST_STD_VER >= 2020
  static_assert(test_evil(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
