//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/mdspan>

// UNSUPPORTED: nvrtc

// template<class... Indices>
//   constexpr reference at(Indices...) const;
//
// template<class OtherIndexType>
//   constexpr reference at(const array<OtherIndexType, rank()>& indices) const;
//
// template<class OtherIndexType>
//   constexpr reference at(span<OtherIndexType, rank()> indices) const;
//
// Constraints:
//   * sizeof...(Indices) == extents_type::rank() is true,
//   * (is_convertible_v<Indices, index_type> && ...) is true, and
//   * (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../ConvertibleToIntegral.h"
#include "../CustomTestLayouts.h"
#include "test_macros.h"

template <class... Args>
TEST_TILE_DEVICE_FUNC constexpr bool check_operator_constraints(Args...)
{
  return false;
}

template <class MDS,
          class... Indices,
          cuda::std::enable_if_t<
            cuda::std::is_same_v<decltype(cuda::std::declval<const MDS>().at(cuda::std::declval<Indices>()...)),
                                 typename MDS::reference>,
            int> = 0>
TEST_TILE_DEVICE_FUNC constexpr bool check_operator_constraints(MDS, Indices...)
{
  return true;
}

template <class MDS, class... Args>
TEST_TILE_DEVICE_FUNC constexpr void assert_access(MDS mds, Args... args)
{
  static_assert(MDS::extents_type::rank() == sizeof...(Args));

  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
  int* ptr2 = &mds.at(args...);
  assert(ptr1 == ptr2);
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
TEST_TILE_DEVICE_FUNC constexpr void iterate(MDS mds, Args... args)
{
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
  assert_access(mds, args...);

  cuda::std::array<typename MDS::index_type, MDS::rank()> args_arr{static_cast<typename MDS::index_type>(args)...};
  int* ptr3 = &mds.at(args_arr);
  assert(ptr3 == ptr1);
  int* ptr4 = &mds.at(cuda::std::span<typename MDS::index_type, MDS::rank()>(args_arr));
  assert(ptr4 == ptr1);
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() != sizeof...(Args)), int> = 0>
TEST_TILE_DEVICE_FUNC constexpr void iterate(MDS mds, Args... args)
{
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  for (typename MDS::index_type i = 0; i < mds.extents().extent(r); i++)
  {
    iterate(mds, i, args...);
  }
}

template <class Mapping>
TEST_TILE_DEVICE_FUNC constexpr void test_iteration(Mapping m)
{
  cuda::std::array<int, 1024> data{};
  using MDS = cuda::device_mdspan<int, typename Mapping::extents_type, typename Mapping::layout_type>;
  MDS mds(data.data(), m);
  iterate(mds);
}

template <class Layout>
TEST_TILE_DEVICE_FUNC constexpr void test_layout()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(7)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, 7>()));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, 7, 8>()));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<signed char, D, D, D, D>(1, 1, 1, 1)));

  test_iteration(construct_mapping(Layout(), cuda::std::extents<int>()));
  int data[1]{};
  // Check at constraint for number of arguments
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0));
  static_assert(!check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0, 0));

  // Check at constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), IntType(0)));
  static_assert(!check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1))), IntType(0)));

  // Check at constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D>(1))), IntType(0)));

  // Check that mixed integrals work: note the second one tests that mdspan casts: layout_wrapping_integral does not
  // accept IntType
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D, D>(1, 1))),
    int(0),
    size_t(0)));
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntType(0)));

  constexpr bool t = true;
  constexpr bool o = false;
  static_assert(!check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, o, t, t>(0)));
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, t, t, t>(0)));
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, t, o, t>(0)));
  static_assert(!check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<t, o, o, t>(0)));
  static_assert(check_operator_constraints(
    cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<t, o, t, o>(0)));

  // layout_wrapped wouldn't quite work here the way we wrote the check
  // IntConfig has configurable conversion properties: convert from const&, convert from non-const, no-throw-ctor from
  // const&, no-throw-ctor from non-const
  if constexpr (cuda::std::is_same<Layout, cuda::std::layout_left>::value)
  {
    static_assert(!check_operator_constraints(
      cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<o, o, t, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<o, t, t, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, o, o, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, t, o, t>(0)}));
    static_assert(check_operator_constraints(
      cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, o, t, o>(0)}));
    static_assert(check_operator_constraints(
      cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, t, t, t>(0)}));

    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, t, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, o, o, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, t, o, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, o, t, o>(0)};
      cuda::std::span s(idx);
      assert(check_operator_constraints(
        cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, t, t, t>(0)};
      cuda::std::span s(idx);
      assert(check_operator_constraints(
        cuda::device_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
  }
}

template <class Layout>
TEST_TILE_DEVICE_FUNC constexpr void test_layout_large()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int64_t, D, 4, D, D>(3, 5, 6)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int64_t, D, 4, 1, D>(3, 6)));
}

// mdspan::at casts to index_type before calling mapping
// mapping requirements only require the index operator to mixed integer types not anything convertible to index_type
TEST_TILE_DEVICE_FUNC constexpr void test_index_cast_happens()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  int data[4]{};
  using extents_t = cuda::std::extents<int, D, D>;
  using mds_t     = cuda::device_mdspan<int, extents_t, layout_wrapping_integral<4>>;

  mds_t mds(data, construct_mapping(layout_wrapping_integral<4>(), extents_t(2, 2)));
  cuda::std::array idx{IntType(1), IntType(1)};
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(1, 1)));
  int* ptr2 = &mds.at(IntType(1), IntType(1));
  assert(ptr1 == ptr2);
  int* ptr3 = &mds.at(idx);
  assert(ptr1 == ptr3);
  int* ptr4 = &mds.at(cuda::std::span<IntType, 2>(idx));
  assert(ptr1 == ptr4);
}

TEST_TILE_DEVICE_FUNC constexpr bool test()
{
  test_layout<cuda::std::layout_left>();
  test_layout<cuda::std::layout_right>();
  test_layout<layout_wrapping_integral<4>>();
  test_index_cast_happens();
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, test();)

  // The large test iterates over ~10k loop indices.
  // With assertions enabled this triggered the maximum default limit
  // for steps in consteval expressions. Assertions roughly double the
  // total number of instructions, so this was already close to the maximum.
  // test_large();
  return 0;
}
