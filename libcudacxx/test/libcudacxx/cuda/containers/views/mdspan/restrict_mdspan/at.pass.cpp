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
//
// Throws:
//   * std::out_of_range if extents_type::index-cast(indices) is not a multidimensional index in extents_.

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../ConvertibleToIntegral.h"
#include "../CustomTestLayouts.h"
#include "test_macros.h"

#if TEST_HAS_EXCEPTIONS()
#  include <stdexcept>
#endif // TEST_HAS_EXCEPTIONS()

template <class... Args>
TEST_FUNC constexpr bool check_operator_constraints(Args...)
{
  return false;
}

template <class MDS,
          class... Indices,
          cuda::std::enable_if_t<
            cuda::std::is_same_v<decltype(cuda::std::declval<const MDS>().at(cuda::std::declval<Indices>()...)),
                                 typename MDS::reference>,
            int> = 0>
TEST_FUNC constexpr bool check_operator_constraints(MDS, Indices...)
{
  return true;
}

template <class MDS, class... Args>
TEST_FUNC constexpr void assert_access(MDS mds, Args... args)
{
  static_assert(MDS::extents_type::rank() == sizeof...(Args));

  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
  int* ptr2 = &mds.at(args...);
  assert(ptr1 == ptr2);
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
TEST_FUNC constexpr void iterate(MDS mds, Args... args)
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
TEST_FUNC constexpr void iterate(MDS mds, Args... args)
{
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  for (typename MDS::index_type i = 0; i < mds.extents().extent(r); i++)
  {
    iterate(mds, i, args...);
  }
}

#if TEST_HAS_EXCEPTIONS()
template <class Fn>
void check_at_throws_out_of_range(Fn&& fn)
{
  try
  {
    fn();
    assert(false);
  }
  catch (const std::out_of_range&)
  {
    assert(true);
  }
  catch (...)
  {
    assert(false);
  }
}

template <class MDS, class IndexType, size_t... Idxs>
void check_all_at_overloads_throw(
  MDS mds, cuda::std::array<IndexType, MDS::rank()> indices, cuda::std::index_sequence<Idxs...>)
{
  check_at_throws_out_of_range([&]() {
    return mds.at(indices[Idxs]...);
  });
  check_at_throws_out_of_range([&]() {
    return mds.at(indices);
  });
  check_at_throws_out_of_range([&]() {
    return mds.at(cuda::std::span{indices});
  });
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
void iterate_invalid(MDS mds, Args... args)
{
  if constexpr (MDS::rank() > 0)
  {
    cuda::std::array<typename MDS::index_type, MDS::rank()> args_arr{static_cast<typename MDS::index_type>(args)...};
    for (typename MDS::rank_type r = 0; r < MDS::rank(); r++)
    {
      auto invalid_args = args_arr;
      invalid_args[r]   = static_cast<typename MDS::index_type>(mds.extents().extent(r));
      check_all_at_overloads_throw(mds, invalid_args, cuda::std::make_index_sequence<MDS::rank()>());

      cuda::std::array<int, MDS::rank()> negative_args{};
      negative_args[r] = -1;
      check_all_at_overloads_throw(mds, negative_args, cuda::std::make_index_sequence<MDS::rank()>());
    }
  }
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() != sizeof...(Args)), int> = 0>
void iterate_invalid(MDS mds, Args... args)
{
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  for (typename MDS::index_type i = 0; i < mds.extents().extent(r); i++)
  {
    iterate_invalid(mds, i, args...);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

template <class Mapping>
TEST_FUNC constexpr void test_iteration(Mapping m)
{
  cuda::std::array<int, 1024> data{};
  using MDS = cuda::restrict_mdspan<int, typename Mapping::extents_type, typename Mapping::layout_type>;
  MDS mds(data.data(), m);
  iterate(mds);
}

#if TEST_HAS_EXCEPTIONS()
template <class Mapping>
void test_iteration_invalid(Mapping m)
{
  cuda::std::array<int, 1024> data{};
  using MDS = cuda::restrict_mdspan<int, typename Mapping::extents_type, typename Mapping::layout_type>;
  MDS mds(data.data(), m);

  iterate_invalid(mds);
}
#endif // TEST_HAS_EXCEPTIONS()

template <class Layout>
TEST_FUNC constexpr void test_layout()
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
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0));
  static_assert(!check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0, 0));

  // Check at constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), IntType(0)));
  static_assert(!check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1))), IntType(0)));

  // Check at constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D>(1))), IntType(0)));

  // Check that mixed integrals work: note the second one tests that mdspan casts: layout_wrapping_integral does not
  // accept IntType
  static_assert(check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D, D>(1, 1))),
    int(0),
    size_t(0)));
  static_assert(check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntType(0)));

  constexpr bool t = true;
  constexpr bool o = false;
  static_assert(!check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, o, t, t>(0)));
  static_assert(check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, t, t, t>(0)));
  static_assert(check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, t, o, t>(0)));
  static_assert(!check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<t, o, o, t>(0)));
  static_assert(check_operator_constraints(
    cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<t, o, t, o>(0)));

  // layout_wrapped wouldn't quite work here the way we wrote the check
  // IntConfig has configurable conversion properties: convert from const&, convert from non-const, no-throw-ctor from
  // const&, no-throw-ctor from non-const
  if constexpr (cuda::std::is_same<Layout, cuda::std::layout_left>::value)
  {
    static_assert(!check_operator_constraints(
      cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<o, o, t, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<o, t, t, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, o, o, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, t, o, t>(0)}));
    static_assert(check_operator_constraints(
      cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, o, t, o>(0)}));
    static_assert(check_operator_constraints(
      cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, t, t, t>(0)}));

    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, t, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, o, o, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, t, o, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, o, t, o>(0)};
      cuda::std::span s(idx);
      assert(check_operator_constraints(
        cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<t, t, t, t>(0)};
      cuda::std::span s(idx);
      assert(check_operator_constraints(
        cuda::restrict_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
  }
}

template <class Layout>
TEST_FUNC constexpr void test_layout_large()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int64_t, D, 4, D, D>(3, 5, 6)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int64_t, D, 4, 1, D>(3, 6)));
}

#if TEST_HAS_EXCEPTIONS()
template <class Layout>
void test_layout_invalid()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration_invalid(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1)));
  test_iteration_invalid(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(7)));
  test_iteration_invalid(construct_mapping(Layout(), cuda::std::extents<unsigned, 7>()));
  test_iteration_invalid(construct_mapping(Layout(), cuda::std::extents<unsigned, 7, 8>()));
  test_iteration_invalid(construct_mapping(Layout(), cuda::std::extents<signed char, D, D, D, D>(1, 1, 1, 1)));
}

void test_exceptions()
{
  test_layout_invalid<cuda::std::layout_left>();
  test_layout_invalid<cuda::std::layout_right>();
  test_layout_invalid<layout_wrapping_integral<4>>();
}
#endif // TEST_HAS_EXCEPTIONS()

// mdspan::at casts to index_type before calling mapping
// mapping requirements only require the index operator to mixed integer types not anything convertible to index_type
TEST_FUNC constexpr void test_index_cast_happens()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  int data[4]{};
  using extents_t = cuda::std::extents<int, D, D>;
  using mds_t     = cuda::restrict_mdspan<int, extents_t, layout_wrapping_integral<4>>;

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

TEST_FUNC constexpr bool test()
{
  test_layout<cuda::std::layout_left>();
  test_layout<cuda::std::layout_right>();
  test_layout<layout_wrapping_integral<4>>();
  test_index_cast_happens();
  return true;
}

int main(int, char**)
{
  test();

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  // The large test iterates over ~10k loop indices.
  // With assertions enabled this triggered the maximum default limit
  // for steps in consteval expressions. Assertions roughly double the
  // total number of instructions, so this was already close to the maximum.
  // test_large();
  return 0;
}
