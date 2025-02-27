//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test default iteration:
//
// template<class... Indices>
//   constexpr reference operator[](Indices...) const noexcept;
//
// Constraints:
//   * sizeof...(Indices) == extents_type::rank() is true,
//   * (is_convertible_v<Indices, index_type> && ...) is true, and
//   * (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
//
// Preconditions:
//   * extents_type::index-cast(i) is a multidimensional index in extents_.

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

#include "../ConvertibleToIntegral.h"
#include "../CustomTestLayouts.h"
#include "test_macros.h"

// GCC warns about comma operator changing its meaning inside [] in C++23
#if _CCCL_COMPILER(GCC, >=, 10)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wcomma-subscript"
#endif // _CCCL_COMPILER(GCC, >=, 10)

template <class MDS>
__host__ __device__ constexpr auto& access(MDS mds, int64_t i0)
{
  return mds[i0];
}

#if defined(_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS)
template <class MDS,
          class... Indices,
          class  = cuda::std::enable_if_t<
             cuda::std::__all<cuda::std::is_same<decltype(cuda::std::declval<MDS>()[cuda::std::declval<Indices>()...]),
                                                 typename MDS::reference>::value>::value,
             int> = 0>
__host__ __device__ constexpr bool check_operator_constraints(MDS m, Indices... idxs)
{
  unused(m[idxs...]);
  return true;
}
#else // ^^^ _LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS ^^^ / vvv!_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS vvv
template <
  class MDS,
  class Index,
  class = cuda::std::enable_if_t<cuda::std::is_same<decltype(cuda::std::declval<MDS>()[cuda::std::declval<Index>()]),
                                                    typename MDS::reference>::value>>
__host__ __device__ constexpr bool check_operator_constraints(MDS m, Index idx)
{
  unused(m[idx]);
  return true;
}
#endif // !_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS

template <class MDS, class... Indices>
__host__ __device__ constexpr bool check_operator_constraints(MDS, Indices...)
{
  return false;
}

#if defined(_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS)
template <class MDS>
__host__ __device__ constexpr auto& access(MDS mds)
{
  return mds[];
}
template <class MDS>
__host__ __device__ constexpr auto& access(MDS mds, int64_t i0, int64_t i1)
{
  return mds[i0, i1];
}
template <class MDS>
__host__ __device__ constexpr auto& access(MDS mds, int64_t i0, int64_t i1, int64_t i2)
{
  return mds[i0, i1, i2];
}
template <class MDS>
__host__ __device__ constexpr auto& access(MDS mds, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
{
  return mds[i0, i1, i2, i3];
}
#endif // !_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS

// We must ensure that we do not try to access multiarg accessors
template <class MDS, class Arg, cuda::std::enable_if_t<(MDS::extents_type::rank() == 1), int> = 0>
__host__ __device__ constexpr void assert_access(MDS mds, Arg arg)
{
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(arg)));
  int* ptr2 = &access(mds, arg);
  assert(ptr1 == ptr2);
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void assert_access(MDS mds, Args... args)
{
#if defined(_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS)
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
  int* ptr2 = &access(mds, args...);
  assert(ptr1 == ptr2);
#else
  unused(mds, args...);
#endif // !_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void iterate(MDS mds, Args... args)
{
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
  assert_access(mds, args...);

  cuda::std::array<typename MDS::index_type, MDS::rank()> args_arr{static_cast<typename MDS::index_type>(args)...};
  int* ptr3 = &mds[args_arr];
  assert(ptr3 == ptr1);
  int* ptr4 = &mds[cuda::std::span<typename MDS::index_type, MDS::rank()>(args_arr)];
  assert(ptr4 == ptr1);
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() != sizeof...(Args)), int> = 0>
__host__ __device__ constexpr void iterate(MDS mds, Args... args)
{
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  for (typename MDS::index_type i = 0; i < mds.extents().extent(r); i++)
  {
    iterate(mds, i, args...);
  }
}

template <class Mapping>
__host__ __device__ constexpr void test_iteration(Mapping m)
{
  cuda::std::array<int, 1024> data{};
  using MDS = cuda::std::mdspan<int, typename Mapping::extents_type, typename Mapping::layout_type>;
  MDS mds(data.data(), m);
  iterate(mds);
}

template <class Layout>
__host__ __device__ constexpr void test_layout()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(7)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, 7>()));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, 7, 8>()));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<char, D, D, D, D>(1, 1, 1, 1)));

#if defined(_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS)
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int>()));
  int data[1];
  // Check operator constraint for number of arguments
  static_assert(
    check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0),
    "");
  static_assert(!check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0, 0),
                "");

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), IntType(0)),
                "");
  static_assert(!check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1))), IntType(0)),
                "");

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(
    !check_operator_constraints(
      cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D>(1))), IntType(0)),
    "");

  // Check that mixed integrals work: note the second one tests that mdspan casts: layout_wrapping_integral does not
  // accept IntType
  static_assert(check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D, D>(1, 1))),
                  int(0),
                  size_t(0)),
                "");
  static_assert(check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
                  unsigned(0),
                  IntType(0)),
                "");

  constexpr bool t = true;
  constexpr bool o = false;
  static_assert(!check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
                  unsigned(0),
                  IntConfig<o, o, t, t>(0)),
                "");
  static_assert(check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
                  unsigned(0),
                  IntConfig<o, t, t, t>(0)),
                "");
  static_assert(check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
                  unsigned(0),
                  IntConfig<o, t, o, t>(0)),
                "");
  static_assert(!check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
                  unsigned(0),
                  IntConfig<t, o, o, t>(0)),
                "");
  static_assert(check_operator_constraints(
                  cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
                  unsigned(0),
                  IntConfig<t, o, t, o>(0)),
                "");

  // layout_wrapped wouldn't quite work here the way we wrote the check
  // IntConfig has configurable conversion properties: convert from const&, convert from non-const, no-throw-ctor from
  // const&, no-throw-ctor from non-const
  if constexpr (cuda::std::is_same<Layout, cuda::std::layout_left>::value)
  {
    static_assert(
      !check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
                                  cuda::std::array{IntConfig<o, o, t, t>(0)}),
      "");
    static_assert(
      !check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
                                  cuda::std::array{IntConfig<o, t, t, t>(0)}),
      "");
    static_assert(
      !check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
                                  cuda::std::array{IntConfig<t, o, o, t>(0)}),
      "");
    static_assert(
      !check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
                                  cuda::std::array{IntConfig<t, t, o, t>(0)}),
      "");
    static_assert(
      check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
                                 cuda::std::array{IntConfig<t, o, t, o>(0)}),
      "");
    static_assert(
      check_operator_constraints(cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
                                 cuda::std::array{IntConfig<t, t, t, t>(0)}),
      "");

    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::std::mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
  }
#endif // _LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS
}

template <class Layout>
__host__ __device__ constexpr void test_layout_large()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int64_t, D, 4, D, D>(3, 5, 6)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int64_t, D, 4, 1, D>(3, 6)));
}

// mdspan::operator[] casts to index_type before calling mapping
// mapping requirements only require the index operator to mixed integer types not anything convertible to index_type
__host__ __device__ constexpr void test_index_cast_happens() {}

__host__ __device__ constexpr bool test()
{
  test_layout<cuda::std::layout_left>();
  test_layout<cuda::std::layout_right>();
  test_layout<layout_wrapping_integral<4>>();
  return true;
}

__host__ __device__ constexpr bool test_large()
{
  test_layout_large<cuda::std::layout_left>();
  test_layout_large<cuda::std::layout_right>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  // The large test iterates over ~10k loop indices.
  // With assertions enabled this triggered the maximum default limit
  // for steps in consteval expressions. Assertions roughly double the
  // total number of instructions, so this was already close to the maximum.
  // test_large();
  return 0;
}

#if defined(TEST_COMPILER_GCC)
#  pragma GCC diagnostic pop
#endif // TEST_COMPILER_GCC
