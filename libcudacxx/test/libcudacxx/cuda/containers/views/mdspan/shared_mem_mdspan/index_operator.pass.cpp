//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// UNSUPPORTED: nvrtc

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
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "../ConvertibleToIntegral.h"
#include "../CustomTestLayouts.h"
#include "test_macros.h"

// GCC warns about comma operator changing its meaning inside [] in C++23
#if TEST_COMPILER(GCC, >=, 10)
TEST_DIAG_SUPPRESS_GCC("-Wcomma-subscript")
#endif // TEST_COMPILER(GCC, >=, 10)

template <class MDS>
__device__ constexpr auto& access(MDS mds, int64_t i0)
{
  return mds[i0];
}

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <
  class MDS,
  class... Indices,
  class  = cuda::std::enable_if_t<
     cuda::std::__all_v<cuda::std::is_same_v<decltype(cuda::std::declval<MDS>()[cuda::std::declval<Indices>()...]),
                                             typename MDS::reference>>,
     int> = 0>
__device__ constexpr bool check_operator_constraints(MDS m, Indices... idxs)
{
  unused(m[idxs...]);
  return true;
}
#else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
template <
  class MDS,
  class Index,
  class = cuda::std::enable_if_t<
    cuda::std::is_same_v<decltype(cuda::std::declval<MDS>()[cuda::std::declval<Index>()]), typename MDS::reference>>>
__device__ constexpr bool check_operator_constraints(MDS m, Index idx)
{
  unused(m[idx]);
  return true;
}
#endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

template <class MDS, class... Indices>
__device__ constexpr bool check_operator_constraints(MDS, Indices...)
{
  return false;
}

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <class MDS>
__device__ constexpr auto& access(MDS mds)
{
  return mds[];
}
template <class MDS>
__device__ constexpr auto& access(MDS mds, int64_t i0, int64_t i1)
{
  return mds[i0, i1];
}
template <class MDS>
__device__ constexpr auto& access(MDS mds, int64_t i0, int64_t i1, int64_t i2)
{
  return mds[i0, i1, i2];
}
template <class MDS>
__device__ constexpr auto& access(MDS mds, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
{
  return mds[i0, i1, i2, i3];
}
#endif // _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()

// We must ensure that we do not try to access multiarg accessors
template <class MDS, class Arg, cuda::std::enable_if_t<(MDS::extents_type::rank() == 1), int> = 0>
__device__ constexpr void assert_access(MDS mds, Arg arg)
{
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(arg)));
  int* ptr2 = &access(mds, arg);
  assert(ptr1 == ptr2);
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
__device__ constexpr void assert_access(MDS mds, Args... args)
{
#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  int* ptr1 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
  int* ptr2 = &access(mds, args...);
  assert(ptr1 == ptr2);
#else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
  unused(mds, args...);
#endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^
}

template <class MDS, class... Args, cuda::std::enable_if_t<(MDS::extents_type::rank() == sizeof...(Args)), int> = 0>
__device__ constexpr void iterate(MDS mds, Args... args)
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
__device__ constexpr void iterate(MDS mds, Args... args)
{
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  for (typename MDS::index_type i = 0; i < mds.extents().extent(r); i++)
  {
    iterate(mds, i, args...);
  }
}

template <class Mapping>
__device__ void test_iteration(Mapping m)
{
  __shared__ cuda::std::array<int, 1024> iteration_data;
  using MDS = cuda::shared_memory_mdspan<int, typename Mapping::extents_type, typename Mapping::layout_type>;
  MDS mds(iteration_data.data(), m);
  iterate(mds);
}

template <class Layout>
__device__ void test_layout()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, D>(7)));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, 7>()));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<unsigned, 7, 8>()));
  test_iteration(construct_mapping(Layout(), cuda::std::extents<char, D, D, D, D>(1, 1, 1, 1)));

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  test_iteration(construct_mapping(Layout(), cuda::std::extents<int>()));
  __shared__ int data[16];
  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0));
  static_assert(!check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), 0, 0));

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), IntType(0)));
  static_assert(!check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned, D>(1))), IntType(0)));

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D>(1))),
    IntType(0)));

  // Check that mixed integrals work: note the second one tests that mdspan casts: layout_wrapping_integral does not
  // accept IntType
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<unsigned char, D, D>(1, 1))),
    int(0),
    size_t(0)));
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntType(0)));

  constexpr bool t = true;
  constexpr bool o = false;
  static_assert(!check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, o, t, t>(0)));
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, t, t, t>(0)));
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<o, t, o, t>(0)));
  static_assert(!check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<t, o, o, t>(0)));
  static_assert(check_operator_constraints(
    cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D, D>(1, 1))),
    unsigned(0),
    IntConfig<t, o, t, o>(0)));

  // layout_wrapped wouldn't quite work here the way we wrote the check
  // IntConfig has configurable conversion properties: convert from const&, convert from non-const, no-throw-ctor from
  // const&, no-throw-ctor from non-const
  if constexpr (cuda::std::is_same_v<Layout, cuda::std::layout_left>)
  {
    static_assert(!check_operator_constraints(
      cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<o, o, t, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<o, t, t, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, o, o, t>(0)}));
    static_assert(!check_operator_constraints(
      cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, t, o, t>(0)}));
    static_assert(check_operator_constraints(
      cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, o, t, o>(0)}));
    static_assert(check_operator_constraints(
      cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))),
      cuda::std::array{IntConfig<t, t, t, t>(0)}));

    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
    {
      cuda::std::array idx{IntConfig<o, o, t, t>(0)};
      cuda::std::span s(idx);
      assert(!check_operator_constraints(
        cuda::shared_memory_mdspan(data, construct_mapping(Layout(), cuda::std::extents<int, D>(1))), s));
    }
  }
#endif // _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
}

__global__ void test()
{
  test_layout<cuda::std::layout_left>();
  test_layout<cuda::std::layout_right>();
  test_layout<layout_wrapping_integral<4>>();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
