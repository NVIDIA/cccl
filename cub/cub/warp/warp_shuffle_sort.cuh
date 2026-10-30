// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/null_type.cuh>
#include <cub/thread/thread_sort.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__utility/swap.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! @rst
//! WarpShuffleSort sorts keys in a blocked arrangement across a warp.
//!
//! All 32 lanes must invoke the collective. Sorting is not stable: the order of equivalent keys is unspecified. The
//! comparison must have strict weak ordering.
//!
//! The algorithm requires no shared-memory scratch or associated initialization. Larger per-thread arrays can incur
//! compiler spills to local memory.
//!
//! For example, launch the following kernel with one block of 32 threads to sort 64 keys:
//!
//! .. code-block:: c++
//!
//!    #include <cub/warp/warp_shuffle_sort.cuh>
//!    #include <cuda/std/functional>
//!
//!    __global__ void example(int* sorted_output)
//!    {
//!        const int tid  = threadIdx.x;
//!        const int lane = tid % 32;
//!        int keys[2]    = {63 - 2 * lane, 62 - 2 * lane};
//!        cub::detail::WarpShuffleSort<int, 2>{}.Sort(keys, cuda::std::less<>{});
//!        sorted_output[tid * 2]     = keys[0];
//!        sorted_output[tid * 2 + 1] = keys[1];
//!    }
//!
//! Both input and output use a :ref:`blocked arrangement <flexible-data-arrangement>`:
//! rank ``lane * ItemsPerThread + i`` belongs to that lane's ``keys[i]``.
//!
//! @endrst
//!
//! @tparam KeyT
//!   Key type. Must be default constructible and trivially copyable.
//! @tparam ItemsPerThread
//!   Number of keys per thread. Must be a power of two.
template <typename KeyT, int ItemsPerThread>
class WarpShuffleSort
{
  // TODO(fbusato): Support power-of-two logical warp sizes.
  static_assert(ItemsPerThread > 0 && ::cuda::is_power_of_two(ItemsPerThread),
                "ItemsPerThread must be a positive power of two");
  static_assert(::cuda::std::is_default_constructible_v<KeyT> && ::cuda::is_trivially_copyable_v<KeyT>,
                "KeyT must be default constructible and trivially copyable");

  // Select the earlier key for a lower position or the later key for an upper position
  template <typename CompareOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static KeyT
  choose(const KeyT key, const KeyT other, const bool lower, CompareOp compare_op)
  {
    // two comparisons are needed in the case of equivalent keys, to avoid a tie, e.g. +0.0, -0.0
    const bool key_less   = compare_op(key, other);
    const bool other_less = compare_op(other, key);
    return ((key_less || other_less) && lower == other_less) ? other : key;
  }

  template <int Distance, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void
  merge_lanes(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, const int lane)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int distance = Distance; distance > 0; distance /= 2)
    {
      const bool lower = (lane & distance) == 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < ItemsPerThread; ++i)
      {
        const KeyT other = ::cuda::device::warp_shuffle_xor(keys[i], distance);
        keys[i]          = choose(keys[i], other, lower, compare_op);
      }
    }
  }
  // Precondition: keys are bitonic, i.e. keys[i] <= keys[j] for 0 < i < P and P < j < Size
  template <int Distance, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void merge_within_lane(KeyT (&keys)[ItemsPerThread], CompareOp compare_op)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int distance = Distance; distance > 0; distance /= 2)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < ItemsPerThread; ++i)
      {
        if ((i & distance) == 0 && compare_op(keys[i ^ distance], keys[i]))
        {
          ::cuda::std::swap(keys[i], keys[i ^ distance]);
        }
      }
    }
  }

  // Bitonic merge
  //   Group: size of the lanes that cooperate.
  //   Assumption: The input is already sorted
  template <int Group, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void
  merge(KeyT (&keys)[ItemsPerThread], CompareOp compare_op, const int lane)
  {
    KeyT next[ItemsPerThread];
    const bool lower = (lane & (Group / 2)) == 0; // if a lane is in the lower or upper half of the group
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const int reverse_index = ItemsPerThread - 1 - i; // opposite side of the array
      const KeyT other_key    = ::cuda::device::warp_shuffle_xor(keys[reverse_index], Group - 1);
      next[i]                 = choose(keys[i], other_key, lower, compare_op);
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      keys[i] = next[i];
    }
    // Each half is now bitonic. Within each group, every key in the lower half <= every key in the upper half
    merge_lanes<Group / 4>(keys, compare_op, lane);
    // Every key in one lane is ≤ every key in the next lane
    merge_within_lane<ItemsPerThread / 2>(keys, compare_op);
  }

public:
  struct TempStorage // Empty storage (unneeded)
  {};

  WarpShuffleSort() = default;

  explicit _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpShuffleSort(TempStorage&) {}

  // Each thread first sorts its registers using an unstable pairwise sorting network.
  // Bitonic merges then combine groups of 2, 4, 8, 16, and 32 threads, communicating through warp shuffles.
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op) const
  {
    NullType values[ItemsPerThread];
    CUB_NS_QUALIFIER::UnstablePairwiseSort(keys, values, compare_op);
    const int lane = CUB_NS_QUALIFIER::detail::logical_lane_id<warp_threads>();
    merge<2>(keys, compare_op, lane);
    merge<4>(keys, compare_op, lane);
    merge<8>(keys, compare_op, lane);
    merge<16>(keys, compare_op, lane);
    merge<32>(keys, compare_op, lane);
  }
};
} // namespace detail
CUB_NAMESPACE_END
