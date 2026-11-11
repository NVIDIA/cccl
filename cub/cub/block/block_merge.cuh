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

#include <cub/block/block_merge_sort.cuh>
#include <cub/block/specializations/block_merge_ping_pong.cuh>
#include <cub/block/specializations/block_merge_register_staged.cuh>
#include <cub/detail/log2.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>

CUB_NAMESPACE_BEGIN

//! @brief Shared-memory strategies for BlockMerge.
enum class BlockMergeAlgorithm
{
  //! @rst
  //! Merge between the input tile and a second shared-memory tile. Each merge
  //! round uses one block barrier, and the final shared-memory result can be
  //! consumed without an additional copy. Scratch storage contains one full tile
  //! plus a readable slack key, in addition to the caller-owned input tile.
  //!
  //! This is the default strategy. Its lower barrier count can be useful when
  //! enough shared memory is available. Shared-memory use can limit occupancy.
  //! Each thread merges a fixed output window, so uneven run lengths do not
  //! concentrate the output keys in a small number of warps.
  //! @endrst
  PING_PONG,

  //! @rst
  //! Stage each merge round in per-thread registers before writing back into the
  //! caller-owned input tile. Intermediate rounds use two block barriers to
  //! separate reads from writes. The final round can return register output
  //! directly; shared output requires a final write-back and synchronization.
  //! No second shared-memory tile is allocated; TempStorage has one unused key slot.
  //!
  //! This strategy is useful when shared memory limits occupancy or tile size.
  //! It uses additional registers and barriers, which can reduce throughput.
  //! Each round uses balanced per-thread output windows. Choose between strategies
  //! using measurements with the application's key type, tile size, and run lengths.
  //! @endrst
  REGISTER_STAGED
};

// Shared implementation helpers used by both merge strategies.
namespace detail
{
// The storage type depends only on the keys and the number of scratch slots.
template <typename KeyT, unsigned int NumKeys>
struct block_merge_storage
{
  KeyT keys[NumKeys];
};

// Bracket the crossing near a proportional estimate, then reuse CUB's binary search.
template <typename KeyIt1, typename KeyIt2, typename OffsetT = int, typename CompareOp>
[[nodiscard]] _CCCL_DEVICE_API OffsetT
block_merge_path(KeyIt1 keys1, KeyIt2 keys2, OffsetT count1, OffsetT count2, OffsetT diagonal, CompareOp compare_op)
{
  static_assert(::cuda::std::is_integral_v<OffsetT>);
  using CounterT _CCCL_NODEBUG = ::cuda::std::conditional_t<(sizeof(OffsetT) < sizeof(int)), int, OffsetT>;
  const CounterT a_count       = count1;
  const CounterT b_count       = count2;
  const CounterT target        = diagonal;
  CounterT first               = target < b_count ? CounterT{0} : target - b_count;
  CounterT last                = (::cuda::std::min) (target, a_count);

  if (first < last)
  {
    const float prediction =
      ::__fdividef(static_cast<float>(target) * static_cast<float>(a_count), static_cast<float>(a_count + b_count));
    // A rounded prediction can exceed CounterT's range near its maximum value.
    // Clamp its upper end before converting back to the integral offset type.
    const CounterT estimate =
      prediction >= static_cast<float>(last) ? last - CounterT{1} : static_cast<CounterT>(prediction);
    const CounterT pivot = ::cuda::std::clamp(estimate, first, last - CounterT{1});
    const auto left      = compare_op(keys2[target - CounterT{1} - pivot], keys1[pivot]);

    if (left)
    {
      last = pivot;
    }
    else
    {
      first = pivot + CounterT{1};
    }
    for (int probe = 0; probe < 3; ++probe)
    {
      const CounterT distance = CounterT{16} << (2 * probe);
      // Check before subtracting/adding so unsigned offsets cannot wrap around.
      if (left ? distance > pivot - first : distance >= last - pivot)
      {
        break;
      }

      const CounterT index = left ? pivot - distance : pivot + distance;

      if (compare_op(keys2[target - CounterT{1} - index], keys1[index]))
      {
        last = index;
        if (!left)
        {
          break;
        }
      }
      else
      {
        first = index + CounterT{1};
        if (left)
        {
          break;
        }
      }
    }
  }
  const CounterT span = last - first;

  return static_cast<OffsetT>(
    first + CUB_NS_QUALIFIER::MergePath(keys1 + first, keys2 + (target - last), span, span, span, compare_op));
}

struct block_merge_window
{
  int begin;
  int end;
};

struct block_merge_pair
{
  int begin;
  int middle;
  int end;
};

// Reusable scheduling helpers do not depend on the key type or storage strategy.
template <unsigned int BlockThreads, unsigned int ItemsPerThread, bool UseWarpPairs = false>
struct block_merge_layout
{
  static constexpr unsigned int tile_size = BlockThreads * ItemsPerThread;
  static constexpr unsigned int warps     = BlockThreads / detail::warp_threads;
  static constexpr bool full_warps =
    BlockThreads % detail::warp_threads == 0 && warps > 0 && (warps & (warps - 1)) == 0;
  static constexpr int log_warps = CUB_NS_QUALIFIER::Log2<(warps > 0 ? warps : 1)>::VALUE;

  [[nodiscard]] _CCCL_DEVICE_API static block_merge_window thread_window(unsigned int tid, int total)
  {
    const int begin = static_cast<int>(tid * ItemsPerThread);
    const int end   = (::cuda::std::min) (begin + static_cast<int>(ItemsPerThread), total);

    return {begin, (::cuda::std::max) (begin, end)};
  }

  [[nodiscard]] _CCCL_DEVICE_API static bool use_pairs(int runs)
  {
    return UseWarpPairs && full_warps && (runs & (runs - 1)) == 0 && static_cast<unsigned int>(runs) <= 2 * warps;
  }

  [[nodiscard]] _CCCL_DEVICE_API static int effective_runs(int runs)
  {
    if (runs <= 2)
    {
      return runs;
    }

    const int padded = 1 << (32 - ::__clz(static_cast<unsigned int>(runs - 1)));
    // A lone unmatched run does not benefit from padding.
    const bool pad = UseWarpPairs && full_warps && (runs & (runs - 1)) != 0 && ((runs - 1) & (runs - 2)) != 0
                  && static_cast<unsigned int>(padded) <= 2 * warps;
    return pad ? padded : runs;
  }

  [[nodiscard]] _CCCL_DEVICE_API static int pair_thread_log(int log_runs, int round)
  {
    return log_warps - (log_runs - round - 1) + 5;
  }

  [[nodiscard]] _CCCL_DEVICE_API static block_merge_pair pair_bounds(const int* offsets, int runs, int pair, int span)
  {
    const int begin = 2 * pair * span;

    return {offsets[(::cuda::std::min) (begin, runs)],
            offsets[(::cuda::std::min) (begin + span, runs)],
            offsets[(::cuda::std::min) (begin + 2 * span, runs)]};
  }

  [[nodiscard]] _CCCL_DEVICE_API static block_merge_pair
  find_pair(int position, const int* offsets, int runs, int span, int round)
  {
    const int live = (runs + span - 1) >> round;
    int lo         = 0;
    int hi         = (live + 1) >> 1;

    while (hi - lo > 1)
    {
      const int mid = lo + (hi - lo) / 2;

      if (offsets[mid * 2 * span] <= position)
      {
        lo = mid;
      }
      else
      {
        hi = mid;
      }
    }
    return block_merge_layout::pair_bounds(offsets, runs, lo, span);
  }

  // Returns the merge window for a single thread, using log of the number of threads
  // cooperating as an input. Merge windows are determined by forcing an odd IPT
  // to minimize bank conflicts. Tail end of threads may receive less or 0 items
  // to force this odd-alignment.
  [[nodiscard]] _CCCL_DEVICE_API static block_merge_window pair_window(int count, int log_threads, int lane)
  {
    constexpr int items = static_cast<int>(ItemsPerThread);
    const int threads   = 1 << log_threads;

    if (count <= items * threads)
    {
      const int begin = (::cuda::std::min) (items * lane, count);

      return {begin, (::cuda::std::min) (begin + items, count)};
    }

    const int stride    = count <= (items + 1) * threads ? items : ((count >> log_threads) - 1) | 1;
    const int remainder = count - stride * threads;
    const int pairs     = remainder >> 1;
    const int odd       = remainder & 1;
    const int begin     = stride * lane + 2 * (::cuda::std::min) (lane, pairs) + (lane > pairs ? odd : 0);

    return {begin, begin + stride + (lane < pairs ? 2 : (lane == pairs ? odd : 0))};
  }
};

// One serial merge cursor, shared by both strategies. The one-past-run refills are
// readable because callers initialize a slack key after the tile's live prefix.
template <typename KeyT, typename CompareOp>
class block_merge_cursor
{
  const KeyT* keys;
  int index1;
  int index2;
  const int end1;
  const int end2;
  KeyT key1;
  KeyT key2;
  CompareOp compare_op;

public:
  _CCCL_DEVICE_API block_merge_cursor(const KeyT* input, block_merge_pair pair, int diagonal, CompareOp compare)
      : keys(input)
      , index1(
          pair.begin
          + CUB_NS_QUALIFIER::detail::block_merge_path(
            input + pair.begin, input + pair.middle, pair.middle - pair.begin, pair.end - pair.middle, diagonal, compare))
      , index2(pair.middle + diagonal - (index1 - pair.begin))
      , end1(pair.middle)
      , end2(pair.end)
      , key1(input[index1])
      , key2(input[index2])
      , compare_op(compare)
  {}

  [[nodiscard]] _CCCL_DEVICE_API KeyT next()
  {
    const bool take2  = index2 < end2 && (index1 >= end1 || compare_op(key2, key1));
    const KeyT result = take2 ? key2 : key1;

    if (take2)
    {
      key2 = keys[++index2];
    }
    else
    {
      key1 = keys[++index1];
    }
    return result;
  }
};

// Supplies shared helpers without making the algorithm headers include this header.
// Device-wide callers may opt into pair-warps scheduling for suitably balanced runs.
template <typename KeyT, unsigned int BlockThreads, unsigned int ItemsPerThread, bool UseWarpPairs = false>
struct block_merge_policy
{
  using key_t _CCCL_NODEBUG = KeyT;
  using layout_t _CCCL_NODEBUG =
    CUB_NS_QUALIFIER::detail::block_merge_layout<BlockThreads, ItemsPerThread, UseWarpPairs>;
  using window_t _CCCL_NODEBUG = CUB_NS_QUALIFIER::detail::block_merge_window;

  static constexpr unsigned int items_per_thread = ItemsPerThread;

  template <unsigned int Slots>
  using storage_t _CCCL_NODEBUG = CUB_NS_QUALIFIER::detail::block_merge_storage<KeyT, Slots>;

  template <typename CompareOp>
  using cursor_t _CCCL_NODEBUG = CUB_NS_QUALIFIER::detail::block_merge_cursor<KeyT, CompareOp>;

  template <unsigned int Items>
  _CCCL_DEVICE_API static void load(const KeyT* keys, unsigned int tid, int total, KeyT (&out)[Items])
  {
    const unsigned int begin = tid * Items;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned int item = 0; item < Items; ++item)
    {
      out[item] = begin + item < static_cast<unsigned int>(total) ? keys[begin + item] : KeyT{};
    }
  }

  template <unsigned int Items>
  _CCCL_DEVICE_API static void store(KeyT* keys, block_merge_window window, const KeyT (&items)[Items])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned int item = 0; item < Items; ++item)
    {
      if (static_cast<int>(item) < window.end - window.begin)
      {
        keys[window.begin + item] = items[item];
      }
    }
  }
};
} // namespace detail

//! @brief Cooperatively merges up to MaxRuns sorted runs already in shared memory.
//!
//! Runs occupy consecutive ranges [offsets[r], offsets[r + 1]) of keys, and empty runs are allowed.
//! All threads must participate with identical pointers, offsets, run count, and comparator.
//! The result is stable: equal keys from an earlier run precede equal keys from a later run.
//!
//! @rst
//! The caller initializes the live prefix and one readable slack key, then synchronizes before calling.
//! The input buffer has ``BlockDimX * BlockDimY * BlockDimZ * ItemsPerThread + 1`` key slots.
//! ``offsets`` has ``num_runs + 1`` nondecreasing entries, starts at zero, and ends at the live key count.
//! The live count must fit in the tile and ``1 <= num_runs <= MaxRuns``. Input keys are overwritten.
//! Keys must be default-constructible, copyable, and comparable by a strict weak ordering.
//! A block barrier is required before reusing the input or temporary storage after either output API.
//!
//! @endrst
//!
//! Example with sixteen shared-memory runs and blocked register output:
//!
//! @snippet catch2_test_block_merge_api.cu block-merge-example
//!
//! @tparam KeyT Key type.
//! @tparam BlockDimX X dimension of the block.
//! @tparam ItemsPerThread Maximum number of output keys per thread.
//! @tparam MaxRuns Maximum number of sorted runs (1 through 32).
//! @tparam Algorithm Shared-memory strategy.
//! @tparam BlockDimY Y dimension of the block.
//! @tparam BlockDimZ Z dimension of the block.
template <typename KeyT,
          unsigned int BlockDimX,
          unsigned int ItemsPerThread,
          unsigned int MaxRuns          = 32,
          BlockMergeAlgorithm Algorithm = BlockMergeAlgorithm::PING_PONG,
          unsigned int BlockDimY        = 1,
          unsigned int BlockDimZ        = 1>
class BlockMerge
{
private:
  static constexpr unsigned int block_threads = BlockDimX * BlockDimY * BlockDimZ;
  static_assert(BlockDimX > 0 && BlockDimY > 0 && BlockDimZ > 0 && BlockDimX <= 1024 && BlockDimY <= 1024
                  && BlockDimZ <= 64 && block_threads <= 1024,
                "Block dimensions must specify 1 through 1024 threads");
  static_assert(ItemsPerThread > 0);
  static_assert(MaxRuns >= 1 && MaxRuns <= 32);
  static_assert(ItemsPerThread <= static_cast<unsigned int>((::cuda::std::numeric_limits<int>::max)())
                                    / (::cuda::std::max) (1u, block_threads),
                "Tile size must fit in the int run offsets");

  using merge_policy_t _CCCL_NODEBUG =
    CUB_NS_QUALIFIER::detail::block_merge_policy<KeyT, block_threads, ItemsPerThread>;
  using ping_pong_t _CCCL_NODEBUG       = CUB_NS_QUALIFIER::detail::block_merge_ping_pong<merge_policy_t>;
  using register_staged_t _CCCL_NODEBUG = CUB_NS_QUALIFIER::detail::block_merge_register_staged<merge_policy_t>;

  // Delegate type for the selected algorithm.
  using internal_block_merge_t _CCCL_NODEBUG =
    ::cuda::std::conditional_t<Algorithm == BlockMergeAlgorithm::PING_PONG, ping_pong_t, register_staged_t>;
  using temp_storage_t _CCCL_NODEBUG = typename internal_block_merge_t::storage_t;

  // Shared storage reference and linear thread ID.
  temp_storage_t& temp_storage;
  const unsigned int linear_tid;

  [[nodiscard]] _CCCL_DEVICE_API static temp_storage_t& private_storage()
  {
    __shared__ CUB_NS_QUALIFIER::Uninitialized<temp_storage_t> storage;
    return storage.Alias();
  }

public:
  //! @brief Opaque shared temporary storage; may be aliased with other block primitives after a barrier.
  using TempStorage = CUB_NS_QUALIFIER::Uninitialized<temp_storage_t>;

  //! @name Collective constructors
  //! @{

  //! @brief Constructs the collective with internally allocated shared storage.
  _CCCL_DEVICE_API BlockMerge()
      : temp_storage(BlockMerge::private_storage())
      , linear_tid(CUB_NS_QUALIFIER::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @brief Constructs the collective using caller-provided shared storage.
  _CCCL_DEVICE_API explicit BlockMerge(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(CUB_NS_QUALIFIER::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @}
  //! @name Collective merge operations
  //! @{

  //! @brief Returns blocked register output. Ranks outside the live prefix are value-initialized.
  //! @param[in,out] keys Shared key buffer, including one initialized slack key after the live prefix.
  //! @param[in] offsets Shared run boundaries.
  //! @param[in] num_runs Number of sorted runs.
  //! @param[out] out Per-thread blocked output.
  //! @param[in] compare_op Strict weak ordering used to sort every input run.
  template <typename CompareOp>
  _CCCL_DEVICE_API void
  Merge(KeyT* keys, const int* offsets, int num_runs, KeyT (&out)[ItemsPerThread], CompareOp compare_op)
  {
    internal_block_merge_t(temp_storage, linear_tid).merge(keys, offsets, num_runs, out, compare_op);
  }

  //! @brief Returns the shared buffer containing the merged live prefix.
  //! @param[in,out] keys Shared key buffer, including one initialized slack key after the live prefix.
  //! @param[in] offsets Shared run boundaries.
  //! @param[in] num_runs Number of sorted runs.
  //! @param[in] compare_op Strict weak ordering used to sort every input run.
  //! @return A pointer into keys or TempStorage, valid until those allocations are reused.
  template <typename CompareOp>
  [[nodiscard]] _CCCL_DEVICE_API const KeyT*
  MergeToShared(KeyT* keys, const int* offsets, int num_runs, CompareOp compare_op)
  {
    return internal_block_merge_t(temp_storage, linear_tid).merge_to_shared(keys, offsets, num_runs, compare_op);
  }
  //! @}
};
CUB_NAMESPACE_END
