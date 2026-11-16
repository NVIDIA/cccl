// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/thread/thread_sort.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/less.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace group_merge_sort
{
/**
 * @brief GroupMergeSort provides a reusable runtime-width group merge-sort primitive for segmented sort.
 *
 * @rst
 * Overview
 * ++++++++++++++++++++++++++
 *
 * Fine-grained segmented sort assigns a configurable group of cooperating threads to each segment.
 * The same merge-sort building block applies whether that group contains fewer than one warp, exactly
 * one warp, or multiple warps; the cases differ only in the number of threads cooperating on a segment.
 *
 * ``GroupMergeSort`` accepts ``threads_per_group`` (or ``threads_per_segment``) at runtime and uses the
 * same unified implementation structure for:
 *   - Sub-warp groups (``threads_per_group < 32``), e.g. 1, 2, 4, 8, 16 threads
 *   - Warp groups (``threads_per_group == 32``)
 *   - Multi-warp groups (``threads_per_group > 32``), e.g. 64, 128, 256, 512, 1024 threads
 *
 * Key Guarantees
 * ++++++++++++++++++++++++++
 *
 * 1. **Group Isolation**: Sorts strictly within independent groups of consecutive threads.
 * 2. **Boundary-Safe Synchronization**:
 *      - Sub-warp groups use ``__syncwarp(mask)`` with dynamically calculated member lane masks.
 *      - Warp groups use full warp masks ``__syncwarp(0xFFFFFFFF)``.
 *      - Multi-warp groups use hardware named barriers ``bar.sync (1 + group_id), threads_per_group``.
 *        A CTA may host at most 15 concurrent multi-warp groups (barriers 1..15; barrier 0 is reserved for
 *        ``__syncthreads()``).
 * 3. **Shared Memory Isolation**: Shared memory accesses never cross group bounds. Thread 0 of each group
 *    initializes a dedicated padding element past its tile slice to satisfy serial merge prefetch.
 * 4. **Runtime Clamping**: Supports runtime ``valid_items`` for partially filled groups without requiring
 *    sentinel values (``oob_default``) or out-of-bounds padding reads. Redundant merge passes are skipped early.
 * 5. **Keys and Key-Value Pairs**: Efficient zero-overhead sorting for keys-only (``ValueT = NullType``) and
 *    lockstep value gathering for key-value pairs.
 * 6. **Ordering**: Supports arbitrary Strict Weak Ordering comparison functors (ascending, descending, custom).
 * 7. **Storage Lifetime and Reuse**: A ``Sort`` invocation does not perform a trailing synchronization upon return;
 *    threads that complete serial merge early return immediately. Callers intending to reuse ``temp_storage`` (or
 *    the shared memory buffer slice) directly after ``Sort`` must perform group-appropriate synchronization (e.g.
 *    ``__syncwarp()`` for sub-warp/warp groups, or a named barrier for multi-warp groups) before overwriting shared
 * memory.
 *
 * Example
 * ++++++++++++++++++++++++++
 *
 * .. code-block:: c++
 *
 *    #include <cub/detail/group_merge_sort.cuh>
 *
 *    __global__ void SegmentedKernel(int* keys, int threads_per_segment, int valid_items)
 *    {
 *        using GroupSortT = cub::detail::GroupMergeSort<int, 4, 256>;
 *        __shared__ typename GroupSortT::TempStorage temp_storage[4];
 *
 *        const int group_id = threadIdx.x / threads_per_segment;
 *        const int member_tid = threadIdx.x % threads_per_segment;
 *
 *        GroupSortT sort(temp_storage[group_id], member_tid, group_id, threads_per_segment);
 *
 *        int thread_keys[4];
 *        // load keys ...
 *
 *        sort.Sort(thread_keys, ::cuda::std::less<int>{}, threads_per_segment, valid_items);
 *        // store keys ...
 *    }
 *
 * @endrst
 *
 * @tparam KeyT
 *   Key type
 * @tparam ITEMS_PER_THREAD
 *   Number of items processed per thread
 * @tparam MAX_GROUP_THREADS
 *   Maximum number of threads in a cooperating group (used to size TempStorage, default 1024).
 *   Must be a power of two.
 * @tparam ValueT
 *   Value type. cub::NullType indicates keys-only sort (default: NullType)
 * @tparam Unroll
 *   Whether to unroll inner sorting and merging loops (default: true)
 */
template <typename KeyT, int ITEMS_PER_THREAD, int MAX_GROUP_THREADS = 1024, typename ValueT = NullType, bool Unroll = true>
class GroupMergeSort
{
  static_assert(ITEMS_PER_THREAD > 0, "ITEMS_PER_THREAD must be greater than 0");
  static_assert(MAX_GROUP_THREADS > 0, "MAX_GROUP_THREADS must be greater than 0");
  static_assert(::cuda::is_power_of_two(MAX_GROUP_THREADS), "MAX_GROUP_THREADS must be a power of two");

public:
  static constexpr bool KEYS_ONLY         = ::cuda::std::is_same_v<ValueT, NullType>;
  static constexpr int MAX_ITEMS_PER_TILE = ITEMS_PER_THREAD * MAX_GROUP_THREADS;

#ifndef _CCCL_DOXYGEN_INVOKED
  /// Temporary storage required by one cooperating group.
  /// Padded by 1 element because serial merge prefetches one item past the end of a run.
  union _TempStorage
  {
    KeyT keys_shared[MAX_ITEMS_PER_TILE + 1];
    ValueT items_shared[MAX_ITEMS_PER_TILE + 1];
  };
#endif // _CCCL_DOXYGEN_INVOKED

  /// Temporary storage wrapper type required by GroupMergeSort
  struct TempStorage : Uninitialized<_TempStorage>
  {};

private:
  KeyT* keys_shared;
  ValueT* items_shared;
  const unsigned int member_tid;
  const unsigned int group_id;

public:
  GroupMergeSort() = delete;

  /**
   * @brief Collective constructor using specified temporary storage, explicit member_tid within group,
   *        and group_id within CTA.
   *
   * @param[in,out] temp_storage Reference to this group's temporary storage allocation.
   * @param[in] member_tid Rank of calling thread within its cooperating group (0 <= member_tid < threads_per_group).
   * @param[in] group_id Rank of the group within the CTA (0 <= group_id < 15 for multi-warp groups).
   * @param[in] threads_per_group Number of threads cooperating in this group (default: MAX_GROUP_THREADS).
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE GroupMergeSort(
    TempStorage& temp_storage,
    unsigned int member_tid,
    unsigned int group_id,
    int threads_per_group = MAX_GROUP_THREADS) noexcept
      : keys_shared(temp_storage.Alias().keys_shared)
      , items_shared(temp_storage.Alias().items_shared)
      , member_tid(member_tid)
      , group_id(group_id)
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(threads_per_group <= MAX_GROUP_THREADS,
                 "threads_per_group must not exceed MAX_GROUP_THREADS for TempStorage allocation");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(threads_per_group <= 32 || group_id < 15u,
                 "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  }

  /**
   * @brief Collective constructor calculating member_tid and group_id from CTA-level linear_tid
   *        and runtime group width.
   *
   * @param[in,out] temp_storage Reference to this group's temporary storage allocation.
   * @param[in] linear_tid Rank of calling thread within the CTA (e.g. threadIdx.x).
   * @param[in] threads_per_group Number of threads cooperating in this group (must be a power of two).
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE
  GroupMergeSort(TempStorage& temp_storage, unsigned int linear_tid, int threads_per_group) noexcept
      : keys_shared(temp_storage.Alias().keys_shared)
      , items_shared(temp_storage.Alias().items_shared)
      , member_tid(threads_per_group > 0 ? (linear_tid % static_cast<unsigned int>(threads_per_group)) : 0)
      , group_id(threads_per_group > 0 ? (linear_tid / static_cast<unsigned int>(threads_per_group)) : 0)
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(threads_per_group <= MAX_GROUP_THREADS,
                 "threads_per_group must not exceed MAX_GROUP_THREADS for TempStorage allocation");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(threads_per_group <= 32
                   || (threads_per_group > 0 && (linear_tid / static_cast<unsigned int>(threads_per_group)) < 15u),
                 "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  }

  /**
   * @brief Constructor accepting raw shared memory buffer pointers for sliced allocations.
   *
   * @param[in] keys_shared_ptr Pointer to the group's slice of shared memory for keys.
   *            Must provide at least `threads_per_group * ITEMS_PER_THREAD + 1` elements;
   *            the trailing element is the serial-merge prefetch padding slot.
   * @param[in] items_shared_ptr Pointer to the group's slice of shared memory for values.
   *            Must provide at least `threads_per_group * ITEMS_PER_THREAD + 1` elements
   *            for key-value sorts; may be null for keys-only sorts.
   * @param[in] member_tid Rank of calling thread within its cooperating group (0 <= member_tid < threads_per_group).
   * @param[in] group_id Rank of the group within the CTA (0 <= group_id < 15 for multi-warp groups).
   * @param[in] threads_per_group Number of threads cooperating in this group (default: MAX_GROUP_THREADS).
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE GroupMergeSort(
    KeyT* keys_shared_ptr,
    ValueT* items_shared_ptr,
    unsigned int member_tid,
    unsigned int group_id,
    int threads_per_group = MAX_GROUP_THREADS) noexcept
      : keys_shared(keys_shared_ptr)
      , items_shared(items_shared_ptr)
      , member_tid(member_tid)
      , group_id(group_id)
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(threads_per_group <= 32 || group_id < 15u,
                 "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  }

  /**
   * @brief Constructor accepting raw shared memory buffer pointers and calculating ranks from linear_tid.
   *
   * @param[in] keys_shared_ptr Pointer to the group's slice of shared memory for keys.
   *            Must provide at least `threads_per_group * ITEMS_PER_THREAD + 1` elements;
   *            the trailing element is the serial-merge prefetch padding slot.
   * @param[in] items_shared_ptr Pointer to the group's slice of shared memory for values.
   *            Must provide at least `threads_per_group * ITEMS_PER_THREAD + 1` elements
   *            for key-value sorts; may be null for keys-only sorts.
   * @param[in] linear_tid Rank of calling thread within the CTA (e.g. threadIdx.x).
   * @param[in] threads_per_group Number of threads cooperating in this group (must be a power of two).
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE GroupMergeSort(
    KeyT* keys_shared_ptr, ValueT* items_shared_ptr, unsigned int linear_tid, int threads_per_group) noexcept
      : keys_shared(keys_shared_ptr)
      , items_shared(items_shared_ptr)
      , member_tid(threads_per_group > 0 ? (linear_tid % static_cast<unsigned int>(threads_per_group)) : 0)
      , group_id(threads_per_group > 0 ? (linear_tid / static_cast<unsigned int>(threads_per_group)) : 0)
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(threads_per_group <= 32
                   || (threads_per_group > 0 && (linear_tid / static_cast<unsigned int>(threads_per_group)) < 15u),
                 "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned int get_member_tid() const noexcept
  {
    return member_tid;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned int get_group_id() const noexcept
  {
    return group_id;
  }

  //---------------------------------------------------------------------
  // Public Keys-Only Sort Interfaces
  //---------------------------------------------------------------------

  /**
   * @brief Sorts a full group tile of keys across threads_per_group threads.
   *
   * @tparam CompareOp Functor type having member `bool operator()(KeyT lhs, KeyT rhs)` (Strict Weak Ordering).
   * @param[in,out] keys Keys array held by this thread.
   * @param[in] compare_op Comparison function object returning true if lhs < rhs.
   * @param[in] threads_per_group Number of cooperating threads in this group (must be a power of two).
   */
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int threads_per_group) noexcept
  {
    static_assert(KEYS_ONLY, "Keys-only Sort requires ValueT == NullType; use the key-value overload instead");
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    ValueT items[ITEMS_PER_THREAD];
    const int tile_size = threads_per_group * ITEMS_PER_THREAD;
    detail::stable_odd_even_sort<Unroll>(keys, items, compare_op);
    merge_rounds<false>(keys, items, compare_op, threads_per_group, tile_size);
  }

  /**
   * @brief Sorts the first valid_items keys across threads_per_group threads without requiring an out-of-bounds
   * sentinel.
   *
   * On output, the first `valid_items` positions of the group tile hold the sorted keys.
   * Tile positions at or after `valid_items` are unspecified.
   *
   * @tparam CompareOp Functor type having member `bool operator()(KeyT lhs, KeyT rhs)` (Strict Weak Ordering).
   * @param[in,out] keys Keys array held by this thread.
   * @param[in] compare_op Comparison function object returning true if lhs < rhs.
   * @param[in] threads_per_group Number of cooperating threads in this group (must be a power of two).
   * @param[in] valid_items Total number of valid keys in this group segment (0 <= valid_items <= threads_per_group *
   * ITEMS_PER_THREAD).
   */
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD], CompareOp compare_op, int threads_per_group, int valid_items) noexcept
  {
    static_assert(KEYS_ONLY, "Keys-only Sort requires ValueT == NullType; use the key-value overload instead");
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(valid_items >= 0, "valid_items must be non-negative");
    ValueT items[ITEMS_PER_THREAD];
    sort_partial_tile<false>(keys, items, compare_op, threads_per_group, valid_items, keys[0]);
  }

  /**
   * @brief Sorts the first valid_items keys across threads_per_group threads using an out-of-bounds default sentinel.
   *
   * On output, the first `valid_items` positions of the group tile hold the sorted keys.
   * Tile positions at or after `valid_items` are unspecified and not guaranteed to hold `oob_default`.
   *
   * @tparam CompareOp Functor type having member `bool operator()(KeyT lhs, KeyT rhs)` (Strict Weak Ordering).
   * @param[in,out] keys Keys array held by this thread.
   * @param[in] compare_op Comparison function object returning true if lhs < rhs.
   * @param[in] threads_per_group Number of cooperating threads in this group (must be a power of two).
   * @param[in] valid_items Total number of valid keys in this group segment.
   * @param[in] oob_default Sentinel value ordered after any valid key in the segment.
   */
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       int threads_per_group,
       int valid_items,
       KeyT oob_default) noexcept
  {
    static_assert(KEYS_ONLY, "Keys-only Sort requires ValueT == NullType; use the key-value overload instead");
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(valid_items >= 0, "valid_items must be non-negative");
    ValueT items[ITEMS_PER_THREAD];
    sort_partial_tile<true>(keys, items, compare_op, threads_per_group, valid_items, oob_default);
  }

  //---------------------------------------------------------------------
  // Public Key-Value Pairs Sort Interfaces
  //---------------------------------------------------------------------

  /**
   * @brief Sorts a full group tile of key-value pairs across threads_per_group threads.
   *
   * @tparam CompareOp Functor type having member `bool operator()(KeyT lhs, KeyT rhs)` (Strict Weak Ordering).
   * @param[in,out] keys Keys array held by this thread.
   * @param[in,out] items Values array held by this thread.
   * @param[in] compare_op Comparison function object returning true if lhs < rhs.
   * @param[in] threads_per_group Number of cooperating threads in this group (must be a power of two).
   */
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],
       ValueT (&items)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       int threads_per_group) noexcept
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    const int tile_size = threads_per_group * ITEMS_PER_THREAD;
    detail::stable_odd_even_sort<Unroll>(keys, items, compare_op);
    merge_rounds<false>(keys, items, compare_op, threads_per_group, tile_size);
  }

  /**
   * @brief Sorts the first valid_items key-value pairs across threads_per_group threads without requiring an
   * out-of-bounds sentinel.
   *
   * On output, the first `valid_items` positions of the group tile hold the sorted key-value pairs.
   * Tile positions at or after `valid_items` are unspecified.
   *
   * @tparam CompareOp Functor type having member `bool operator()(KeyT lhs, KeyT rhs)` (Strict Weak Ordering).
   * @param[in,out] keys Keys array held by this thread.
   * @param[in,out] items Values array held by this thread.
   * @param[in] compare_op Comparison function object returning true if lhs < rhs.
   * @param[in] threads_per_group Number of cooperating threads in this group (must be a power of two).
   * @param[in] valid_items Total number of valid items in this group segment.
   */
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],
       ValueT (&items)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       int threads_per_group,
       int valid_items) noexcept
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(valid_items >= 0, "valid_items must be non-negative");
    sort_partial_tile<false>(keys, items, compare_op, threads_per_group, valid_items, keys[0]);
  }

  /**
   * @brief Sorts the first valid_items key-value pairs across threads_per_group threads using an out-of-bounds
   * sentinel.
   *
   * On output, the first `valid_items` positions of the group tile hold the sorted key-value pairs.
   * Tile positions at or after `valid_items` are unspecified and not guaranteed to hold `oob_default` or default
   * values.
   *
   * @tparam CompareOp Functor type having member `bool operator()(KeyT lhs, KeyT rhs)` (Strict Weak Ordering).
   * @param[in,out] keys Keys array held by this thread.
   * @param[in,out] items Values array held by this thread.
   * @param[in] compare_op Comparison function object returning true if lhs < rhs.
   * @param[in] threads_per_group Number of cooperating threads in this group (must be a power of two).
   * @param[in] valid_items Total number of valid items in this group segment.
   * @param[in] oob_default Sentinel value ordered after any valid key in the segment.
   */
  template <typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],
       ValueT (&items)[ITEMS_PER_THREAD],
       CompareOp compare_op,
       int threads_per_group,
       int valid_items,
       KeyT oob_default) noexcept
  {
    _CCCL_ASSERT(threads_per_group > 0, "threads_per_group must be greater than 0");
    _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_group), "threads_per_group must be a power of two");
    _CCCL_ASSERT(valid_items >= 0, "valid_items must be non-negative");
    sort_partial_tile<true>(keys, items, compare_op, threads_per_group, valid_items, oob_default);
  }

  //---------------------------------------------------------------------
  // StableSort Interfaces (synonyms for Sort since merge sort is stable)
  //---------------------------------------------------------------------

  template <typename... Args>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void StableSort(Args&&... args) noexcept
  {
    Sort(::cuda::std::forward<Args>(args)...);
  }

private:
  /**
   * @brief Synchronizes threads strictly within this cooperating group without affecting other groups in the CTA.
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void sync(int threads_per_group) const noexcept
  {
    if (threads_per_group <= 1)
    {
      return;
    }
    else if (threads_per_group <= 32)
    {
      const unsigned int lane_id    = ::cuda::ptx::get_sreg_laneid();
      const unsigned int subwarp_id = lane_id / static_cast<unsigned int>(threads_per_group);
      const unsigned int mask =
        (threads_per_group >= 32)
          ? 0xFFFFFFFFu
          : ((0xFFFFFFFFu >> (32 - threads_per_group)) << (subwarp_id * threads_per_group));
      __syncwarp(mask);
    }
    else
    {
      // Named barrier 0 is reserved for __syncthreads(); use ids 1..15.
      // Distinct groups must map to distinct ids, so a CTA may host at most 15 multi-warp groups.
      _CCCL_ASSERT(group_id < 15u, "GroupMergeSort supports at most 15 multi-warp groups per CTA");
      const unsigned int barrier_id = 1u + group_id;
      NV_IF_TARGET(NV_PROVIDES_SM_50,
                   (asm volatile("bar.sync %0, %1;" : : "r"(barrier_id),
                                 "r"(static_cast<unsigned int>(threads_per_group)) : "memory");),
                   (_CCCL_ASSERT(false, "Multi-warp GroupMergeSort requires SM 50+ hardware named barrier support");));
    }
  }

  /**
   * @brief Stores keys to shared memory and sets the single prefetch padding slot at threads_per_group *
   * ITEMS_PER_THREAD.
   *
   * The serial merge prefetches (but never uses) one element past the end of a run, and the run ending at
   * threads_per_group * ITEMS_PER_THREAD would otherwise read an uninitialized value (see NVIDIA/cccl#5327).
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void store_keys(KeyT (&keys)[ITEMS_PER_THREAD], int threads_per_group) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      const int idx    = ITEMS_PER_THREAD * member_tid + item;
      keys_shared[idx] = keys[item];
    }
    if (member_tid == 0)
    {
      keys_shared[threads_per_group * ITEMS_PER_THREAD] = keys[0];
    }
  }

  /**
   * @brief Exchanges items according to indices recorded during the serial key merge.
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  exchange_items(ValueT (&items)[ITEMS_PER_THREAD], int (&indices)[ITEMS_PER_THREAD], int threads_per_group) noexcept
  {
    sync(threads_per_group);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      const int idx     = ITEMS_PER_THREAD * member_tid + item;
      items_shared[idx] = items[item];
    }

    sync(threads_per_group);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      items[item] = items_shared[indices[item]];
    }
  }

  /**
   * @brief Sorts the first valid_items elements of the group tile, clamping every merge run to that boundary.
   */
  template <bool HasOobDefault, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void sort_partial_tile(
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&items)[ITEMS_PER_THREAD],
    CompareOp compare_op,
    int threads_per_group,
    int valid_items,
    const KeyT& oob_default) noexcept
  {
    if (static_cast<int>(ITEMS_PER_THREAD * member_tid) < valid_items)
    {
      KeyT max_key = keys[0];
      if constexpr (HasOobDefault)
      {
        max_key = compare_op(max_key, oob_default) ? oob_default : max_key;
      }

      _CCCL_PRAGMA_UNROLL(Unroll ? ITEMS_PER_THREAD : 1)
      for (int item = 1; item < ITEMS_PER_THREAD; ++item)
      {
        if (static_cast<int>(ITEMS_PER_THREAD * member_tid + item) < valid_items)
        {
          max_key = compare_op(max_key, keys[item]) ? keys[item] : max_key;
        }
        else
        {
          keys[item] = max_key;
        }
      }

      detail::stable_odd_even_sort<Unroll>(keys, items, compare_op);
    }

    merge_rounds<true>(keys, items, compare_op, threads_per_group, valid_items);
  }

  /**
   * @brief Runs the log2(threads_per_group) merge rounds.
   *
   * When Clamped is false: full tile is merged with exact boundaries.
   * When Clamped is true: every run is clamped to valid_items and trailing redundant rounds are skipped.
   */
  template <bool Clamped, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void merge_rounds(
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&items)[ITEMS_PER_THREAD],
    CompareOp compare_op,
    int threads_per_group,
    int valid_items) noexcept
  {
    for (int target_merged_threads_number = 2; target_merged_threads_number <= threads_per_group;
         target_merged_threads_number *= 2)
    {
      const int merged_threads_number = target_merged_threads_number / 2;
      const int mask                  = target_merged_threads_number - 1;
      const int size                  = ITEMS_PER_THREAD * merged_threads_number;

      if constexpr (Clamped)
      {
        // The group tile's first `size` positions already form a single sorted run covering all valid items.
        // valid_items is group-uniform, so this break is uniform and barrier-safe within the group.
        if (size >= valid_items)
        {
          break;
        }
      }

      sync(threads_per_group);
      store_keys(keys, threads_per_group);
      sync(threads_per_group);

      int indices[ITEMS_PER_THREAD];

      const int first_thread_idx_in_thread_group_being_merged = ~mask & member_tid;
      const int start = ITEMS_PER_THREAD * first_thread_idx_in_thread_group_being_merged;

      const int thread_idx_in_thread_group_being_merged = mask & member_tid;

      int keys1_beg, keys1_end, keys2_end, diag;
      if constexpr (Clamped)
      {
        keys1_beg = (::cuda::std::min) (valid_items, start);
        keys1_end = (::cuda::std::min) (valid_items, keys1_beg + size);
        keys2_end = (::cuda::std::min) (valid_items, keys1_end + size);
        diag = (::cuda::std::min) (keys2_end - keys1_beg,
                                   (::cuda::std::min) (valid_items,
                                                       ITEMS_PER_THREAD * thread_idx_in_thread_group_being_merged));
      }
      else
      {
        keys1_beg = start;
        keys1_end = start + size;
        keys2_end = keys1_end + size;
        diag      = ITEMS_PER_THREAD * thread_idx_in_thread_group_being_merged;
      }
      const int keys2_beg   = keys1_end;
      const int keys1_count = keys1_end - keys1_beg;
      const int keys2_count = keys2_end - keys2_beg;

      const int partition_diag =
        MergePath(&keys_shared[keys1_beg], &keys_shared[keys2_beg], keys1_count, keys2_count, diag, compare_op);

      const int keys1_beg_loc   = keys1_beg + partition_diag;
      const int keys2_beg_loc   = keys2_beg + diag - partition_diag;
      const int keys1_count_loc = keys1_end - keys1_beg_loc;
      const int keys2_count_loc = keys2_end - keys2_beg_loc;

      detail::serial_merge<Unroll>(
        keys_shared, keys1_beg_loc, keys2_beg_loc, keys1_count_loc, keys2_count_loc, keys, indices, compare_op);

      if constexpr (!KEYS_ONLY)
      {
        exchange_items(items, indices, threads_per_group);
      }
    }
  }
};
} // namespace group_merge_sort

using group_merge_sort::GroupMergeSort;

//---------------------------------------------------------------------
// Dispatch Helpers: call_group_merge_runtime
//---------------------------------------------------------------------

/**
 * @brief Dispatch helper for runtime-width group merge-sort on keys-only without sentinel using TempStorage.
 *
 * On output, the first `valid_items` positions hold the sorted keys.
 * Tile positions at or after `valid_items` are unspecified.
 *
 * @param[in,out] temp_storage Reference to this group's temporary storage allocation.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid keys in this segment.
 * @param[in] linear_tid Rank of calling thread within the CTA (e.g. threadIdx.x).
 */
template <typename KeyT, int ITEMS_PER_THREAD, int MAX_GROUP_THREADS = 1024, typename CompareOp = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  typename GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, NullType>::TempStorage& temp_storage,
  KeyT (&keys)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  unsigned int linear_tid) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(threads_per_segment <= MAX_GROUP_THREADS,
               "threads_per_segment must not exceed MAX_GROUP_THREADS for TempStorage allocation");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32
                 || (threads_per_segment > 0 && (linear_tid / static_cast<unsigned int>(threads_per_segment)) < 15u),
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, NullType>(temp_storage, linear_tid, threads_per_segment)
    .Sort(keys, compare_op, threads_per_segment, valid_items);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on keys-only with sentinel using TempStorage.
 *
 * On output, the first `valid_items` positions hold the sorted keys.
 * Tile positions at or after `valid_items` are unspecified and not guaranteed to hold `oob_default`.
 *
 * @param[in,out] temp_storage Reference to this group's temporary storage allocation.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid keys in this segment.
 * @param[in] oob_default Sentinel value ordered after any valid key in the segment.
 * @param[in] linear_tid Rank of calling thread within the CTA (e.g. threadIdx.x).
 */
template <typename KeyT, int ITEMS_PER_THREAD, int MAX_GROUP_THREADS = 1024, typename CompareOp = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  typename GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, NullType>::TempStorage& temp_storage,
  KeyT (&keys)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  KeyT oob_default,
  unsigned int linear_tid) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(threads_per_segment <= MAX_GROUP_THREADS,
               "threads_per_segment must not exceed MAX_GROUP_THREADS for TempStorage allocation");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32
                 || (threads_per_segment > 0 && (linear_tid / static_cast<unsigned int>(threads_per_segment)) < 15u),
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, NullType>(temp_storage, linear_tid, threads_per_segment)
    .Sort(keys, compare_op, threads_per_segment, valid_items, oob_default);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on keys-only without sentinel using raw shared memory
 * pointers.
 *
 * On output, the first `valid_items` positions hold the sorted keys.
 * Tile positions at or after `valid_items` are unspecified.
 *
 * @param[in] keys_shared Pointer to the group's slice of shared memory for keys.
 *            Must provide at least `threads_per_segment * ITEMS_PER_THREAD + 1` elements;
 *            the trailing element is the serial-merge prefetch padding slot.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid keys in this segment.
 * @param[in] member_tid Rank of calling thread within its cooperating group.
 * @param[in] group_id Rank of the group within the CTA (0 <= group_id < 15 for multi-warp groups).
 */
template <typename KeyT, int ITEMS_PER_THREAD, int MAX_GROUP_THREADS = 1024, typename CompareOp = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  KeyT* keys_shared,
  KeyT (&keys)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  unsigned int member_tid,
  unsigned int group_id) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32 || group_id < 15u,
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, NullType>(
    keys_shared, static_cast<NullType*>(nullptr), member_tid, group_id, threads_per_segment)
    .Sort(keys, compare_op, threads_per_segment, valid_items);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on keys-only with sentinel using raw shared memory
 * pointers.
 *
 * On output, the first `valid_items` positions hold the sorted keys.
 * Tile positions at or after `valid_items` are unspecified and not guaranteed to hold `oob_default`.
 *
 * @param[in] keys_shared Pointer to the group's slice of shared memory for keys.
 *            Must provide at least `threads_per_segment * ITEMS_PER_THREAD + 1` elements;
 *            the trailing element is the serial-merge prefetch padding slot.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid keys in this segment.
 * @param[in] oob_default Sentinel value ordered after any valid key in the segment.
 * @param[in] member_tid Rank of calling thread within its cooperating group.
 * @param[in] group_id Rank of the group within the CTA (0 <= group_id < 15 for multi-warp groups).
 */
template <typename KeyT, int ITEMS_PER_THREAD, int MAX_GROUP_THREADS = 1024, typename CompareOp = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  KeyT* keys_shared,
  KeyT (&keys)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  KeyT oob_default,
  unsigned int member_tid,
  unsigned int group_id) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32 || group_id < 15u,
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, NullType>(
    keys_shared, static_cast<NullType*>(nullptr), member_tid, group_id, threads_per_segment)
    .Sort(keys, compare_op, threads_per_segment, valid_items, oob_default);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on key-value pairs without sentinel using TempStorage.
 *
 * On output, the first `valid_items` positions hold the sorted key-value pairs.
 * Tile positions at or after `valid_items` are unspecified.
 *
 * @param[in,out] temp_storage Reference to this group's temporary storage allocation.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in,out] items Values array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid items in this segment.
 * @param[in] linear_tid Rank of calling thread within the CTA (e.g. threadIdx.x).
 */
template <typename KeyT,
          int ITEMS_PER_THREAD,
          int MAX_GROUP_THREADS = 1024,
          typename ValueT       = NullType,
          typename CompareOp    = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  typename GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>::TempStorage& temp_storage,
  KeyT (&keys)[ITEMS_PER_THREAD],
  ValueT (&items)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  unsigned int linear_tid) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(threads_per_segment <= MAX_GROUP_THREADS,
               "threads_per_segment must not exceed MAX_GROUP_THREADS for TempStorage allocation");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32
                 || (threads_per_segment > 0 && (linear_tid / static_cast<unsigned int>(threads_per_segment)) < 15u),
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>(temp_storage, linear_tid, threads_per_segment)
    .Sort(keys, items, compare_op, threads_per_segment, valid_items);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on key-value pairs with sentinel using TempStorage.
 *
 * On output, the first `valid_items` positions hold the sorted key-value pairs.
 * Tile positions at or after `valid_items` are unspecified and not guaranteed to hold `oob_default` or default values.
 *
 * @param[in,out] temp_storage Reference to this group's temporary storage allocation.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in,out] items Values array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid items in this segment.
 * @param[in] oob_default Sentinel value ordered after any valid key in the segment.
 * @param[in] linear_tid Rank of calling thread within the CTA (e.g. threadIdx.x).
 */
template <typename KeyT,
          int ITEMS_PER_THREAD,
          int MAX_GROUP_THREADS = 1024,
          typename ValueT       = NullType,
          typename CompareOp    = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  typename GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>::TempStorage& temp_storage,
  KeyT (&keys)[ITEMS_PER_THREAD],
  ValueT (&items)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  KeyT oob_default,
  unsigned int linear_tid) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(threads_per_segment <= MAX_GROUP_THREADS,
               "threads_per_segment must not exceed MAX_GROUP_THREADS for TempStorage allocation");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32
                 || (threads_per_segment > 0 && (linear_tid / static_cast<unsigned int>(threads_per_segment)) < 15u),
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>(temp_storage, linear_tid, threads_per_segment)
    .Sort(keys, items, compare_op, threads_per_segment, valid_items, oob_default);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on key-value pairs without sentinel using raw shared memory
 * pointers.
 *
 * On output, the first `valid_items` positions hold the sorted key-value pairs.
 * Tile positions at or after `valid_items` are unspecified.
 *
 * @param[in] keys_shared Pointer to the group's slice of shared memory for keys.
 *            Must provide at least `threads_per_segment * ITEMS_PER_THREAD + 1` elements;
 *            the trailing element is the serial-merge prefetch padding slot.
 * @param[in] items_shared Pointer to the group's slice of shared memory for values.
 *            Must provide at least `threads_per_segment * ITEMS_PER_THREAD + 1` elements.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in,out] items Values array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid items in this segment.
 * @param[in] member_tid Rank of calling thread within its cooperating group.
 * @param[in] group_id Rank of the group within the CTA (0 <= group_id < 15 for multi-warp groups).
 */
template <typename KeyT,
          int ITEMS_PER_THREAD,
          int MAX_GROUP_THREADS = 1024,
          typename ValueT       = NullType,
          typename CompareOp    = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  KeyT* keys_shared,
  ValueT* items_shared,
  KeyT (&keys)[ITEMS_PER_THREAD],
  ValueT (&items)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  unsigned int member_tid,
  unsigned int group_id) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32 || group_id < 15u,
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>(
    keys_shared, items_shared, member_tid, group_id, threads_per_segment)
    .Sort(keys, items, compare_op, threads_per_segment, valid_items);
}

/**
 * @brief Dispatch helper for runtime-width group merge-sort on key-value pairs with sentinel using raw shared memory
 * pointers.
 *
 * On output, the first `valid_items` positions hold the sorted key-value pairs.
 * Tile positions at or after `valid_items` are unspecified and not guaranteed to hold `oob_default` or default values.
 *
 * @param[in] keys_shared Pointer to the group's slice of shared memory for keys.
 *            Must provide at least `threads_per_segment * ITEMS_PER_THREAD + 1` elements;
 *            the trailing element is the serial-merge prefetch padding slot.
 * @param[in] items_shared Pointer to the group's slice of shared memory for values.
 *            Must provide at least `threads_per_segment * ITEMS_PER_THREAD + 1` elements.
 * @param[in,out] keys Keys array held by this thread.
 * @param[in,out] items Values array held by this thread.
 * @param[in] compare_op Comparison function object returning true if lhs < rhs.
 * @param[in] threads_per_segment Number of cooperating threads in this segment (must be a power of two).
 * @param[in] valid_items Total number of valid items in this segment.
 * @param[in] oob_default Sentinel value ordered after any valid key in the segment.
 * @param[in] member_tid Rank of calling thread within its cooperating group.
 * @param[in] group_id Rank of the group within the CTA (0 <= group_id < 15 for multi-warp groups).
 */
template <typename KeyT,
          int ITEMS_PER_THREAD,
          int MAX_GROUP_THREADS = 1024,
          typename ValueT       = NullType,
          typename CompareOp    = ::cuda::std::less<KeyT>>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void call_group_merge_runtime(
  KeyT* keys_shared,
  ValueT* items_shared,
  KeyT (&keys)[ITEMS_PER_THREAD],
  ValueT (&items)[ITEMS_PER_THREAD],
  CompareOp compare_op,
  int threads_per_segment,
  int valid_items,
  KeyT oob_default,
  unsigned int member_tid,
  unsigned int group_id) noexcept
{
  _CCCL_ASSERT(threads_per_segment > 0, "threads_per_segment must be greater than 0");
  _CCCL_ASSERT(::cuda::is_power_of_two(threads_per_segment), "threads_per_segment must be a power of two");
  _CCCL_ASSERT(threads_per_segment <= 32 || group_id < 15u,
               "GroupMergeSort supports at most 15 multi-warp groups per CTA");
  GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>(
    keys_shared, items_shared, member_tid, group_id, threads_per_segment)
    .Sort(keys, items, compare_op, threads_per_segment, valid_items, oob_default);
}
} // namespace detail

CUB_NAMESPACE_END
