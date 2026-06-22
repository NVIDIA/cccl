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

#include <cub/block/specializations/block_topk_rank_atomic.cuh>
#include <cub/block/specializations/block_topk_sieve_air.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @brief Per-thread bundle of per-item selection states for top-k iterative
//!        refinement. Opaque, k-locked, free-standing template with no public
//!        constructors.
//!
//! Instances of this class are obtained exclusively via
//! `block_topk_sieve::select_max` / `select_min`, which build the state and
//! run the first radix sweep in one call. Subsequent `block_topk_sieve::refine_*`
//! calls mutate the state in place; `block_topk_rank::rank_key_states`
//! consumes it. Callers observe the state through the const predicate
//! `has_ties()`.
//!
//! Each item held by the thread is in exactly one of three internal states:
//!
//!   - `candidate`: refinement has not yet ruled the item in or out of the
//!                  top-k.
//!   - `selected`:  the item is guaranteed to be in the top-k.
//!   - `rejected`:  the item is guaranteed NOT to be in the top-k. Also used
//!                  to mark invalid / padding items prior to the first refine
//!                  step.
//!
//! The class also caches the block-wide counts of items currently in the
//! `candidate` and `selected` states so the next refine call can short-circuit
//! when no candidates remain and compute the effective k internally without a
//! re-reduction.
//!
//! The internal representation is intentionally hidden so it can be replaced
//! with a packed (e.g. 2-bits-per-item) layout without breaking the API.
//!
//! Decoupling from `KeyT` lets the same state instance flow through refines on
//! different `block_topk_sieve` instantiations.
//!
//! @tparam ItemsPerThread Number of items each thread holds.
template <int ItemsPerThread>
class block_topk_key_states
{
private:
  enum class class_value
  {
    candidate = 0,
    selected  = 1,
    rejected  = 2,
    // Only needed for full ranking/permutation including not only rejected but also invalid items (they would need to
    // be ordered after even the rejected items).
    invalid = 3,
  };

  class_value values_[ItemsPerThread];
  int k_;
  int num_candidates_;
  int num_selected_;
  // Only needed for full ranking/permutation including not only rejected but also invalid items (they would need to be
  // ordered after even the rejected items).
  int num_valid_;

  // Builder is private. The state is built only by
  // `block_topk_sieve::select_*`, mutated by `block_topk_sieve::refine_*` and
  // `block_topk_rank::rank_key_states`. Each of those goes through the
  // friended specializations.
  block_topk_key_states() = default;

  //! @tparam BlockDimX     Number of threads in the (1D) block.
  template <bool IsFullTile, int BlockDimX, bool IsBlockedInput = true>
  static _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_key_states build(int k, int valid_items) noexcept
  {
    [[maybe_unused]] constexpr int tile_items = BlockDimX * ItemsPerThread;
    // Preconditions
    _CCCL_ASSERT(k > 0 && k <= tile_items, "k must be in (0, tile_items]");
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items >= 0 && valid_items <= tile_items,
                   "valid_items must be in [0, BlockDimX * ItemsPerThread]");
      _CCCL_ASSERT(k <= valid_items, "k must be <= valid_items");
    }
    block_topk_key_states states{};
    states.k_                             = k;
    states.num_candidates_                = IsFullTile ? tile_items : valid_items;
    states.num_selected_                  = 0;
    states.num_valid_                     = IsFullTile ? tile_items : valid_items;
    [[maybe_unused]] const int linear_tid = RowMajorTid(BlockDimX, 1, 1);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      [[maybe_unused]] const int idx = IsBlockedInput ? linear_tid * ItemsPerThread + i : i * BlockDimX + linear_tid;
      if constexpr (IsFullTile)
      {
        states.values_[i] = class_value::candidate;
      }
      else
      {
        states.values_[i] = idx < valid_items ? class_value::candidate : class_value::invalid;
      }
    }
    return states;
  }

  // --- Storage-abstracting accessors / mutators ---
  //
  // Friends below never touch the storage directly; all reads and writes go
  // through these. This lets the internal representation (currently a flat
  // `class_value[ItemsPerThread]` plus a few `int` counters) change without
  // touching the sieve / rank specializations.

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int k() const noexcept
  {
    return k_;
  }
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int num_selected() const noexcept
  {
    return num_selected_;
  }
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int num_candidates() const noexcept
  {
    return num_candidates_;
  }
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int num_valid() const noexcept
  {
    return num_valid_;
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void set_num_selected(int new_num_selected) noexcept
  {
    _CCCL_ASSERT(new_num_selected <= num_valid_, "set_num_selected: new value must not exceed num_valid");
    _CCCL_ASSERT(new_num_selected >= num_selected_, "set_num_selected: num_selected must monotonically grow");
    num_selected_ = new_num_selected;
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void set_num_candidates(int new_num_candidates) noexcept
  {
    _CCCL_ASSERT(new_num_candidates >= 0, "set_num_candidates: new value must be non-negative");
    _CCCL_ASSERT(new_num_candidates <= num_candidates_, "set_num_candidates: num_candidates must monotonically shrink");
    num_candidates_ = new_num_candidates;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool is_candidate(int i) const noexcept
  {
    _CCCL_ASSERT(0 <= i && i < ItemsPerThread, "item index out of range");
    return values_[i] == class_value::candidate;
  }
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool is_selected(int i) const noexcept
  {
    _CCCL_ASSERT(0 <= i && i < ItemsPerThread, "item index out of range");
    return values_[i] == class_value::selected;
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void set_selected(int i) noexcept
  {
    _CCCL_ASSERT(values_[i] == class_value::candidate || values_[i] == class_value::selected,
                 "set_selected requires the item to be a candidate or already selected");
    values_[i] = class_value::selected;
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void set_rejected(int i) noexcept
  {
    _CCCL_ASSERT(values_[i] == class_value::candidate || values_[i] == class_value::rejected,
                 "set_rejected requires the item to be a candidate or already rejected");
    values_[i] = class_value::rejected;
  }

  template <typename, int>
  friend class block_topk_sieve;
  template <typename, int, int>
  friend class block_topk_sieve_air;
  template <int>
  friend class block_topk_rank;
  template <int>
  friend class block_topk_rank_atomic;

public:
  //! `true` iff at least one item is still `candidate` and the candidate
  //! count exceeds the slots remaining to fill the top-k. Designed to be
  //! used directly as an `if` / `while` predicate around subsequent
  //! `block_topk_sieve::refine_*` calls. Uniform across the block.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool has_ties() const noexcept
  {
    _CCCL_ASSERT(k_ >= 0, "negative k would report ties incorrectly");
    return num_selected_ + num_candidates_ > k_;
  }
};

//! @brief Block-level radix sieve that builds and updates per-item top-k
//!        states without moving keys/values around.
//!
//! Instead of scattering selected items into shared memory, it writes
//! per-item classifications back into a `block_topk_key_states` object.
//! Suitable for iterative / multi-key refinement before a final tie-break +
//! scatter via `block_topk_rank`.
//!
//! Two pairs of member functions are provided:
//!
//!   - `select_max` / `select_min` build a fresh `block_topk_key_states`
//!     from a tile of keys and the target `k`, run the first radix-pass
//!     sweep, and return the state. They carry an explicit @c IsFullTile
//!     and a defaulted `BlockedInput = true` member-function template
//!     parameter; this is the only step that benefits from those compile-
//!     time information (the per-thread "is this index < num_valid?" check folds
//!     into the rejection-mask seeding *and* into the radix histogram pass).
//!   - `refine_max` / `refine_min` mutate an existing state in place`.
//!
//! @tparam KeyT     Key type for this sieve. May differ between sieves
//!                  refining the same `block_topk_key_states`.
//! @tparam BlockDimX Number of threads in the (1D) block.
template <typename KeyT, int BlockDimX>
class block_topk_sieve
{
private:
  static constexpr int threads_per_block = BlockDimX;

  using algorithm_t  = block_topk_sieve_air<KeyT, BlockDimX>;
  using TempStorage_ = typename algorithm_t::TempStorage;

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

private:
  TempStorage_& storage_;

public:
  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit block_topk_sieve(TempStorage& storage)
      : storage_(storage.Alias())
  {}

private:
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void check_preconditions(int begin_bit, int end_bit) const noexcept
  {
    [[maybe_unused]] constexpr int max_bit = int(sizeof(KeyT) * 8);
    _CCCL_ASSERT(begin_bit >= 0 && begin_bit < max_bit, "begin_bit must be in [0, max_bit)");
    _CCCL_ASSERT(end_bit > begin_bit && end_bit <= max_bit, "end_bit must be in (begin_bit, max_bit]");
  }

public:
  //! Build a fresh `block_topk_key_states` selecting the largest @p k items
  //! and run the first radix-pass sweep on @p keys. The returned state optionally
  //! flows into subsequent `refine_*` calls and finally into
  //! `block_topk_rank::rank_key_states`.
  //!
  //! Items whose key bits in `[begin_bit, end_bit)` are strictly larger than
  //! the kth-key bits become `selected`; strictly smaller become `rejected`;
  //! items tied with the kth key remain `candidate`. Items in
  //! `[num_valid, BlockDimX * ItemsPerThread)` of the *logical* tile are
  //! seeded directly as `rejected`.
  //!
  //! @tparam IsFullTile    If true, the caller guarantees @p num_valid ==
  //!                       `BlockDimX * ItemsPerThread`. The implementation
  //!                       skips per-item `idx < num_valid` checks both in
  //!                       rejection-mask seeding and in the radix histogram
  //!                       pass. When true, @p num_valid is unused.
  //! @tparam BlockedInput  If true (the default), @p keys is in blocked
  //!                       arrangement (thread `linear_tid` owns global
  //!                       indices `[linear_tid * IPT, (linear_tid + 1) * IPT)`);
  //!                       if false, striped (`linear_tid + i * BlockDimX`).
  //!                       Has no effect when @c IsFullTile is true.
  //! @tparam ItemsPerThread Number of items per thread; deduced from @p keys.
  //!
  //! @param[in] keys      Tile of keys held by this thread.
  //! @param[in] k         Target number of top items in the block. Must be
  //!                      in `[1, BlockDimX * ItemsPerThread]`. Locked into
  //!                      the returned state.
  //! @param[in] num_valid Number of valid items in the tile, in
  //!                      `[0, BlockDimX * ItemsPerThread]`. Ignored when
  //!                      @c IsFullTile is true.
  //! @param[in] begin_bit Inclusive lower bit of the bit window to refine on
  //!                      (default `0`).
  //! @param[in] end_bit   Exclusive upper bit of the bit window to refine on
  //!                      (default `sizeof(KeyT) * 8`).
  //!
  //! @return A `block_topk_key_states<ItemsPerThread>` reflecting the
  //!         result of the first radix sweep. Use `has_ties()` to decide
  //!         whether to run another refine.
  template <bool IsFullTile, bool BlockedInput = true, int ItemsPerThread>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_key_states<ItemsPerThread>
  select_max(KeyT (&keys)[ItemsPerThread], int k, int valid_items, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    check_preconditions(begin_bit, end_bit);
    auto states =
      block_topk_key_states<ItemsPerThread>::template build<IsFullTile, BlockDimX, BlockedInput>(k, valid_items);
    if (states.has_ties())
    {
      algorithm_t(storage_).template refine_keys<detail::topk::select::max, IsFullTile>(
        keys, states, begin_bit, end_bit);
    }
    return states;
  }

  //! Same as `select_max` but selects the smallest @p k items.
  template <bool IsFullTile, bool BlockedInput = true, int ItemsPerThread>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_key_states<ItemsPerThread>
  select_min(KeyT (&keys)[ItemsPerThread], int k, int valid_items, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    check_preconditions(begin_bit, end_bit);
    auto states =
      block_topk_key_states<ItemsPerThread>::template build<IsFullTile, BlockDimX, BlockedInput>(k, valid_items);
    if (states.has_ties())
    {
      algorithm_t(storage_).template refine_keys<detail::topk::select::min, IsFullTile, ItemsPerThread>(
        keys, states, begin_bit, end_bit);
    }
    return states;
  }

  //! Refine @p states for the largest k items (k locked at `select_*` time).
  //!
  //! Examines bit range `[begin_bit, end_bit)` of each candidate's key in
  //! @p keys; items whose bits in that range are strictly larger than the
  //! kth-key bits become `selected`, strictly smaller become `rejected`,
  //! ties remain `candidate`. Items already `selected` / `rejected` on entry
  //! are unchanged.
  //!
  //! May be called multiple times with disjoint bit ranges to progressively
  //! narrow the candidate set. The same @p states may be passed to sieves
  //! with different `KeyT` (provided `BlockDimX` matches and the per-call
  //! `ItemsPerThread` deduction agrees).
  //!
  //! When `states.has_ties()` returns false on entry, this function performs
  //! no shared-memory work and no `__syncthreads()`. Callers are still
  //! encouraged to guard refines with `if (states.has_ties())` so they also
  //! avoid loading / decoding the input keys.
  //!
  //! Internally the effective k for this pass is computed from the cached
  //! state counts. On exit those counts (and therefore `has_ties()`) reflect
  //! the new classification.
  //!
  //! @tparam ItemsPerThread Number of items per thread; deduced from @p keys
  //!                       / @p states.
  template <int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void refine_max(
    KeyT (&keys)[ItemsPerThread],
    block_topk_key_states<ItemsPerThread>& states,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    check_preconditions(begin_bit, end_bit);
    if (states.has_ties())
    {
      algorithm_t(storage_).template refine_keys<detail::topk::select::max, false>(keys, states, begin_bit, end_bit);
    }
  }

  //! Same as `refine_max` but for a sieve that started with `select_min`.
  template <int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void refine_min(
    KeyT (&keys)[ItemsPerThread],
    block_topk_key_states<ItemsPerThread>& states,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    check_preconditions(begin_bit, end_bit);
    if (states.has_ties())
    {
      algorithm_t(storage_).template refine_keys<detail::topk::select::min, false>(keys, states, begin_bit, end_bit);
    }
  }
};

//! @rst
//! Block-level finalization: tie-breaks the candidates in a
//! ``block_topk_key_states`` and emits per-item scatter ranks usable with
//! ``cub::BlockExchange``.
//!
//! ``KeyT``-independent and ``ItemsPerThread``-independent at the class
//! level: a single ``block_topk_rank`` instantiation can finalize a state
//! object that was refined by any combination of
//! ``block_topk_sieve<K, BlockDimX>`` instantiations (provided ``BlockDimX``
//! matches).
//!
//! A Simple Example
//! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//!
//! The example below selects the largest ``k`` items by a primary float key,
//! breaks ties using a secondary int key (lazily loaded only when ties
//! actually remain), and then moves the selected items to the front of the
//! block in striped arrangement using ``BlockExchange::ScatterToStripedGuarded``.
//! The single ``__shared__`` union covers the full pipeline (sieves, rank,
//! and exchange), separated by ``__syncthreads()`` between phases.
//!
//! .. code-block:: c++
//!
//!    #include <cub/block/block_exchange.cuh>
//!    #include <cub/block/block_topk_rank.cuh>
//!
//!    constexpr int ThreadsPerBlock   = 128;
//!    constexpr int ItemsPerThread = 4;
//!    // Assume the caller knows it is operating on a full tile.
//!    constexpr bool IsFullTile    = true;
//!
//!    using PrimarySieve   = cub::detail::block_topk_sieve<float, ThreadsPerBlock>;
//!    using SecondarySieve = cub::detail::block_topk_sieve<int,   ThreadsPerBlock>;
//!    using TopKRank       = cub::detail::block_topk_rank<ThreadsPerBlock>;
//!    using Exchange       = cub::BlockExchange<float, ThreadsPerBlock, ItemsPerThread>;
//!
//!    __global__ void example_kernel(float* in, float* out, int k)
//!    {
//!      // Each phase (sieves / rank / exchange) is separated from the next
//!      // by a __syncthreads(), so they share __shared__ bytes via a union.
//!      // The block_topk_key_states itself is a per-thread object and does
//!      // not (and cannot) live in __shared__.
//!      __shared__ union {
//!        typename PrimarySieve::TempStorage   primary_sieve;
//!        typename SecondarySieve::TempStorage secondary_sieve;
//!        typename TopKRank::TempStorage       topk_rank;
//!        typename Exchange::TempStorage       exchange;
//!      } smem;
//!
//!      float primary_keys[ItemsPerThread];
//!      // ... load primary_keys (in blocked arrangement here) ...
//!
//!      // 1) Build the state object and run the first radix sweep on the
//!      //    primary key. The state's k is locked here. For striped inputs,
//!      //    pass `BlockedInput = false` as the second template argument.
//!      auto states = PrimarySieve(smem.primary_sieve)
//!                      .template select_max<IsFullTile>(primary_keys, k, ThreadsPerBlock * ItemsPerThread);
//!      __syncthreads();
//!
//!      // 2) Break ties using the secondary key (different KeyT, same
//!      //    BlockDimX and ItemsPerThread). Guarding on `has_ties()` lets
//!      //    us skip loading the secondary key entirely when no ties
//!      //    remained.
//!      if (states.has_ties())
//!      {
//!        int secondary_keys[ItemsPerThread];
//!        // ... load secondary_keys ...
//!        SecondarySieve(smem.secondary_sieve).refine_max(secondary_keys, states);
//!        __syncthreads();
//!      }
//!
//!      // 3) Final tie-break + scatter ranks for BlockExchange.
//!      int ranks[ItemsPerThread];
//!      TopKRank(smem.topk_rank).rank_key_states(states, ranks);
//!      __syncthreads();
//!
//!      // 4) Move the top-k items to the front of the logical tile. Non-top-k
//!      //    items have rank == -1, which `ScatterToStripedGuarded`
//!      //    interprets internally - no separate flags array is needed.
//!      float out_keys[ItemsPerThread];
//!      Exchange(smem.exchange).ScatterToStripedGuarded(primary_keys, out_keys, ranks);
//!
//!      // ... store the first k items of `out_keys` to `out` ...
//!    }
//!
//! A while-loop variant (e.g. refining one byte at a time, MSB to LSB) is
//! equally idiomatic. The first byte goes through ``select_max`` to build
//! the state; subsequent bytes go through ``refine_max`` under a
//! ``has_ties()`` predicate.
//!
//! .. code-block:: c++
//!
//!    auto sieve = PrimarySieve(smem.primary_sieve);
//!
//!    int hi      = sizeof(float) * 8;
//!    auto states = sieve.template select_max<IsFullTile>(primary_keys, k, num_valid, hi - 8, hi);
//!    hi -= 8;
//!
//!    while (hi > 0 && states.has_ties())
//!    {
//!      __syncthreads();
//!      sieve.refine_max(primary_keys, states, hi - 8, hi);
//!      hi -= 8;
//!    }
//!
//! @endrst
//!
//! @tparam BlockDimX Number of threads in the (1D) block.
// TODO(pauleonix): Add template parameters for deciding between deterministic and non-deterministic tie-breaking,
//                  as well as returning ranks for a full permutation or topk including all ties.
template <int BlockDimX>
class block_topk_rank
{
private:
  using algorithm_t  = block_topk_rank_atomic<BlockDimX>;
  using TempStorage_ = typename algorithm_t::TempStorage;

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

private:
  TempStorage_& storage_;

public:
  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit block_topk_rank(TempStorage& storage)
      : storage_(storage.Alias())
  {}

  //! Final tie-break + scatter-rank emission.
  //!
  //! The target k is read from @p states (locked in
  //! `block_topk_sieve::select_*`). Among items still `candidate` on entry,
  //! an unspecified subset is promoted to `selected` to fill the top-k.
  //!
  //! On output:
  //!
  //!   - For each item promoted to `selected`: `scatter_ranks[i]` is in
  //!     `[0, k)`. Selected items receive contiguous ranks starting at 0.
  //!   - For each item left as `rejected`: `scatter_ranks[i] == -1`.
  //!   - No item is left in the `candidate` state on return; `states`
  //!     contains exactly `min(k, initial num_valid)` selected items, with
  //!     `has_ties()` returning false.
  //!
  //! The ranks are designed for use with
  //! `cub::BlockExchange::ScatterTo[Blocked|Striped]Guarded`.
  //!
  //! @tparam ItemsPerThread Number of items per thread; deduced from
  //!                       @p states / @p scatter_ranks.
  template <int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  rank_key_states(block_topk_key_states<ItemsPerThread>& states, int (&scatter_ranks)[ItemsPerThread])
  {
    algorithm_t(storage_).rank_key_states(states, scatter_ranks);
  }
};
} // namespace detail

CUB_NAMESPACE_END
