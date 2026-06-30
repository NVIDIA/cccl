// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/strong_load.cuh>
#include <cub/detail/strong_store.cuh>
#include <cub/detail/warpspeed/special_registers.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/warp/specializations/warp_redux.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/underlying_type.h>

#if !_CCCL_HAS_NV_ATOMIC_BUILTINS()
#  include <cuda/atomic>
#endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL ::cuda::std::size_t max_native_atomic_size() noexcept
{
#if _CCCL_CUDA_COMPILER(NVHPC)
  return 8;
#else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC)  vvv
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return 16;), (return 8;))
#endif // !_CCCL_CUDA_COMPILER(NVHPC)
}

enum scan_state : ::cuda::std::uint32_t
{
  empty          = 0,
  tile_aggregate = 1,
};

template <typename AccumT>
struct tile_state_unaligned_t
{
  scan_state state;
  AccumT value;
};

// some older nvcc versions do not evaluate next_power_of_two() at compile time when called inside an attribute, so we
// have to force constant evaluation by assigning the result to a template parameter
template <typename AccumT,
          ::cuda::std::size_t _Alignment = ::cuda::next_power_of_two(sizeof(tile_state_unaligned_t<AccumT>))>
struct alignas(_Alignment) tile_state_t : tile_state_unaligned_t<AccumT>
{};

#if __cccl_ptx_isa >= 860

template <typename AccumT>
_CCCL_DEVICE_API void
storeTileAggregate(tile_state_t<AccumT>* ptrTileStates, scan_state scanState, AccumT aggr, int index)
{
  _CCCL_ASSERT(::cuda::is_aligned(ptrTileStates, alignof(tile_state_t<AccumT>)), "");
  _CCCL_ASSERT(index >= 0 && index < gridDim.x, "Reading out of bounds tile state");

  if constexpr (sizeof(tile_state_t<AccumT>) <= cub::detail::warpspeed::max_native_atomic_size()
                && ::cuda::is_trivially_copyable_v<tile_state_t<AccumT>>)
  {
    static_assert(::cuda::is_power_of_two(sizeof(tile_state_t<AccumT>)));
    tile_state_t<AccumT> tmp{scanState, aggr};

#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
    __nv_atomic_store(ptrTileStates + index, &tmp, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
    ::cuda::atomic_ref<tile_state_t<AccumT>, ::cuda::std::thread_scope_device>{ptrTileStates[index]}.store(
      tmp, ::cuda::std::memory_order_relaxed);
#  endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()
  }
  else
  {
    ThreadStore<STORE_CG>(&ptrTileStates[index].value, aggr);
    using state_int = ::cuda::std::underlying_type_t<scan_state>;
    store_release(reinterpret_cast<state_int*>(&ptrTileStates[index].state), scanState);
  }
}

template <typename AccumT>
_CCCL_DEVICE_API tile_state_t<AccumT> loadTileAggregate(tile_state_t<AccumT>* ptrTileStates, int index)
{
  _CCCL_ASSERT(::cuda::is_aligned(ptrTileStates, alignof(tile_state_t<AccumT>)), "");
  _CCCL_ASSERT(index >= 0 && index < gridDim.x, "Reading out of bounds tile state");

  tile_state_t<AccumT> res;
  if constexpr (sizeof(tile_state_t<AccumT>) <= cub::detail::warpspeed::max_native_atomic_size()
                && ::cuda::is_trivially_copyable_v<tile_state_t<AccumT>>)
  {
    static_assert(::cuda::is_power_of_two(sizeof(tile_state_t<AccumT>)));
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
    __nv_atomic_load(ptrTileStates + index, &res, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
    res = ::cuda::atomic_ref<tile_state_t<AccumT>, ::cuda::std::thread_scope_device>{ptrTileStates[index]}.load(
      ::cuda::std::memory_order_relaxed);
#  endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()
  }
  else
  {
    using state_int = ::cuda::std::underlying_type_t<scan_state>;
    res.state = static_cast<scan_state>(load_acquire(reinterpret_cast<const state_int*>(&ptrTileStates[index].state)));
    res.value = ThreadLoad<LOAD_CG>(&ptrTileStates[index].value);
  }
  return res;
}

// warpLoadLookahead loads tmp states:
//   idxTileCur + [0; 32 * numTileStatesPerThread[
//
// The states are loaded in laneId order and warp-strided:
//
// outTmpStates[0] contains:
//   Lane 0:  idxTileCur + 0
//   Lane 1:  idxTileCur + 1
//   ...
//   Lane 31: idxTileCur + 31
//
// outTmpStates[1] contains:
//   Lane 0: idxTileCur + 32
//   ...
//   Lane 31 idxTileCur + 63
//
// If the index idxTileCur + ii of the loaded state is equal to or exceeds idxTileNext, i.e., idxTileCur + ii >=
// idxTileNext, then the state is not loaded from memory and set to empty.
template <int numTileStatesPerThread, typename AccumT>
_CCCL_DEVICE_API void warpLoadLookahead(
  int laneIdx,
  tile_state_t<AccumT> (&outTileStates)[numTileStatesPerThread],
  tile_state_t<AccumT>* ptrTileStates,
  int idxTileCur,
  int idxTileNext)
{
  for (int i = 0; i < numTileStatesPerThread; ++i)
  {
    const int idxTileLookahead = idxTileCur + 32 * i + laneIdx;
    if (idxTileLookahead < idxTileNext)
    {
      outTileStates[i] = loadTileAggregate(ptrTileStates, idxTileLookahead);
    }
    else
    {
      // If we are looking ahead of idxTileNext, then set state to empty
      outTileStates[i].state = scan_state::empty;
    }
  }
}

// warpIncrementalLookahead takes the latest known aggrExclusiveCtaPrev and its tile index, idxTilePrev (which's
// aggregate is NOT included in aggrExclusiveCtaPrev), and computes the aggrExclusiveCta for the next tile of interest,
// idxTileNext (where the returned value will NOT include the aggregate of idxTileNext).
//
// It does so by loading states in chunks of 32 * numTileStatesPerThread elements, starting from idxTilePrev + 1. From
// the chunk of states, it tries to advance its knowledge of aggrExclusiveCta as much as possible. It loops until it can
// calculate the value of aggrExclusiveCta from the preceding states.
//
// The function must be called from a single warp. All passed arguments must be warp-uniform.
template <int numTileStatesPerThread, typename AccumT, typename ScanOpT>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT warpIncrementalLookahead(
  SpecialRegisters specialRegisters,
  tile_state_t<AccumT>* ptrTileStates,
  const int idxTilePrev,
  const AccumT aggrExclusiveCtaPrev,
  const int idxTileNext,
  ScanOpT& scan_op)
{
  const int laneIdx                                       = static_cast<int>(specialRegisters.laneIdx);
  [[maybe_unused]] const ::cuda::std::uint32_t lanemaskEq = ::cuda::ptx::get_sreg_lanemask_eq();

  int idxTileCur             = idxTilePrev;
  AccumT aggrExclusiveCtaCur = aggrExclusiveCtaPrev;

  using warp_reduce_t = WarpReduce<AccumT>;
  static_assert(::cuda::std::is_same_v<typename warp_reduce_t::TempStorage, Uninitialized<NullType>>,
                "WarpReduce for a full warp must not require temporary storage");
  [[maybe_unused]] typename warp_reduce_t::TempStorage temp_storage;

  while (idxTileCur < idxTileNext)
  {
    tile_state_t<AccumT> regTmpStates[numTileStatesPerThread];
    warpLoadLookahead(laneIdx, regTmpStates, ptrTileStates, idxTileCur, idxTileNext);

    for (int idx = 0; idx < numTileStatesPerThread; ++idx)
    {
      // Bitmask with 1 bits indicating which lane has a tile aggregate
      const ::cuda::std::uint32_t warp_has_aggregate_mask =
        __ballot_sync(0xffffffffu, regTmpStates[idx].state == scan_state::tile_aggregate);

      // Bitmask with 1 bits for all rightmost lanes having a tile aggregate
      const ::cuda::std::uint32_t warp_right_aggregates_mask = warp_has_aggregate_mask & (~warp_has_aggregate_mask - 1);

      // Cannot reduce if no rightmost tile aggregates
      if (warp_right_aggregates_mask == 0)
      {
        break;
      }

      const ::cuda::std::uint32_t warp_right_aggregates_count = ::cuda::std::popcount(warp_right_aggregates_mask);

      // Accumulate the rightmost tile aggregates
      AccumT local_aggr;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_80,
        ({ // NOTE: Inlined from warp_reduce_shfl
          if constexpr (is_warp_redux_op_supported_sm80<ScanOpT, AccumT>)
          {
            const bool use_value = lanemaskEq & warp_right_aggregates_mask;
            const AccumT value   = use_value ? regTmpStates[idx].value : cuda::identity_element<ScanOpT, AccumT>();
            local_aggr           = cub::detail::warp_redux_sm80(value, ~0, scan_op);
          }
          else
          {
            // TODO(bgruber): this generates a LOT of SASS. I think it can do better.
            local_aggr =
              warp_reduce_t{temp_storage}.Reduce(regTmpStates[idx].value, scan_op, warp_right_aggregates_count);
          }
        }),
        (local_aggr =
           warp_reduce_t{temp_storage}.Reduce(regTmpStates[idx].value, scan_op, warp_right_aggregates_count);))

      // We never initialized aggrExclusiveCtaCur when starting look ahead at tile 0
      aggrExclusiveCtaCur = idxTileCur == 0 ? local_aggr : scan_op(aggrExclusiveCtaCur, local_aggr);
      idxTileCur += warp_right_aggregates_count;

      // we can only continue on the next 32 tile states, if we consumed all 32 of this iteration
      if (warp_right_aggregates_count < 32)
      {
        break;
      }
    }
  }

  return aggrExclusiveCtaCur; // must only be valid in lane_0
}

// Deterministic version of warpIncrementalLookahead that returns the same aggrExclusiveCta. The difference is that it
// always starts the lookahead from a tile index that is a multiple of 32. The left pointer (idxTilePrev) is itself
// always a multiple of 32, as it starts at 0 and is only ever advanced by whole batches of 32, so the lookahead resumes
// from there directly. Because every reduction begins at the same fixed tiles, no matter which tiles happened to finish
// first, the order in which values are summed is always the same and the result is identical on every run.
// idxTilePrev/aggrExclusiveCtaPrev are updated by reference to the last multiple of 32.
template <int numTileStatesPerThread, typename AccumT, typename ScanOpT>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT warpIncrementalLookaheadStable(
  SpecialRegisters specialRegisters,
  tile_state_t<AccumT>* ptrTileStates,
  int& idxTilePrev,
  AccumT& aggrExclusiveCtaPrev,
  const int idxTileNext,
  ScanOpT& scan_op)
{
  const int laneIdx                      = static_cast<int>(specialRegisters.laneIdx);
  const ::cuda::std::uint32_t lanemaskEq = ::cuda::ptx::get_sreg_lanemask_eq();

  int idxTileCur             = idxTilePrev;
  AccumT aggrExclusiveCtaCur = aggrExclusiveCtaPrev;

  using warp_reduce_t = WarpReduce<AccumT>;
  static_assert(::cuda::std::is_same_v<typename warp_reduce_t::TempStorage, Uninitialized<NullType>>,
                "WarpReduce for a full warp must not require temporary storage");
  [[maybe_unused]] typename warp_reduce_t::TempStorage temp_storage;

  while (idxTileCur < idxTileNext)
  {
    tile_state_t<AccumT> regTmpStates[numTileStatesPerThread];
    warpLoadLookahead(laneIdx, regTmpStates, ptrTileStates, idxTileCur, idxTileNext);

    for (int idx = 0; idx < numTileStatesPerThread; ++idx)
    {
      // Bitmask with 1 bits indicating which lane has a tile aggregate
      const ::cuda::std::uint32_t warp_has_aggregate_mask =
        __ballot_sync(0xffffffffu, regTmpStates[idx].state == scan_state::tile_aggregate);

      // Bitmask with 1 bits for the contiguous run of lanes having a tile aggregate starting from LSB
      const ::cuda::std::uint32_t warp_right_aggregates_mask = warp_has_aggregate_mask & (~warp_has_aggregate_mask - 1);

      const ::cuda::std::uint32_t warp_right_aggregates_count = ::cuda::std::popcount(warp_right_aggregates_mask);

      // Only reduce once 32 contiguous tile aggregates are available, so the reduction order is fixed.
      const ::cuda::std::uint32_t expected_count =
        static_cast<::cuda::std::uint32_t>(::cuda::std::min(32, idxTileNext - idxTileCur));
      if (warp_right_aggregates_count < expected_count)
      {
        break;
      }

      const bool use_value    = lanemaskEq & warp_right_aggregates_mask;
      const AccumT value      = use_value ? regTmpStates[idx].value : cuda::identity_element<ScanOpT, AccumT>();
      const AccumT local_aggr = warp_reduce_t{temp_storage}.Reduce(value, scan_op);

      if (expected_count == 32)
      {
        aggrExclusiveCtaCur = idxTileCur == 0 ? local_aggr : scan_op(aggrExclusiveCtaCur, local_aggr);
        idxTileCur += 32;
      }
      else
      {
        const AccumT full_aggr = idxTileCur == 0 ? local_aggr : scan_op(aggrExclusiveCtaCur, local_aggr);
        idxTilePrev            = idxTileCur;
        aggrExclusiveCtaPrev   = aggrExclusiveCtaCur;
        return full_aggr;
      }
    }
  }

  // Only reached when idxTileNext is a multiple of 32; otherwise the final partial batch full aggregate returns inside
  // the loop above.
  idxTilePrev          = idxTileNext;
  aggrExclusiveCtaPrev = aggrExclusiveCtaCur;
  return aggrExclusiveCtaCur; // must only be valid in lane_0
}

#endif // __cccl_ptx_isa >= 860
} // namespace detail::warpspeed

CUB_NAMESPACE_END
