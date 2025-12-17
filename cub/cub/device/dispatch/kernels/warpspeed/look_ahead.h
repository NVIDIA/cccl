// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__memory/is_aligned.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/strong_load.cuh>
#include <cub/detail/strong_store.cuh>
#include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/warp/specializations/warp_reduce_shfl.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__type_traits/underlying_type.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
enum scan_state : uint32_t
{
  EMPTY          = 0,
  TILE_AGGREGATE = 1,
};

template <typename AccumT>
struct tile_state_unaligned_t
{
  scan_state state;
  AccumT value;
};

// some older nvcc versions do not evaluate next_power_of_two() at compile time when called inside an attribute, so we
// have to force constant evaluation by assigning the result to a template parameter
template <typename AccumT, size_t _Alignment = ::cuda::next_power_of_two(sizeof(tile_state_unaligned_t<AccumT>))>
struct alignas(_Alignment) tile_state_t : tile_state_unaligned_t<AccumT>
{};

#if __cccl_ptx_isa >= 860

template <typename AccumT>
_CCCL_DEVICE_API inline void
storeTileAggregate(tile_state_t<AccumT>* ptrTileStates, scan_state scanState, AccumT sum, int index)
{
  _CCCL_ASSERT(::cuda::is_aligned(ptrTileStates, alignof(tile_state_t<AccumT>)), "");
  _CCCL_ASSERT(index >= 0 && index < gridDim.x, "Reading out of bounds tile state");

  if constexpr (sizeof(tile_state_t<AccumT>) <= 16)
  {
    tile_state_t<AccumT> tmp{scanState, sum};
    __nv_atomic_store(ptrTileStates + index, &tmp, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
  }
  else
  {
    ThreadStore<STORE_CG>(&ptrTileStates[index].value, sum);
    using state_int = ::cuda::std::underlying_type_t<scan_state>;
    store_release(reinterpret_cast<state_int*>(&ptrTileStates[index].state), scanState);
  }
}

template <typename AccumT>
_CCCL_DEVICE_API inline tile_state_t<AccumT> loadTileAggregate(tile_state_t<AccumT>* ptrTileStates, int index)
{
  _CCCL_ASSERT(::cuda::is_aligned(ptrTileStates, alignof(tile_state_t<AccumT>)), "");
  _CCCL_ASSERT(index >= 0 && index < gridDim.x, "Reading out of bounds tile state");

  tile_state_t<AccumT> res;
  if constexpr (sizeof(tile_state_t<AccumT>) <= 16)
  {
    __nv_atomic_load(ptrTileStates + index, &res, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
  }
  else
  {
    using state_int = ::cuda::std::underlying_type_t<scan_state>;
    res.state = static_cast<scan_state>(load_acquire(reinterpret_cast<const state_int*>(&ptrTileStates[index].state)));
    res.value = ThreadLoad<LOAD_CG>(&ptrTileStates[index].value);
  }
  return res;
}

// warpLoadLookback loads tmp states:
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
// idxTileNext, then the state is not loaded from memory and set to EMPTY.
template <int numTileStatesPerThread, typename AccumT>
_CCCL_DEVICE_API inline void warpLoadLookback(
  int laneIdx,
  tile_state_t<AccumT> (&outTileStates)[numTileStatesPerThread],
  tile_state_t<AccumT>* ptrTileStates,
  int idxTileCur,
  int idxTileNext)
{
  for (int i = 0; i < numTileStatesPerThread; ++i)
  {
    const int idxTileLookback = idxTileCur + 32 * i + laneIdx;
    if (idxTileLookback < idxTileNext)
    {
      outTileStates[i] = loadTileAggregate(ptrTileStates, idxTileLookback);
    }
    else
    {
      // If we are looking ahead of idxTileNext, then set state to EMPTY
      outTileStates[i].state = EMPTY;
    }
  }
}

// warpIncrementalLookback takes the latest known sumExclusiveCtaPrev and its tile index, idxTilePrev (which's sum is
// NOT included in sumExclusiveCtaPrev), and computes the sumExclusiveCta for the next tile of interest, idxTileNext
// (where the returned value will NOT include the sum of idxTileNext).
//
// It does so by loading states in chunks of 32 * numTileStatesPerThread
// elements, starting from idxTilePrev + 1. From the chunk of states, it tries
// to advance its knowledge of sumExclusiveCta as much as possible. It loops
// until it can calculate the value of sumExclusiveCta from the preceding
// states.
//
// The function must be called from a single warp. All passed arguments must be
// warp-uniform.
//
template <int numTileStatesPerThread, typename AccumT, typename ScanOpT>
[[nodiscard]] _CCCL_DEVICE_API inline AccumT warpIncrementalLookback(
  SpecialRegisters specialRegisters,
  tile_state_t<AccumT>* ptrTileStates,
  const int idxTilePrev,
  const AccumT sumExclusiveCtaPrev,
  const int idxTileNext,
  ScanOpT& scan_op)
{
  const int laneIdx         = specialRegisters.laneIdx;
  const uint32_t lanemaskEq = ::cuda::ptx::get_sreg_lanemask_eq();

  int idxTileCur            = idxTilePrev;
  AccumT sumExclusiveCtaCur = sumExclusiveCtaPrev;

  using warp_reduce_t = WarpReduce<AccumT>;
  static_assert(sizeof(typename warp_reduce_t::TempStorage) <= 4,
                "WarpReduce with non-trivial temporary storage is not supported yet in this kernel.");
  typename warp_reduce_t::TempStorage temp_storage;

  using warp_reduce_or_t = WarpReduce<uint32_t>;
  typename warp_reduce_or_t::TempStorage temp_storage_or;
  warp_reduce_or_t warp_reduce_or{temp_storage_or};
  constexpr ::cuda::std::bit_or<uint32_t> or_op{};

  while (idxTileCur < idxTileNext)
  {
    tile_state_t<AccumT> regTmpStates[numTileStatesPerThread];
    warpLoadLookback(laneIdx, regTmpStates, ptrTileStates, idxTileCur, idxTileNext);

    for (int idx = 0; idx < numTileStatesPerThread; ++idx)
    {
      // Bitmask with a 1 bit in the position of the current lane if current lane has a tile aggregate
      const uint32_t lane_has_aggregate = lanemaskEq * (regTmpStates[idx].state == TILE_AGGREGATE);

      // Bitmask with 1 bits indicating which lane has a tile aggregate
      const uint32_t warp_has_aggregate_mask = warp_reduce_or.Reduce(lane_has_aggregate, or_op);

      // Bitmask with 1 bits for all rightmost lanes having a tile aggregate
      const uint32_t warp_right_aggregates_mask = warp_has_aggregate_mask & (~warp_has_aggregate_mask - 1);

      // Cannot reduce if no rightmost tile aggregates
      if (warp_right_aggregates_mask == 0)
      {
        break;
      }

      const uint32_t warp_right_aggregates_count = ::cuda::std::popcount(warp_right_aggregates_mask);

      // Accumulate the rightmost tile aggregates
      AccumT local_sum;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_80,
        ({ // NOTE: Inlined from warp_reduce_shfl
          if constexpr (::cuda::std::is_integral_v<AccumT> && sizeof(AccumT) <= sizeof(unsigned)
                        && (is_cuda_std_plus_v<ScanOpT, AccumT> || is_cuda_minimum_maximum_v<ScanOpT, AccumT>
                            || is_cuda_std_bitwise_v<ScanOpT, AccumT>) )
          {
            const bool use_value = lanemaskEq & warp_right_aggregates_mask;
            const AccumT value   = use_value ? regTmpStates[idx].value : identity_v<ScanOpT, AccumT>;
            local_sum            = reduce_op_sync(value, ~0, scan_op);
          }
          else
          {
            // TODO(bgruber): this generates a LOT of SASS. I think it can do better.
            local_sum =
              warp_reduce_t{temp_storage}.Reduce(regTmpStates[idx].value, scan_op, warp_right_aggregates_count);
          }
        }),
        (local_sum = warp_reduce_t{temp_storage}.Reduce(regTmpStates[idx].value, scan_op, warp_right_aggregates_count);))

      // We never initialized sumExclusiveCtaCur when starting look ahead at tile 0
      sumExclusiveCtaCur = idxTileCur == 0 ? local_sum : scan_op(sumExclusiveCtaCur, local_sum);
      idxTileCur += warp_right_aggregates_count;

      // we can only continue on the next 32 tile states, if we consumed all 32 of this iteration
      if (warp_right_aggregates_count < 32)
      {
        break;
      }
    }
  }

  return sumExclusiveCtaCur; // must only be valid in lane_0
}

#endif // __cccl_ptx_isa >= 860
} // namespace detail::scan

CUB_NAMESPACE_END
