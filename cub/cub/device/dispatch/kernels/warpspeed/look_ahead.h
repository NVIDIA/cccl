// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
#include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__type_traits/underlying_type.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
enum scan_state : uint32_t
{
  EMPTY    = 0,
  PRIV_SUM = 1,
  CUM_SUM  = 2
};

template <typename AccumT>
struct tmp_state_t
{
  scan_state state;
  AccumT value;
};

template <typename AccumT>
_CCCL_DEVICE_API inline void storeLookbackTile(tmp_state_t<AccumT>* dst, scan_state scanState, AccumT sum)
{
  if constexpr (sizeof(tmp_state_t<AccumT>) <= 16)
  {
    tmp_state_t<AccumT> tmp{scanState, sum};
    __nv_atomic_store(dst, &tmp, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
  }
  else
  {
    ThreadStore<STORE_CG>(&dst->value, sum);
    using state_int = ::cuda::std::underlying_type_t<scan_state>;
    store_release(reinterpret_cast<state_int*>(&dst->state), scanState);
  }
}

template <typename AccumT, int numTmpStatesPerThread>
_CCCL_DEVICE_API inline void
loadLookbackTile(tmp_state_t<AccumT>* src, tmp_state_t<AccumT> (&outTmpStates)[numTmpStatesPerThread], const int index)
{
  if constexpr (sizeof(tmp_state_t<AccumT>) <= 16)
  {
    __nv_atomic_load(src, outTmpStates + index, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
  }
  else
  {
    using state_int           = ::cuda::std::underlying_type_t<scan_state>;
    outTmpStates[index].state = static_cast<scan_state>(load_acquire(reinterpret_cast<const state_int*>(&src->state)));
    outTmpStates[index].value = ThreadLoad<LOAD_CG>(&src->value);
  }
}

// warpLoadLookback loads tmp states
//
//   idxTileCur + 1, idxTileCur + 2, ..., idxTileCur + 32 * (numTmpStatesPerThread + 1)
//
// The states are loaded in reverse-laneId order, i.e.,
//
// outTmpStates[0] contains:
//   Lane 0:  idxTileCur + 32
//   Lane 1:  idxTileCur + 31
//   ...
//   Lane 31: idxTileCur + 1
//
// outTmpStates[1] contains:
//   Lane 0: idxTileCur + 64
//   ...
//   Lane 31 idxTileCur + 33
//
// If the index idxTileCur + ii of the loaded state is equal to or exceeds
// idxTileNext, i.e., idxTileNext <= idxTileCur + ii, then the state is not
// loaded from memory and filled with {PRIV_SUM, 0}.
template <int numTmpStatesPerThread, typename AccumT>
_CCCL_DEVICE_API inline void warpLoadLookback(
  int laneIdx,
  tmp_state_t<AccumT> (&outTmpStates)[numTmpStatesPerThread],
  tmp_state_t<AccumT>* ptrTmpBuffer,
  int idxTileCur,
  int idxTileNext)
{
  for (int i = 0; i < numTmpStatesPerThread; ++i)
  {
    const int idxTileLookback = idxTileCur + 32 * (i + 1) - laneIdx;
    if (idxTileLookback < idxTileNext)
    {
      loadLookbackTile(ptrTmpBuffer + idxTileLookback, outTmpStates, i);
    }
    else
    {
      // If we are looking ahead of idxTileNext, then set state to a private sum
      // of zero to simplify handling later on.
      outTmpStates[i] = {PRIV_SUM, 0};
    }
  }
}

// warpIncrementalLookback takes the latest current known sumExclusiveCtaPrev
// and its tile index, idxTilePrev, and computes the sumExclusiveCta for the
// next tile of interest, idxTileNext.
//
// It does so by loading states in chunks of 32 * numTmpStatesPerThread
// elements, starting from idxTilePrev + 1. From the chunk of states, it tries
// to advance its knowledge of sumExclusiveCta as much as possible. It loops
// until it can calculate the value of sumExclusiveCta from the preceding
// states.
//
// The function must be called from a single warp. All passed arguments must be
// warp-uniform.
//
template <int numTmpStatesPerThread, typename AccumT, typename ScanOpT>
[[nodiscard]] _CCCL_DEVICE_API inline AccumT warpIncrementalLookback(
  SpecialRegisters specialRegisters,
  tmp_state_t<AccumT>* ptrTmpBuffer,
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
    tmp_state_t<AccumT> regTmpStates[numTmpStatesPerThread];
    warpLoadLookback(laneIdx, regTmpStates, ptrTmpBuffer, idxTileCur, idxTileNext);

    for (int idx = 0; idx < numTmpStatesPerThread; ++idx)
    {
      // Bitmask with a 1 bit in the position of the current lane if current lane has an XXX state;
      const uint32_t laneIsEmpty   = lanemaskEq * (regTmpStates[idx].state == EMPTY);
      const uint32_t laneIsCumSum  = lanemaskEq * (regTmpStates[idx].state == CUM_SUM);
      const uint32_t laneIsPrivSum = lanemaskEq * (regTmpStates[idx].state == PRIV_SUM);

      // Bitmask with 1 bits indicating which lane has an XX state.
      const uint32_t warpIsEmpty   = warp_reduce_or.Reduce(laneIsEmpty, or_op);
      const uint32_t warpIsCumSum  = warp_reduce_or.Reduce(laneIsCumSum, or_op);
      const uint32_t warpIsPrivSum = warp_reduce_or.Reduce(laneIsPrivSum, or_op);

      if (warpIsEmpty != 0)
      {
        break;
      }
      // Now we have either all private sums, or a mix of private sums and
      // cumulative sums.

      // Bitmask with a 1 bit indicating the position of the right-most
      // CUM_SUM state. If no CUM_SUM state present, value is zero.
      const uint32_t warpRightMostCumSum = warpIsCumSum & -warpIsCumSum;
      // Bitmask with 1 bits to the right of the right-most CUM_SUM state.
      // If no CUM_SUM state present, value is all ones.
      const uint32_t maskRightOfCumSum = warpRightMostCumSum - 1;

      // Sum all values of lanes containing either
      // (a) the right-most CUM_SUM, or
      // (b) subsequent PRIV_SUMs.
      AccumT localSum;
      const uint32_t maskSumParticipants = warpRightMostCumSum | maskRightOfCumSum;

      if ((maskSumParticipants & lanemaskEq) != 0)
      {
        localSum = regTmpStates[idx].value;
      }
      localSum = warp_reduce_t{temp_storage}.Reduce(localSum, scan_op);

      if (warpIsCumSum == 0)
      {
        sumExclusiveCtaCur = scan_op(sumExclusiveCtaCur, localSum);
      }
      else
      {
        sumExclusiveCtaCur = localSum;
      }
      // idxTileCur can go beyond idxTileNext.
      idxTileCur += ::cuda::std::popcount(maskSumParticipants);
    }
  }

  // We are not storing CUM_SUM states, because it makes updating idxTileCur
  // difficult. Storing CUM_SUM states is the more robust approach though: if
  // for whatever reason GETNEXTWORKID fails at some point, any freshly launched
  // block must look back many iterations to figure out the current
  // sumExclusiveCta.
  //
  // Just for future reference, the following commented code is kept around.

  // if (idxTileNext > 0) {
  //   if (specialRegisters.laneIdx == 0) {
  //     storeLookbackTile(ptrTmpBuffer, idxTileNext - 1, CUM_SUM, sumExclusiveCtaCur);
  //   }
  // }

  return sumExclusiveCtaCur;
}
} // namespace detail::scan

CUB_NAMESPACE_END
