// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#if __cccl_ptx_isa >= 860

#  include <cub/detail/strong_load.cuh>
#  include <cub/detail/strong_store.cuh>
#  include <cub/device/dispatch/kernels/warpspeed/allocators/SmemAllocator.h>
#  include <cub/device/dispatch/kernels/warpspeed/resource/SmemRef.cuh>
#  include <cub/device/dispatch/kernels/warpspeed/resource/SmemResource.cuh>
#  include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh>
#  include <cub/device/dispatch/kernels/warpspeed/squad/Squad.h>
#  include <cub/device/dispatch/kernels/warpspeed/values.h>
#  include <cub/thread/thread_reduce.cuh>
#  include <cub/thread/thread_scan.cuh>
#  include <cub/warp/warp_reduce.cuh>
#  include <cub/warp/warp_scan.cuh>

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__memory/align_down.h>
#  include <cuda/__memory/align_up.h>
#  include <cuda/ptx>
#  include <cuda/std/__algorithm/clamp.h>
#  include <cuda/std/__cccl/cuda_capabilities.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__type_traits/is_constant_evaluated.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cassert>

#  include <cudaTypedefs.h>

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

_CCCL_DEVICE_API inline void squadGetNextBlockIdx(const Squad& squad, SmemRef<uint4>& refDestSmem)
{
  if (squad.isLeaderThread())
  {
    ::cuda::ptx::clusterlaunchcontrol_try_cancel(&refDestSmem.data(), refDestSmem.ptrCurBarrierRelease());
  }
  refDestSmem.squadIncreaseTxCount(squad, refDestSmem.sizeBytes());
}

struct CpAsyncOobInfo
{
  char* ptrGmem;
  char* ptrGmemStartAlignDown;
  char* ptrGmemStartAlignUp;
  char* ptrGmemEnd;
  char* ptrGmemEndAlignDown;
  char* ptrGmemEndAlignUp;
  uint32_t overCopySizeBytes;
  uint32_t underCopySizeBytes;
  uint32_t origCopySizeBytes;
  uint32_t smemStartOffsetElem;
  uint32_t smemStartOffsetBytes;
  uint32_t smemEndOffsetElem;
  uint32_t smemEndOffsetBytes;
};

template <typename Tp>
_CCCL_DEVICE_API inline CpAsyncOobInfo prepareCpAsyncOob(const Tp* ptrGmem, uint32_t sizeElem)
{
  // We will copy from [ptrGmemBase, ptrGmemEnd). Both pointers have to be 16B
  // aligned.
  const Tp* ptrGmemStartAlignDown = cuda::align_down(ptrGmem, std::size_t(16));
  const Tp* ptrGmemStartAlignUp   = cuda::align_up(ptrGmem, std::size_t(16));
  const Tp* ptrGmemEnd            = ptrGmem + sizeElem;
  const Tp* ptrGmemEndAlignUp     = cuda::align_up(ptrGmemEnd, std::size_t(16));
  const Tp* ptrGmemEndAlignDown   = cuda::align_down(ptrGmemEnd, std::size_t(16));

  // Compute the final copy size in bytes. It can be either sizeElem or sizeElem + 16 / sizeof(T).
  uint32_t origCopySizeBytes  = static_cast<uint32_t>(sizeof(Tp) * sizeElem);
  uint32_t overCopySizeBytes  = static_cast<uint32_t>(sizeof(Tp) * (ptrGmemEndAlignUp - ptrGmemStartAlignDown));
  uint32_t underCopySizeBytes = static_cast<uint32_t>(sizeof(Tp) * (ptrGmemEndAlignDown - ptrGmemStartAlignUp));
  if (origCopySizeBytes < underCopySizeBytes)
  {
    // If ptrGmemStart and ptrGmemEnd are aligned to [1, .., 15] bytes, then
    // when we align the one up and the other down we get overflow. We check for
    // that here. In that case, the undercopy size is zero.
    underCopySizeBytes = 0;
  }
  // The offset in elements to the first valid element in shared memory.
  // ptrSmem + smemOffsetElem will point to the element copied from ptrGmem.
  uint32_t smemStartOffsetElem = static_cast<uint32_t>(ptrGmem - ptrGmemStartAlignDown);
  // The offset in elements between ptrGmemEnd and ptrGmemEndAlignDown
  uint32_t smemEndOffsetElem = static_cast<uint32_t>(ptrGmemEnd - ptrGmemEndAlignDown);

  return CpAsyncOobInfo{
    .ptrGmem               = (char*) ptrGmem,
    .ptrGmemStartAlignDown = (char*) ptrGmemStartAlignDown,
    .ptrGmemStartAlignUp   = (char*) ptrGmemStartAlignUp,
    .ptrGmemEnd            = (char*) ptrGmemEnd,
    .ptrGmemEndAlignDown   = (char*) ptrGmemEndAlignDown,
    .ptrGmemEndAlignUp     = (char*) ptrGmemEndAlignUp,
    .overCopySizeBytes     = overCopySizeBytes,
    .underCopySizeBytes    = underCopySizeBytes,
    .origCopySizeBytes     = static_cast<uint32_t>(sizeof(Tp) * sizeElem),
    .smemStartOffsetElem   = smemStartOffsetElem,
    .smemStartOffsetBytes  = static_cast<uint32_t>(sizeof(Tp) * smemStartOffsetElem),
    .smemEndOffsetElem     = smemEndOffsetElem,
    .smemEndOffsetBytes    = static_cast<uint32_t>(sizeof(Tp) * smemEndOffsetElem),
  };
}

template <typename Tp>
_CCCL_DEVICE_API inline void squadLoadBulk(const Squad& squad, SmemRef<Tp>& refDestSmem, CpAsyncOobInfo cpAsyncOobInfo)
{
  void* ptrSmem    = refDestSmem.data();
  uint64_t* ptrBar = refDestSmem.ptrCurBarrierRelease();

  if (squad.isLeaderThread())
  {
    ::cuda::ptx::cp_async_bulk(
      ::cuda::ptx::space_cluster,
      ::cuda::ptx::space_global,
      ptrSmem,
      cpAsyncOobInfo.ptrGmemStartAlignDown,
      cpAsyncOobInfo.overCopySizeBytes,
      ptrBar);
  }
  refDestSmem.squadIncreaseTxCount(squad, cpAsyncOobInfo.overCopySizeBytes);
}

template <typename OutputT>
_CCCL_DEVICE_API inline void squadStoreBulkSync(const Squad& squad, CpAsyncOobInfo cpAsyncOobInfo, OutputT* srcSmem)
{
  // This function performs either 1 copy, or three copies, depending on the
  // size and alignment of the output tile in global memory.
  //
  // If the output tile is contained in a single 16-byte aligned region, then we
  // only perform a single masked copy.
  //
  // If the output tile is larger or straddles two 16-byte aligned regions, then
  // we perform up to three copies:
  // - One copy for the first up to 15 bytes at the start of the region.
  // - One copy that starts at a 16-byte aligned address and ends at the latest 16-byte aligned address.
  // - One copy that cleans up the last up to 15 bytes.
  if (squad.isLeaderWarp())
  {
    // Acquire shared memory in async proxy
    // Perform fence.proxy.async with full warp to avoid BSSY+BSYNC
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);

    bool doStartCopy  = cpAsyncOobInfo.smemStartOffsetBytes > 0;
    bool doEndCopy    = cpAsyncOobInfo.smemEndOffsetBytes > 0;
    bool doMiddleCopy = cpAsyncOobInfo.ptrGmemStartAlignUp != cpAsyncOobInfo.ptrGmemEndAlignUp;

    uint16_t byteMask      = 0xFFFF;
    uint16_t byteMaskStart = byteMask << cpAsyncOobInfo.smemStartOffsetBytes;
    uint16_t byteMaskEnd   = byteMask >> (16 - cpAsyncOobInfo.smemEndOffsetBytes);
    // byteMaskStart contains zeroes at the left.
    uint16_t byteMaskSmall =
      byteMaskStart & (byteMask >> (16 - (cpAsyncOobInfo.ptrGmemEnd - cpAsyncOobInfo.ptrGmemStartAlignDown)));

    char* ptrSmemMiddle = (char*) srcSmem;
    if (doStartCopy)
    {
      ptrSmemMiddle += 16;
    }

    if (doMiddleCopy)
    {
      // Copy the middle part. Starting at byte 0 or 16 in shared memory. This
      // is the large copy. We perform this one first, so that the compiler can
      // (hopefully) hide all the arithmetic behind this instruction.
      if (::cuda::ptx::elect_sync(~0))
      {
        ::cuda::ptx::cp_async_bulk(
          ::cuda::ptx::space_global,
          ::cuda::ptx::space_shared,
          cpAsyncOobInfo.ptrGmemStartAlignUp,
          ptrSmemMiddle,
          cpAsyncOobInfo.underCopySizeBytes);
      }
      if (doStartCopy)
      {
        // Copy a subset of the first 16 bytes
        if (::cuda::ptx::elect_sync(~0))
        {
          ::cuda::ptx::cp_async_bulk_cp_mask(
            ::cuda::ptx::space_global,
            ::cuda::ptx::space_shared,
            cpAsyncOobInfo.ptrGmemStartAlignDown,
            srcSmem,
            /*size*/ 16,
            byteMaskStart);
        }
      }
      if (doEndCopy)
      {
        // Copy a subset of the last 16 bytes
        if (::cuda::ptx::elect_sync(~0))
        {
          ::cuda::ptx::cp_async_bulk_cp_mask(
            ::cuda::ptx::space_global,
            ::cuda::ptx::space_shared,
            ((char*) cpAsyncOobInfo.ptrGmemStartAlignUp) + cpAsyncOobInfo.underCopySizeBytes,
            ptrSmemMiddle + cpAsyncOobInfo.underCopySizeBytes,
            /*size*/ 16,
            byteMaskEnd);
        }
      }
    }
    else
    {
      // Copy a subset of the first 16 bytes
      if (::cuda::ptx::elect_sync(~0))
      {
        ::cuda::ptx::cp_async_bulk_cp_mask(
          ::cuda::ptx::space_global,
          ::cuda::ptx::space_shared,
          cpAsyncOobInfo.ptrGmemStartAlignDown,
          srcSmem,
          /*size*/ 16,
          byteMaskSmall);
      }
    }
    // Commit and wait for store to have completed reading from shared memory
    ::cuda::ptx::cp_async_bulk_commit_group();
    ::cuda::ptx::cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<0>{});
  }
}

template <typename InputT, typename AccumT, int elemPerThread>
_CCCL_DEVICE_API inline void squadLoadSmem(Squad squad, AccumT (&outReg)[elemPerThread], const InputT* smemBuf)
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    outReg[i] = smemBuf[squad.threadRank() * elemPerThread + i];
  }
}

template <typename OutputT, typename AccumT, int elemPerThread>
_CCCL_DEVICE_API inline void squadStoreSmem(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[elemPerThread])
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    smemBuf[squad.threadRank() * elemPerThread + i] = static_cast<OutputT>(inReg[i]);
  }
}

template <typename OutputT, typename AccumT, int elemPerThread>
_CCCL_DEVICE_API inline void
squadStoreSmemPartial(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[elemPerThread], int beginIndex, int endIndex)
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * elemPerThread + i;
    if (beginIndex <= elem_idx && elem_idx < beginIndex + endIndex)
    {
      smemBuf[elem_idx - beginIndex] = static_cast<OutputT>(inReg[i]);
    }
  }
}

template <typename Tp, int elemPerThread, typename ScanOpT>
_CCCL_DEVICE_API inline Tp threadReduce(const Tp (&regInput)[elemPerThread], ScanOpT& scan_op)
{
  return ThreadReduce(regInput, scan_op);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API inline Tp warpReduce(const Tp input, ScanOpT& scan_op)
{
  using warp_reduce_t = WarpReduce<Tp>;

  // TODO (elstehle): Do proper temporary storage allocation in case WarpReduce may rely on it
  static_assert(sizeof(typename warp_reduce_t::TempStorage) <= 4,
                "WarpReduce with non-trivial temporary storage is not supported yet in this kernel.");

  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API inline Tp warpReducePartial(const Tp input, ScanOpT& scan_op, const int num_items)
{
  using warp_reduce_t = WarpReduce<Tp>;

  // TODO (elstehle): Do proper temporary storage allocation in case WarpReduce may rely on it
  static_assert(sizeof(typename warp_reduce_t::TempStorage) <= 4,
                "WarpReduce with non-trivial temporary storage is not supported yet in this kernel.");

  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op, num_items);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API inline Tp warpScanExclusive(const Tp regInput, ScanOpT& scan_op)
{
  using warp_scan_t = WarpScan<Tp>;

  // TODO (elstehle): Do proper temporary storage allocation in case WarpReduce may rely on it
  static_assert(sizeof(typename warp_scan_t::TempStorage) <= 4,
                "WarpScan with non-trivial temporary storage is not supported yet in this kernel.");

  Tp result;
  typename warp_scan_t::TempStorage temp_storage;

  warp_scan_t{temp_storage}.ExclusiveScan(regInput, result, scan_op);

  return result;
}

template <int elemPerThread, typename AccumT, typename ScanOpT>
_CCCL_DEVICE_API inline void threadScanInclusive(AccumT (&regArray)[elemPerThread], ScanOpT& scan_op)
{
  detail::ThreadScanInclusive(regArray, regArray, scan_op);
}

template <typename AccumT>
_CCCL_DEVICE_API inline void
storeLookback(tmp_state_t<AccumT>* ptrTmpBuffer, int idxTile, scan_state scanState, AccumT sum)
{
  tmp_state_t<AccumT>* dst = ptrTmpBuffer + idxTile;
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
    int idxTileLookback = idxTileCur + 32 * (i + 1) - laneIdx;
    if (idxTileLookback < idxTileNext)
    {
      tmp_state_t<AccumT>* src = ptrTmpBuffer + idxTileLookback;
      if constexpr (sizeof(tmp_state_t<AccumT>) <= 16)
      {
        __nv_atomic_load(src, outTmpStates + i, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
      }
      else
      {
        using state_int       = ::cuda::std::underlying_type_t<scan_state>;
        outTmpStates[i].state = static_cast<scan_state>(load_acquire(reinterpret_cast<const state_int*>(&src->state)));
        outTmpStates[i].value = ThreadLoad<LOAD_CG>(&src->value);
      }
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
  const int laneIdx    = specialRegisters.laneIdx;
  const int lanemaskEq = ::cuda::ptx::get_sreg_lanemask_eq();

  int idxTileCur            = idxTilePrev;
  AccumT sumExclusiveCtaCur = sumExclusiveCtaPrev;

  while (idxTileCur < idxTileNext)
  {
    tmp_state_t<AccumT> regTmpStates[numTmpStatesPerThread] = {{EMPTY, AccumT{}}};
    warpLoadLookback(laneIdx, regTmpStates, ptrTmpBuffer, idxTileCur, idxTileNext);

    for (int idx = 0; idx < numTmpStatesPerThread; ++idx)
    {
      // Bitmask with a 1 bit in the position of the current lane if current
      // lane has an XXX state;
      int laneIsEmpty   = lanemaskEq * (regTmpStates[idx].state == EMPTY);
      int laneIsCumSum  = lanemaskEq * (regTmpStates[idx].state == CUM_SUM);
      int laneIsPrivSum = lanemaskEq * (regTmpStates[idx].state == PRIV_SUM);
      // Bitmask with 1 bits indicating which lane has an XX state.
      // TODO(miscco): This requires SM80 for clang-cuda
      int warpIsEmpty = 0;
      NV_IF_TARGET(NV_PROVIDES_SM_80, (warpIsEmpty = __reduce_or_sync(~0, laneIsEmpty);))
      int warpIsCumSum = 0;
      NV_IF_TARGET(NV_PROVIDES_SM_80, (warpIsCumSum = __reduce_or_sync(~0, laneIsCumSum);))
      int warpIsPrivSum = 0;
      NV_IF_TARGET(NV_PROVIDES_SM_80, (warpIsPrivSum = __reduce_or_sync(~0, laneIsPrivSum);))

      if (warpIsEmpty != 0)
      {
        break;
      }
      // Now we have either all private sums, or a mix of private sums and
      // cumulative sums.

      // Bitmask with a 1 bit indicating the position of the right-most
      // CUM_SUM state. If no CUM_SUM state present, value is zero.
      int warpRightMostCumSum = warpIsCumSum & -warpIsCumSum;
      // Bitmask with 1 bits to the right of the right-most CUM_SUM state.
      // If no CUM_SUM state present, value is all ones.
      int maskRightOfCumSum = warpRightMostCumSum - 1;

      // Sum all values of lanes containing either
      // (a) the right-most CUM_SUM, or
      // (b) subsequent PRIV_SUMs.
      AccumT localSum{};
      int maskSumParticipants = warpRightMostCumSum | maskRightOfCumSum;

      if ((maskSumParticipants & lanemaskEq) != 0)
      {
        localSum = regTmpStates[idx].value;
      }
      localSum = warpReduce(localSum, scan_op);

      if (warpIsCumSum == 0)
      {
        sumExclusiveCtaCur = scan_op(sumExclusiveCtaCur, localSum);
      }
      else
      {
        sumExclusiveCtaCur = localSum;
      }
      // idxTileCur can go beyond idxTileNext.
      idxTileCur += __popc(maskSumParticipants);
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
  //     storeLookback(ptrTmpBuffer, idxTileNext - 1, CUM_SUM, sumExclusiveCtaCur);
  //   }
  // }

  return sumExclusiveCtaCur;
}

namespace ptx = cuda::ptx;

template <typename InputT, typename OutputT, typename AccumT>
struct scanKernelParams
{
  InputT* ptrIn;
  OutputT* ptrOut;
  tmp_state_t<AccumT>* ptrTmp;
  size_t numElem;
  int numStages;
};

// Struct holding all scan kernel resources
template <typename WarpspeedPolicy, typename InputT, typename OutputT, typename AccumT>
struct ScanResources
{
  // Handle unaligned loads. We have 16 extra bytes of padding in every stage
  // for squadLoadBulk.
  using InT = InputT[WarpspeedPolicy::tile_size + 16 / sizeof(InputT)];
  using SumThreadAndWarpT =
    AccumT[WarpspeedPolicy::squadReduce().threadCount() + WarpspeedPolicy::squadReduce().warpCount()];

  SmemResource<InT> smemInOut; // will also be used to stage the output (as OutputT) for the bulk copy
  SmemResource<uint4> smemNextBlockIdx;
  SmemResource<AccumT> smemSumExclusiveCta;
  SmemResource<SumThreadAndWarpT> smemSumThreadAndWarp;
};
// Function to allocate resources.

template <typename WarpspeedPolicy, typename InputT, typename OutputT, typename AccumT>
[[nodiscard]] _CCCL_API constexpr ScanResources<WarpspeedPolicy, InputT, OutputT, AccumT>
allocResources(SyncHandler& syncHandler, SmemAllocator& smemAllocator, int numStages)
{
  using ScanResourcesT    = ScanResources<WarpspeedPolicy, InputT, OutputT, AccumT>;
  using InT               = typename ScanResourcesT::InT;
  using SumThreadAndWarpT = typename ScanResourcesT::SumThreadAndWarpT;

  // If numBlockIdxStages is one less than the number of stages, we find a small
  // speedup compared to setting it equal to num_stages. Not sure why.
  int numBlockIdxStages = numStages - 1;
  // Ensure we have at least 1 stage
  numBlockIdxStages = numBlockIdxStages < 1 ? 1 : numBlockIdxStages;

  // We do not need too many sumExclusiveCta stages. The lookback warp is the
  // bottleneck. As soon as it produces a new value, it will be consumed by the
  // scanStore squad, releasing the stage.
  int numSumExclusiveCtaStages = 2;

  ScanResourcesT res = {
    SmemResource<InT>(syncHandler, smemAllocator, Stages{numStages}),
    SmemResource<uint4>(syncHandler, smemAllocator, Stages{numBlockIdxStages}),
    SmemResource<AccumT>(syncHandler, smemAllocator, Stages{numSumExclusiveCtaStages}),
    SmemResource<SumThreadAndWarpT>(syncHandler, smemAllocator, Stages{numStages}),
  };
  // asdfasdf
  constexpr SquadDesc scanSquads[WarpspeedPolicy::num_squads] = {
    WarpspeedPolicy::squadReduce(),
    WarpspeedPolicy::squadScanStore(),
    WarpspeedPolicy::squadLoad(),
    WarpspeedPolicy::squadSched(),
    WarpspeedPolicy::squadLookback(),
  };

  res.smemInOut.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadLoad());
  res.smemInOut.addPhase(
    syncHandler, smemAllocator, {WarpspeedPolicy::squadReduce(), WarpspeedPolicy::squadScanStore()});

  res.smemNextBlockIdx.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadSched());
  res.smemNextBlockIdx.addPhase(syncHandler, smemAllocator, scanSquads);

  res.smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadLookback());
  res.smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadScanStore());

  res.smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadReduce());
  res.smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadScanStore());

  return res;
}

// The kernelBody device function is a straight-line implementation of the
// warp-specialized kernel.
//
// It is called from the __global__ kernel body with a squad argument that is
// the active squad on the current thread.
//
// Using this structure, all code that is not executed by the current
// squad is DCE (dead-code-eliminated) by the compiler and all
// warp-specialization dispatch is performed once at the start of the kernel and
// not in any of the hot loops (even if that may seem the case from a first
// glance at the code).
template <typename WarpspeedPolicy,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename RealInitValueT,
          bool ForceInclusive>
_CCCL_DEVICE_API inline void kernelBody(
  Squad squad,
  SpecialRegisters specialRegisters,
  const scanKernelParams<InputT, OutputT, AccumT>& params,
  ScanOpT scan_op,
  RealInitValueT real_init_value)
{
  ////////////////////////////////////////////////////////////////////////////////
  // Tuning dependent variables
  ////////////////////////////////////////////////////////////////////////////////
  static constexpr SquadDesc squadReduce    = WarpspeedPolicy::squadReduce();
  static constexpr SquadDesc squadScanStore = WarpspeedPolicy::squadScanStore();
  static constexpr SquadDesc squadLoad      = WarpspeedPolicy::squadLoad();
  static constexpr SquadDesc squadSched     = WarpspeedPolicy::squadSched();
  static constexpr SquadDesc squadLookback  = WarpspeedPolicy::squadLookback();

  constexpr int tile_size          = WarpspeedPolicy::tile_size;
  constexpr int num_lookback_tiles = WarpspeedPolicy::num_lookback_tiles;

  constexpr int elemPerThread = tile_size / squadReduce.threadCount();

  ////////////////////////////////////////////////////////////////////////////////
  // Resources
  ////////////////////////////////////////////////////////////////////////////////
  SyncHandler syncHandler{};
  SmemAllocator smemAllocator{};

  ScanResources<WarpspeedPolicy, InputT, OutputT, AccumT> res =
    allocResources<WarpspeedPolicy, InputT, OutputT, AccumT>(syncHandler, smemAllocator, params.numStages);

  syncHandler.clusterInitSync(specialRegisters);

  // Inclusive scan if no init_value type is provided
  static constexpr bool hasInit     = !::cuda::std::is_same_v<RealInitValueT, NullType>;
  static constexpr bool isInclusive = ForceInclusive || !hasInit;

  ////////////////////////////////////////////////////////////////////////////////
  // Pre-loop
  ////////////////////////////////////////////////////////////////////////////////

  // Start with the tile indicated by blockIdx.x
  int idxTile = specialRegisters.blockIdxX;
  // Lookback-specific variables:
  int idxTilePrev = -1;
  AccumT sumExclusiveCtaPrev{};
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();

  ////////////////////////////////////////////////////////////////////////////////
  // Loop over tiles
  ////////////////////////////////////////////////////////////////////////////////
#  pragma unroll 1
  while (true)
  {
    // Get stages. When these objects go out of scope, the stage of the resource
    // is automatically incremented.
    SmemStage stageNextBlockIdx     = res.smemNextBlockIdx.popStage();
    SmemStage stageInOut            = res.smemInOut.popStage();
    SmemStage stageSumThreadAndWarp = res.smemSumThreadAndWarp.popStage();
    SmemStage stageSumExclusiveCta  = res.smemSumExclusiveCta.popStage();

    // Split the stages into phases. Each resource goes through phases where it
    // is writeable by a set of threads and readable by a set of threads. To
    // acquire and release a phase, we need to arrive and wait on certain
    // barriers. The selection of the barriers is handled under the hood.
    auto [phaseNextBlockIdxW, phaseNextBlockIdxR]         = bindPhases<2>(stageNextBlockIdx);
    auto [phaseInOutW, phaseInOutRW]                      = bindPhases<2>(stageInOut);
    auto [phaseSumThreadAndWarpW, phaseSumThreadAndWarpR] = bindPhases<2>(stageSumThreadAndWarp);
    auto [phaseSumExclusiveCtaW, phaseSumExclusiveCtaR]   = bindPhases<2>(stageSumExclusiveCta);

    // We need to handle the first and the last -partial- tile differently
    const bool is_first_tile = idxTile == 0;

    if (squad == squadSched)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load next tile index
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
      squadGetNextBlockIdx(squad, refNextBlockIdxW);
    }

    const size_t idxTileBase = idxTile * size_t(tile_size);
    _CCCL_ASSERT(idxTileBase < params.numElem, "");
    const int valid_items   = static_cast<int>(cuda::std::min(params.numElem - idxTileBase, size_t(tile_size)));
    const bool is_last_tile = valid_items < tile_size;
    CpAsyncOobInfo loadInfo = prepareCpAsyncOob(params.ptrIn + idxTileBase, valid_items);

    if (squad == squadLoad)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load current tile
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refInOutW = phaseInOutW.acquireRef();
      squadLoadBulk(squad, refInOutW, loadInfo);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Get next tile index from shared memory (all squads)
    ////////////////////////////////////////////////////////////////////////////////
    uint4 regNextBlockIdx{};
    {
      SmemRef refNextBlockIdxR = phaseNextBlockIdxR.acquireRef();
      regNextBlockIdx          = refNextBlockIdxR.data();
      refNextBlockIdxR.setFenceLdsToAsyncProxy();
    }
    bool nextIdxTileValid = ::cuda::ptx::clusterlaunchcontrol_query_cancel_is_canceled(regNextBlockIdx);

    if (squad == squadReduce)
    {
      const int valid_items_this_thread =
        cuda::std::clamp(valid_items - squad.threadRank() * elemPerThread, 0, elemPerThread);
      const int valid_threads_this_warp =
        cuda::std::clamp(::cuda::ceil_div(valid_items, elemPerThread) - squad.warpRank() * 32, 0, 32);
      const int valid_warps = ::cuda::ceil_div(valid_items, elemPerThread * 32);
      _CCCL_ASSERT(0 < valid_warps && valid_warps <= squad.warpCount(), "");

      ////////////////////////////////////////////////////////////////////////////////
      // Load tile from shared memory
      ////////////////////////////////////////////////////////////////////////////////
      AccumT regThreadSum;
      AccumT regWarpSum;
      {
        // Acquire phaseInOutRW in this short scope
        SmemRef refInOutRW = phaseInOutRW.acquireRef();
        // Load data
        AccumT regInput[elemPerThread];
        // Handle unaligned refInOutRW.data() + loadInfo.smemStartOffsetElem points to the first element of the tile.
        // in the last tile, we load some invalid elements, but don't process them later
        squadLoadSmem(squad, regInput, &refInOutRW.data()[0] + loadInfo.smemStartOffsetElem);

        ////////////////////////////////////////////////////////////////////////////////
        // Reduce across thread and warp
        ////////////////////////////////////////////////////////////////////////////////
        if (is_last_tile)
        {
          regThreadSum = ThreadReducePartial(regInput, scan_op, valid_items_this_thread);
          regWarpSum   = warpReducePartial(regThreadSum, scan_op, valid_threads_this_warp);
        }
        else
        {
          regThreadSum = threadReduce(regInput, scan_op);
          regWarpSum   = warpReduce(regThreadSum, scan_op);
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store warp sump to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refSumThreadAndWarpW = phaseSumThreadAndWarpW.acquireRef();

      if (squad.isLeaderThreadOfWarp())
      {
        refSumThreadAndWarpW.data()[squadReduce.threadCount() + squad.warpRank()] = regWarpSum;
      }
      squad.syncThreads();

      ////////////////////////////////////////////////////////////////////////////////
      // Reduce across squad
      ////////////////////////////////////////////////////////////////////////////////
      // We need to accumulate the first element by hand because of the potential initial element and partial tiles
      AccumT regSquadSum;
      if constexpr (hasInit)
      {
        if (is_first_tile)
        {
          regSquadSum = scan_op(real_init_value, refSumThreadAndWarpW.data()[squadReduce.threadCount()]);
        }
        else
        {
          regSquadSum = refSumThreadAndWarpW.data()[squadReduce.threadCount()];
        }
      }
      else
      {
        regSquadSum = refSumThreadAndWarpW.data()[squadReduce.threadCount()];
      }

      if (is_last_tile)
      {
        for (int i = 1; i < valid_warps; ++i)
        {
          regSquadSum = scan_op(regSquadSum, refSumThreadAndWarpW.data()[squadReduce.threadCount() + i]);
        }
      }
      else
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 1; i < squadReduce.warpCount(); ++i)
        {
          regSquadSum = scan_op(regSquadSum, refSumThreadAndWarpW.data()[squadReduce.threadCount() + i]);
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store private sum for lookback
      ////////////////////////////////////////////////////////////////////////////////
      if (squad.isLeaderThread())
      {
        storeLookback(params.ptrTmp, idxTile, PRIV_SUM, regSquadSum);
      }
      ////////////////////////////////////////////////////////////////////////////////
      // Store thread sum
      ////////////////////////////////////////////////////////////////////////////////
      refSumThreadAndWarpW.data()[squad.threadRank()] = regThreadSum;
    }

    if (squad == squadLookback)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Perform lookback
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refSumExclusiveCtaW = phaseSumExclusiveCtaW.acquireRef();

      constexpr int numTmpStatesPerThread = num_lookback_tiles / 32;
      static_assert(num_lookback_tiles % 32 == 0, "num_lookback_tiles must be a multiple of 32");

      AccumT regSumExclusiveCta = warpIncrementalLookback<numTmpStatesPerThread>(
        specialRegisters, params.ptrTmp, idxTilePrev, sumExclusiveCtaPrev, idxTile, scan_op);
      if (squad.isLeaderThread())
      {
        refSumExclusiveCtaW.data() = regSumExclusiveCta;
      }
      sumExclusiveCtaPrev = regSumExclusiveCta;
      idxTilePrev         = idxTile - 1;
    }

    if (squad == squadScanStore)
    {
      static_assert(tile_size % squadScanStore.threadCount() == 0);

      // Sum of all threads up to but not including this one
      AccumT sumExclusive;

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across warp and thread sums
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Acquire refSumExclusiveCtaW briefly
        SmemRef refSumThreadAndWarpR = phaseSumThreadAndWarpR.acquireRef();
        // Add the sums of the preceding warps in this CTA to the cumulative
        // sum. These sums have been calculated in squadReduce(). We need
        // the reduce and scan squads to be the same size to do this.
        static_assert(squadReduce.warpCount() == squadScanStore.warpCount());

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < squadScanStore.warpCount(); ++i)
        {
          // We want a predicated unrolled loop here.
          if (i < squad.warpRank())
          {
            // First warp has nothing to add
            if (i == 0)
            {
              sumExclusive = refSumThreadAndWarpR.data()[squadReduce.threadCount()];
            }
            else
            {
              sumExclusive = scan_op(sumExclusive, refSumThreadAndWarpR.data()[squadReduce.threadCount() + i]);
            }
          }
        }

        // Add the sums of preceding threads in this warp to the cumulative sum.
        // Lane 0 reads invalid data.
        AccumT regSumThread = refSumThreadAndWarpR.data()[squad.threadRank()];

        // Perform scan of thread sums. If the warp contains partial data, we pass invalid elements to scan_op, and
        // sumExclusiveIntraWarp is invalid when the inputs were invalid and for warp_0/thread_0
        AccumT sumExclusiveIntraWarp = warpScanExclusive(regSumThread, scan_op);

        // warp_0 does not hold a valid value in sumExclusive, so only include it in other warps
        sumExclusive = squad.warpRank() == 0 ? sumExclusiveIntraWarp : scan_op(sumExclusive, sumExclusiveIntraWarp);
      }

      // sumExclusive is valid except for warp_0/thread_0

      ////////////////////////////////////////////////////////////////////////////////
      // Include sum of previous CTAs
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Briefly acquire refSumExclusiveCtaR (we have to do this for the first tile as well to prevent a hang)
        SmemRef refSumExclusiveCtaR = phaseSumExclusiveCtaR.acquireRef();

        if (!is_first_tile)
        {
          // Add the sums of preceding CTAs to the cumulative sum.
          AccumT regSumExclusiveCta = refSumExclusiveCtaR.data(); // this valid would be invalid in the first tile
          // sumExclusive is invalid in warp_0/thread_0, so only include it in other threads/warps
          sumExclusive = squad.threadRank() == 0 ? regSumExclusiveCta : scan_op(sumExclusive, regSumExclusiveCta);
        }
      }

      // sumExclusive is valid except for warp_0/thread_0 in the first tile

      // TODO(bgruber): consider merging the below branch into the next block of branches with `hasInit`
      if constexpr (hasInit)
      {
        if (is_first_tile)
        {
          // The first thread cannot use scan_op because sumExclusive holds garbage data
          if (squad.threadRank() == 0)
          {
            sumExclusive = static_cast<AccumT>(real_init_value);
          }
          else
          {
            sumExclusive = scan_op(static_cast<AccumT>(real_init_value), sumExclusive);
          }
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across elements allocated to this thread
      ////////////////////////////////////////////////////////////////////////////////
      AccumT regSumInclusive[elemPerThread] = {{}};

      // Acquire refInOut for remainder of scope.
      SmemRef refInOutRW = phaseInOutRW.acquireRef();
      // Handle unaligned refInOutRW.data() + loadInfo.smemStartOffsetElem points to
      // the first element of the tile.
      squadLoadSmem(squad, regSumInclusive, &refInOutRW.data()[0] + loadInfo.smemStartOffsetElem);

      // Perform inclusive scan of register array in current thread.
      if constexpr (hasInit)
      {
        if constexpr (isInclusive)
        {
          ThreadScanInclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive);
        }
        else
        {
          ThreadScanExclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive);
        }
      }
      else
      {
        if (is_first_tile && squad.threadRank() == 0)
        {
          // warp_0/thread_0 in the first tile when there is no initial value, we MUST NOT use sumExclusive
          if constexpr (isInclusive)
          {
            ThreadScanInclusive(regSumInclusive, regSumInclusive, scan_op);
          }
          else
          {
            ThreadScanExclusive(regSumInclusive, regSumInclusive, scan_op);
          }
        }
        else
        {
          if constexpr (isInclusive)
          {
            ThreadScanInclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive);
          }
          else
          {
            ThreadScanExclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive);
          }
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      // Sync before storing to avoid data races on SMEM
      squad.syncThreads();

      // if the output types fit into the input type tile, we can alias it
      OutputT* smem_output_tile = reinterpret_cast<OutputT*>(refInOutRW.data());
      if constexpr (sizeof(OutputT) <= sizeof(InputT))
      {
        CpAsyncOobInfo storeInfo = prepareCpAsyncOob(params.ptrOut + idxTileBase, valid_items);

        squadStoreSmem(squad, smem_output_tile + storeInfo.smemStartOffsetElem, regSumInclusive);
        // We do *not* release refSmemInOut here, because we will issue a TMA
        // instruction below. Instead, we issue a squad-local syncthreads +
        // fence.proxy.async to sync the shared memory writes with the TMA store.
        squad.syncThreads();

        ////////////////////////////////////////////////////////////////////////////////
        // Store result to global memory using TMA
        ////////////////////////////////////////////////////////////////////////////////
        squadStoreBulkSync(squad, storeInfo, smem_output_tile);
      }
      else
      {
        // otherwise, issue multiple bulk copies in chunks of the input tile size
        // TODO(bgruber): I am sure this could be implemented a lot more efficiently
        static constexpr int elem_per_chunk = (WarpspeedPolicy::tile_size * sizeof(InputT)) / sizeof(OutputT);
        for (int chunk_offset = 0; chunk_offset < static_cast<int>(valid_items); chunk_offset += elem_per_chunk)
        {
          const int chunk_size     = ::cuda::std::min(static_cast<int>(valid_items) - chunk_offset, elem_per_chunk);
          CpAsyncOobInfo storeInfo = prepareCpAsyncOob(params.ptrOut + idxTileBase + chunk_offset, chunk_size);
          OutputT* smemBuf         = smem_output_tile + storeInfo.smemStartOffsetElem;

          // only stage elements of the current chunk to SMEM
          squadStoreSmemPartial(
            squad,
            smem_output_tile + storeInfo.smemStartOffsetElem,
            regSumInclusive,
            chunk_offset,
            chunk_offset + chunk_size);

          // We do *not* release refSmemInOut here, because we will issue a TMA
          // instruction below. Instead, we issue a squad-local syncthreads +
          // fence.proxy.async to sync the shared memory writes with the TMA store.
          squad.syncThreads();

          ////////////////////////////////////////////////////////////////////////////////
          // Store result to global memory using TMA
          ////////////////////////////////////////////////////////////////////////////////
          squadStoreBulkSync(squad, storeInfo, smem_output_tile);

          squad.syncThreads();
        }
      }

      // Release refInOut. No need to do any cross-proxy fencing here, because
      // the TMA store in this warp and the TMA load in the load warp are both
      // async proxy.
    }

    ////////////////////////////////////////////////////////////////////////////////
    // All squads: Check loop condition and update next tile index
    ////////////////////////////////////////////////////////////////////////////////
    if (!nextIdxTileValid)
    {
      break;
    }
    // Update idxTile
    idxTile = ::cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(regNextBlockIdx);
  }

  if (squad == squadLoad)
  {
    _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  }
}

template <typename ActivePolicy, class = void>
inline constexpr int get_scan_block_threads = 1;

template <typename ActivePolicy>
inline constexpr int get_scan_block_threads<ActivePolicy, ::cuda::std::void_t<typename ActivePolicy::WarpspeedPolicy>> =
  ActivePolicy::WarpspeedPolicy::num_total_threads;

template <typename MaxPolicy,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT,
          bool ForceInclusive>
__launch_bounds__(get_scan_block_threads<typename MaxPolicy::ActivePolicy>, 1) __global__ void scan(
  const __grid_constant__ scanKernelParams<InputT, OutputT, AccumT> params, ScanOpT scan_op, InitValueT init_value)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_100, ({
      using ActivePolicy    = typename MaxPolicy::ActivePolicy;
      using WarpspeedPolicy = typename ActivePolicy::WarpspeedPolicy;

      // Cache special registers at start of kernel
      SpecialRegisters specialRegisters = getSpecialRegisters();

      // Dispatch for warp-specialization
      static constexpr SquadDesc scanSquads[WarpspeedPolicy::num_squads] = {
        WarpspeedPolicy::squadReduce(),
        WarpspeedPolicy::squadScanStore(),
        WarpspeedPolicy::squadLoad(),
        WarpspeedPolicy::squadSched(),
        WarpspeedPolicy::squadLookback(),
      };

      using real_init_value_t = typename InitValueT::value_type;

      squadDispatch(specialRegisters, scanSquads, [&](Squad squad) {
        kernelBody<WarpspeedPolicy, InputT, OutputT, AccumT, ScanOpT, real_init_value_t, ForceInclusive>(
          squad, specialRegisters, params, ::cuda::std::move(scan_op), static_cast<real_init_value_t>(init_value));
      });
    }))
}

template <typename AccumT>
__launch_bounds__(128) __global__ void initTmpStates(tmp_state_t<AccumT>* tmp, const size_t num_temp_states)
{
  const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (tile_id >= num_temp_states)
  {
    return;
  }
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  tmp[tile_id] = {EMPTY, AccumT{}};
}
} // namespace detail::scan

CUB_NAMESPACE_END

#endif // __cccl_ptx_isa >= 860
