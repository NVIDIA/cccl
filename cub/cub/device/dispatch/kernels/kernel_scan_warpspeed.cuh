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

#  include <cuda/__memory/align_down.h>
#  include <cuda/__memory/align_up.h>
#  include <cuda/ptx>
#  include <cuda/std/__cccl/cuda_capabilities.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cassert>

#  include <cudaTypedefs.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
_CCCL_DEVICE_API inline int shfl_sync_up(bool& in_range, int value, int offset, int c, int member_mask)
{
  unsigned int output;
  int tmp_in_range;
  asm volatile(
    "{"
    "  .reg .pred p;"
    "  shfl.sync.up.b32 %0|p, %2, %3, %4, %5;"
    "  selp.b32 %1, 1, 0, p;\n"
    "  "
    "}"
    : "=r"(output), "=r"(tmp_in_range)
    : "r"(value), "r"(offset), "r"(c), "r"(member_mask));
  in_range = bool(tmp_in_range);
  return output;
}

enum scan_state
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
  void* ptrGmemBase;
  uint32_t copySizeBytes;
  uint32_t smemOffsetElem;
};

template <typename Tp>
_CCCL_DEVICE_API inline CpAsyncOobInfo prepareCpAsyncOob(const Tp* ptrGmem, uint32_t sizeElem)
{
  // We will copy from [ptrGmemBase, ptrGmemEnd). Both pointers have to be 16B
  // aligned. We align ptrGmemBase down and ptrGmemEnd up.
  const Tp* ptrGmemBase = cuda::align_down(ptrGmem, std::size_t(16));
  const Tp* ptrGmemEnd  = cuda::align_up(ptrGmem + sizeElem, std::size_t(16));

  // Compute the final copy size in bytes. It can be either sizeElem or sizeElem + 16 / sizeof(T).
  uint32_t copySizeBytes = static_cast<uint32_t>(sizeof(Tp) * (ptrGmemEnd - ptrGmemBase));
  // The offset in elements to the first valid element in shared memory.
  // ptrSmem + smemOffsetElem will point to the element copied from ptrGmem.
  uint32_t smemOffsetElem = static_cast<uint32_t>(ptrGmem - ptrGmemBase);

  return CpAsyncOobInfo{(void*) ptrGmemBase, copySizeBytes, smemOffsetElem};
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
      cpAsyncOobInfo.ptrGmemBase,
      cpAsyncOobInfo.copySizeBytes,
      ptrBar);
  }
  refDestSmem.squadIncreaseTxCount(squad, cpAsyncOobInfo.copySizeBytes);
}

template <typename OutputT>
_CCCL_DEVICE_API inline void
squadStoreBulkSync(const Squad& squad, OutputT* ptrOut, OutputT* srcSmem, uint32_t sizeBytes)
{
  // Acquire shared memory in async proxy
  if (squad.isLeaderWarp())
  {
    // Perform fence.proxy.async with full warp to avoid BSSY+BSYNC
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
  }
  if (squad.isLeaderThread())
  {
    // Issue TMA store
    ::cuda::ptx::cp_async_bulk(::cuda::ptx::space_global, ::cuda::ptx::space_shared, ptrOut, srcSmem, sizeBytes);
  }
  // Commit and wait for store to have completed reading from shared memory
  ::cuda::ptx::cp_async_bulk_commit_group();
  ::cuda::ptx::cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<0>{});
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

  tmp_state_t<AccumT> tmp{scanState, sum};
  uint64_t data = *reinterpret_cast<uint64_t*>(&tmp);
  asm("st.relaxed.gpu.global.b64 [%0], %1;" : : "l"(__cvta_generic_to_global(dst)), "l"(data) : "memory");
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
      uint64_t data;
      asm("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(data) : "l"(__cvta_generic_to_global(src)) : "memory");
      outTmpStates[i] = *reinterpret_cast<tmp_state_t<AccumT>*>(&data);
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
// Definition of squads

_CCCL_GLOBAL_CONSTANT SquadDesc squadReduce{/*squadIdx=*/0, /*numWarps=*/4};
_CCCL_GLOBAL_CONSTANT SquadDesc squadScanStore{/*squadIdx=*/1, /*numWarps=*/4};
_CCCL_GLOBAL_CONSTANT SquadDesc squadLoad{/*squadIdx=*/2, /*numWarps=*/1};
_CCCL_GLOBAL_CONSTANT SquadDesc squadSched{/*squadIdx=*/3, /*numWarps=*/1};
_CCCL_GLOBAL_CONSTANT SquadDesc squadLookback{/*squadIdx=*/4, /*numWarps=*/1};

_CCCL_GLOBAL_CONSTANT SquadDesc scanSquads[] = {
  squadReduce,
  squadScanStore,
  squadLoad,
  squadSched,
  squadLookback,
};
// Struct holding all scan kernel resources

template <int tileSize, typename InputT, typename OutputT, typename AccumT>
struct ScanResources
{
  static constexpr int elemPerThread = tileSize / squadReduce.threadCount();

  // Handle unaligned loads. We have 16 extra bytes of padding in every stage
  // for squadLoadBulk.
  using InT               = InputT[squadReduce.threadCount() * elemPerThread + 16 / sizeof(InputT)];
  using SumThreadAndWarpT = AccumT[squadReduce.threadCount() + squadReduce.warpCount()];

  SmemResource<InT> smemIn;
  OutputT* smemOut;
  SmemResource<uint4> smemNextBlockIdx;
  SmemResource<AccumT> smemSumExclusiveCta;
  SmemResource<SumThreadAndWarpT> smemSumThreadAndWarp;
};
// Function to allocate resources.

template <int tileSize, typename InputT, typename OutputT, typename AccumT>
[[nodiscard]] _CCCL_API ScanResources<tileSize, InputT, OutputT, AccumT> allocResources(
  SyncHandler& syncHandler, SmemAllocator& smemAllocator, const scanKernelParams<InputT, OutputT, AccumT>& params)
{
  using ScanResourcesT    = ScanResources<tileSize, InputT, OutputT, AccumT>;
  using InT               = typename ScanResourcesT::InT;
  using SumThreadAndWarpT = typename ScanResourcesT::SumThreadAndWarpT;

  // If numBlockIdxStages is one less than the number of stages, we find a small
  // speedup compared to setting it equal to num_stages. Not sure why.
  int numBlockIdxStages = params.numStages - 1;
  // Ensure we have at least 1 stage
  numBlockIdxStages = numBlockIdxStages < 1 ? 1 : numBlockIdxStages;

  // We do not need too many sumExclusiveCta stages. The lookback warp is the
  // bottleneck. As soon as it produces a new value, it will be consumed by the
  // scanStore squad, releasing the stage.
  int numSumExclusiveCtaStages = 2;

  ScanResources<tileSize, InputT, OutputT, AccumT> res{
    SmemResource<InT>(syncHandler, smemAllocator, stages(params.numStages)),
    reinterpret_cast<OutputT*>(smemAllocator.alloc(sizeof(OutputT) * tileSize, alignof(OutputT))),
    SmemResource<uint4>(syncHandler, smemAllocator, stages(numBlockIdxStages)),
    SmemResource<AccumT>(syncHandler, smemAllocator, stages(numSumExclusiveCtaStages)),
    SmemResource<SumThreadAndWarpT>(syncHandler, smemAllocator, stages(params.numStages)),
  };
  // asdfasdf
  res.smemIn.addPhase(syncHandler, smemAllocator, squadLoad);
  res.smemIn.addPhase(syncHandler, smemAllocator, {squadReduce, squadScanStore});

  res.smemNextBlockIdx.addPhase(syncHandler, smemAllocator, squadSched);
  res.smemNextBlockIdx.addPhase(syncHandler, smemAllocator, scanSquads);

  res.smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, squadLookback);
  res.smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, squadScanStore);

  res.smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, squadReduce);
  res.smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, squadScanStore);

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
template <int numLookbackTiles,
          int tile_size,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT,
          bool ForceInclusive>
_CCCL_DEVICE_API inline void kernelBody(
  Squad squad,
  SpecialRegisters specialRegisters,
  const scanKernelParams<InputT, OutputT, AccumT>& params,
  ScanOpT scan_op,
  InitValueT init_value)
{
  ////////////////////////////////////////////////////////////////////////////////
  // Resources
  ////////////////////////////////////////////////////////////////////////////////
  SyncHandler syncHandler{};
  SmemAllocator smemAllocator{};

  ScanResources<tile_size, InputT, OutputT, AccumT> res =
    allocResources<tile_size, InputT, OutputT, AccumT>(syncHandler, smemAllocator, params);

  syncHandler.clusterInitSync(specialRegisters);

  // Inclusive scan if no init_value type is provided
  static constexpr bool hasInit     = !::cuda::std::is_same_v<InitValueT, NullType>;
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
    SmemStage stageInOut            = res.smemIn.popStage();
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

    if (squad == squadSched)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load next tile index
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
      squadGetNextBlockIdx(squad, refNextBlockIdxW);
    }

    size_t idxTileBase      = idxTile * size_t(tile_size);
    size_t copyNumElem      = cuda::std::min(params.numElem - idxTileBase, size_t(tile_size));
    CpAsyncOobInfo loadInfo = prepareCpAsyncOob(params.ptrIn + idxTile * size_t(tile_size), copyNumElem);

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
      ////////////////////////////////////////////////////////////////////////////////
      // Load tile from shared memory
      ////////////////////////////////////////////////////////////////////////////////
      AccumT regThreadSum{};
      AccumT regWarpSum{};
      {
        // Acquire phaseInOutRW in this short scope
        SmemRef refInOutRW = phaseInOutRW.acquireRef();
        // Load data
        constexpr int elemPerThread    = tile_size / squadReduce.threadCount();
        AccumT regInput[elemPerThread] = {AccumT{}};
        // Handle unaligned refInOutRW.data() + loadInfo.smemOffsetElem points to the first element of the tile.
        squadLoadSmem(squad, regInput, &refInOutRW.data()[0] + loadInfo.smemOffsetElem);

        ////////////////////////////////////////////////////////////////////////////////
        // Reduce across thread and warp
        ////////////////////////////////////////////////////////////////////////////////
        regThreadSum = threadReduce(regInput, scan_op);
        regWarpSum   = warpReduce(regThreadSum, scan_op);
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
      AccumT regSquadSum{};
      if constexpr (hasInit)
      {
        if (idxTile == 0)
        {
          regSquadSum = scan_op(init_value, refSumThreadAndWarpW.data()[squadReduce.threadCount()]);
        }
      }

      for (int i = (idxTile == 0) && hasInit; i < squadReduce.warpCount(); ++i)
      {
        regSquadSum = scan_op(regSquadSum, refSumThreadAndWarpW.data()[squadReduce.threadCount() + i]);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store private sum for lookback
      ////////////////////////////////////////////////////////////////////////////////
      if (squad.isLeaderThread() == 0)
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

      constexpr int numTmpStatesPerThread = numLookbackTiles / 32;
      static_assert(numLookbackTiles % 32 == 0, "numLookbackTiles must be a multiple of 32");

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
      constexpr int elem_per_thread = tile_size / squadScanStore.threadCount();
      static_assert(tile_size % squadScanStore.threadCount() == 0);

      // Sum of all threads up to but not including this one
      AccumT sumExclusive{};

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across warp and thread sums
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Acquire refSumExclusiveCtaW briefly
        SmemRef refSumThreadAndWarpR = phaseSumThreadAndWarpR.acquireRef();
        // Add the sums of the preceding warps in this CTA to the cumulative
        // sum. These sums have been calculated in squadReduce. We need
        // the reduce and scan squads to be the same size to do this.
        static_assert(squadReduce.warpCount() == squadScanStore.warpCount());

        for (int i = 0; i < squad.warpCount(); ++i)
        {
          // We want a predicated unrolled loop here.
          if (i < squad.warpRank())
          {
            sumExclusive = scan_op(sumExclusive, refSumThreadAndWarpR.data()[squadReduce.threadCount() + i]);
          }
        }
        // Add the sums of preceding threads in this warp to the cumulative sum.
        // Lane 0 reads invalid data.
        AccumT regSumThread = refSumThreadAndWarpR.data()[squad.threadRank()];
        // Perform scan of thread sums
        AccumT sumExclusiveIntraWarp = warpScanExclusive(regSumThread, scan_op);
        sumExclusive                 = scan_op(sumExclusive, sumExclusiveIntraWarp);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Include sum of previous CTAs
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Briefly acquire refSumExclusiveCtaR
        SmemRef refSumExclusiveCtaR = phaseSumExclusiveCtaR.acquireRef();
        // Add the sums of preceding CTAs to the cumulative sum.
        AccumT regSumExclusiveCta = refSumExclusiveCtaR.data();
        sumExclusive              = scan_op(sumExclusive, regSumExclusiveCta);
      }

      if constexpr (hasInit)
      {
        if (idxTile == 0)
        {
          sumExclusive = scan_op(static_cast<AccumT>(init_value), sumExclusive);
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across elements allocated to this thread
      ////////////////////////////////////////////////////////////////////////////////
      AccumT regSumInclusive[elem_per_thread] = {{}};

      // Acquire refInOut for remainder of scope.
      {
        SmemRef refInOutRW = phaseInOutRW.acquireRef();
        // Handle unaligned refInOutRW.data() + loadInfo.smemOffsetElem points to
        // the first element of the tile.
        squadLoadSmem(squad, regSumInclusive, &refInOutRW.data()[0] + loadInfo.smemOffsetElem);
        refInOutRW.setFenceLdsToAsyncProxy();
      }

      // Perform inclusive scan of register array in current thread.
      if constexpr (isInclusive)
      {
        ThreadScanInclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive);
      }
      else
      {
        ThreadScanExclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      // Sync before storing to avoid data races.
      squad.syncThreads();

      squadStoreSmem(squad, res.smemOut, regSumInclusive);
      // We do *not* release refSmemInOut here, because we will issue a TMA
      // instruction below. Instead, we issue a squad-local syncthreads +
      // fence.proxy.async to sync the shared memory writes with the TMA store.
      squad.syncThreads();

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to global memory using TMA
      ////////////////////////////////////////////////////////////////////////////////
      squadStoreBulkSync(squad,
                         params.ptrOut + idxTile * size_t(tile_size),
                         res.smemOut,
                         static_cast<uint32_t>(sizeof(OutputT) * tile_size));
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

template <int tile_size,
          int numLookbackTiles,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT,
          bool ForceInclusive>
__launch_bounds__(squadCountThreads(scanSquads), 1) __global__ void scan(
  const __grid_constant__ scanKernelParams<InputT, OutputT, AccumT> params, ScanOpT scan_op, InitValueT init_value)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
      // Cache special registers at start of kernel
      SpecialRegisters specialRegisters = getSpecialRegisters();

      // Dispatch for warp-specialization
      squadDispatch(specialRegisters, scanSquads, [&](Squad squad) {
        kernelBody<numLookbackTiles, tile_size, InputT, OutputT, AccumT, ScanOpT, InitValueT, ForceInclusive>(
          squad, specialRegisters, params, ::cuda::std::move(scan_op), init_value);
      });))
}

template <int tile_size, typename AccumT>
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
