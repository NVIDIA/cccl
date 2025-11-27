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

#include <cub/device/dispatch/kernels/warpspeed/allocators/SmemAllocator.h> // SmemAllocator
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemRef.cuh> // SmemRef
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemResource.cuh> // SmemResource
#include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh> // SpecialRegisters
#include <cub/device/dispatch/kernels/warpspeed/squad/Squad.h> // squadDispatch, ...
#include <cub/device/dispatch/kernels/warpspeed/values.h> // stages

#include <cuda/ptx>

#include <cassert>

#include <cudaTypedefs.h>

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

struct tmp_state_t
{
  scan_state state;
  int value;
};

_CCCL_DEVICE_API inline void squadGetNextBlockIdx(const Squad& squad, SmemRef<uint4>& refDestSmem)
{
  if (squad.isLeaderThread())
  {
    ::cuda::ptx::clusterlaunchcontrol_try_cancel(&refDestSmem.data(), refDestSmem.ptrCurBarrierRelease());
  }
  refDestSmem.squadIncreaseTxCount(squad, refDestSmem.sizeBytes());
}

template <typename T>
_CCCL_DEVICE_API inline void squadLoadTma(const Squad& squad, SmemRef<T>& refDestSmem, const int* ptrIn)
{
  void* ptrSmem    = refDestSmem.data();
  uint64_t* ptrBar = refDestSmem.ptrCurBarrierRelease();

  if (squad.isLeaderThread())
  {
    ::cuda::ptx::cp_async_bulk(
      ::cuda::ptx::space_cluster, ::cuda::ptx::space_global, ptrSmem, ptrIn, refDestSmem.sizeBytes(), ptrBar);
  }
  refDestSmem.squadIncreaseTxCount(squad, refDestSmem.sizeBytes());
}

template <typename T>
_CCCL_DEVICE_API inline void squadStoreTmaSync(const Squad& squad, int* ptrOut, SmemRef<T>& refSrcSmem)
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
    ::cuda::ptx::cp_async_bulk(
      ::cuda::ptx::space_global, ::cuda::ptx::space_shared, ptrOut, refSrcSmem.data(), refSrcSmem.sizeBytes());
  }
  // Commit and wait for store to have completed reading from shared memory
  ::cuda::ptx::cp_async_bulk_commit_group();
  ::cuda::ptx::cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<0>{});
}

template <typename Tp, int elemPerThread>
_CCCL_DEVICE_API inline void
squadLoadSmem(Squad squad, Tp (&outReg)[elemPerThread], const Tp * smemBuf)
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    outReg[i] = smemBuf[squad.threadRank() * elemPerThread + i];
  }
}

template <typename Tp, int elemPerThread>
_CCCL_DEVICE_API inline void
squadStoreSmem(Squad squad, Tp* smemBuf, const Tp (&inReg)[elemPerThread])
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    smemBuf[squad.threadRank() * elemPerThread + i] = inReg[i];
  }
}

template <int elemPerThread>
_CCCL_DEVICE_API inline int threadReduce(const int (&regInput)[elemPerThread])
{
  int sum = 0;
  for (int i = 0; i < elemPerThread; ++i)
  {
    sum += regInput[i];
  }
  return sum;
}

_CCCL_DEVICE_API inline int warpReduce(const int input)
{
  int sum = __reduce_add_sync(~0, input);
  return sum;
}

_CCCL_DEVICE_API inline int warpScanExclusive(const int regInput)
{
  // Warp scan of cumsum
  int sumInclusive = regInput;
  for (int i = 1; i < 32; i *= 2)
  {
    bool shlfInRange;
    int tmp = shfl_sync_up(shlfInRange, sumInclusive, i, 0, ~0);
    if (shlfInRange)
    {
      sumInclusive += tmp;
    }
  }
  // Make scan exclusive
  bool shlfInRange;
  int tmp          = shfl_sync_up(shlfInRange, sumInclusive, 1, 0, ~0);
  int sumExclusive = shlfInRange ? tmp : 0;

  return sumExclusive;
}

template <int elemPerThread>
_CCCL_DEVICE_API inline void threadScanInclusive(int (&regArray)[elemPerThread])
{
  int sumInclusive = 0;
  for (int i = 0; i < elemPerThread; ++i)
  {
    sumInclusive += regArray[i];
    regArray[i] = sumInclusive;
  }
}

template <int elemPerThread>
_CCCL_DEVICE_API inline void threadAdd(int (&reg)[elemPerThread], int value)
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    reg[i] += value;
  }
}

_CCCL_DEVICE_API inline void storeLookback(tmp_state_t* ptrTmpBuffer, int idxTile, scan_state scanState, int sum)
{
  tmp_state_t* dst = ptrTmpBuffer + idxTile;

  tmp_state_t tmp{scanState, sum};
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
template <int numTmpStatesPerThread>
_CCCL_DEVICE_API inline void warpLoadLookback(
  int laneIdx,
  tmp_state_t (&outTmpStates)[numTmpStatesPerThread],
  tmp_state_t* ptrTmpBuffer,
  int idxTileCur,
  int idxTileNext)
{
  for (int i = 0; i < numTmpStatesPerThread; ++i)
  {
    int idxTileLookback = idxTileCur + 32 * (i + 1) - laneIdx;
    if (idxTileLookback < idxTileNext)
    {
      tmp_state_t* src = ptrTmpBuffer + idxTileLookback;
      uint64_t data;
      asm("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(data) : "l"(__cvta_generic_to_global(src)) : "memory");
      outTmpStates[i] = *reinterpret_cast<tmp_state_t*>(&data);
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
template <int numTmpStatesPerThread>
_CCCL_DEVICE_API inline int warpIncrementalLookback(
  SpecialRegisters specialRegisters,
  tmp_state_t* ptrTmpBuffer,
  const int idxTilePrev,
  const int sumExclusiveCtaPrev,
  const int idxTileNext)
{
  const int laneIdx    = specialRegisters.laneIdx;
  const int lanemaskEq = ::cuda::ptx::get_sreg_lanemask_eq();

  int idxTileCur         = idxTilePrev;
  int sumExclusiveCtaCur = sumExclusiveCtaPrev;

  while (idxTileCur < idxTileNext)
  {
    tmp_state_t regTmpStates[numTmpStatesPerThread] = {{EMPTY, 0}};
    warpLoadLookback(laneIdx, regTmpStates, ptrTmpBuffer, idxTileCur, idxTileNext);

    for (int idx = 0; idx < numTmpStatesPerThread; ++idx)
    {
      // Bitmask with a 1 bit in the position of the current lane if current
      // lane has an XXX state;
      int laneIsEmpty   = lanemaskEq * (regTmpStates[idx].state == EMPTY);
      int laneIsCumSum  = lanemaskEq * (regTmpStates[idx].state == CUM_SUM);
      int laneIsPrivSum = lanemaskEq * (regTmpStates[idx].state == PRIV_SUM);
      // Bitmask with 1 bits indicating which lane has an XX state.
      int warpIsEmpty   = __reduce_or_sync(~0, laneIsEmpty);
      int warpIsCumSum  = __reduce_or_sync(~0, laneIsCumSum);
      int warpIsPrivSum = __reduce_or_sync(~0, laneIsPrivSum);

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
      int localSum            = 0;
      int maskSumParticipants = warpRightMostCumSum | maskRightOfCumSum;

      if ((maskSumParticipants & lanemaskEq) != 0)
      {
        localSum = regTmpStates[idx].value;
      }
      localSum = __reduce_add_sync(~0, localSum);

      if (warpIsCumSum == 0)
      {
        sumExclusiveCtaCur += localSum;
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

struct scanKernelParams
{
  int* ptrIn;
  int* ptrOut;
  tmp_state_t* ptrTmp;
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

template <int tileSize>
struct ScanResources
{
  static constexpr int elemPerThread = tileSize / squadReduce.threadCount();

  using InOutT            = int[squadReduce.threadCount() * elemPerThread];
  using SumThreadAndWarpT = int[squadReduce.threadCount() + squadReduce.warpCount()];

  SmemResource<InOutT> smemInOut;
  SmemResource<uint4> smemNextBlockIdx;
  SmemResource<int> smemSumExclusiveCta;
  SmemResource<SumThreadAndWarpT> smemSumThreadAndWarp;
};
// Function to allocate resources.

template <int tileSize>
_CCCL_API ScanResources<tileSize>
allocResources(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const scanKernelParams& params)
{
  using ScanResourcesT    = ScanResources<tileSize>;
  using InOutT            = typename ScanResourcesT::InOutT;
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

  ScanResources<tileSize> res{
    .smemInOut            = makeSmemResource<InOutT>(syncHandler, smemAllocator, stages(params.numStages)),
    .smemNextBlockIdx     = makeSmemResource<uint4>(syncHandler, smemAllocator, stages(numBlockIdxStages)),
    .smemSumExclusiveCta  = makeSmemResource<int>(syncHandler, smemAllocator, stages(numSumExclusiveCtaStages)),
    .smemSumThreadAndWarp = makeSmemResource<SumThreadAndWarpT>(syncHandler, smemAllocator, stages(params.numStages)),
  };
  // asdfasdf
  res.smemInOut.addPhase(syncHandler, smemAllocator, squadLoad);
  res.smemInOut.addPhase(syncHandler, smemAllocator, {squadReduce, squadScanStore});

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
template <int numLookbackTiles, int tile_size>
_CCCL_DEVICE_API inline void kernelBody(Squad squad, SpecialRegisters specialRegisters, const scanKernelParams& params)
{
  ////////////////////////////////////////////////////////////////////////////////
  // Resources
  ////////////////////////////////////////////////////////////////////////////////
  SyncHandler syncHandler{};
  SmemAllocator smemAllocator{};

  ScanResources<tile_size> res = allocResources<tile_size>(syncHandler, smemAllocator, params);

  syncHandler.clusterInitSync(specialRegisters);

  ////////////////////////////////////////////////////////////////////////////////
  // Pre-loop
  ////////////////////////////////////////////////////////////////////////////////

  // Start with the tile indicated by blockIdx.x
  int idxTile = specialRegisters.blockIdxX;
  // Lookback-specific variables:
  int idxTilePrev         = -1;
  int sumExclusiveCtaPrev = 0;

  ////////////////////////////////////////////////////////////////////////////////
  // Loop over tiles
  ////////////////////////////////////////////////////////////////////////////////
#pragma unroll 1
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

    if (squad == squadSched)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load next tile index
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
      squadGetNextBlockIdx(squad, refNextBlockIdxW);
    }

    if (squad == squadLoad)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load current tile
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refInOutW = phaseInOutW.acquireRef();
      squadLoadTma(squad, refInOutW, params.ptrIn + idxTile * size_t(tile_size));
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
      int regThreadSum = 0;
      int regWarpSum   = 0;
      {
        // Acquire phaseInOutRW in this short scope
        SmemRef refInOutRW = phaseInOutRW.acquireRef();
        // Load data
        constexpr int elemPerThread = tile_size / squadReduce.threadCount();
        int regInput[elemPerThread] = {0};
        squadLoadSmem(squad, regInput, refInOutRW.data());

        ////////////////////////////////////////////////////////////////////////////////
        // Reduce across thread and warp
        ////////////////////////////////////////////////////////////////////////////////
        regThreadSum = threadReduce(regInput);
        regWarpSum   = warpReduce(regThreadSum);
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
      int regSquadSum = 0;
      for (int i = 0; i < squadReduce.warpCount(); ++i)
      {
        regSquadSum += refSumThreadAndWarpW.data()[squadReduce.threadCount() + i];
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

      int regSumExclusiveCta = warpIncrementalLookback<numTmpStatesPerThread>(
        specialRegisters, params.ptrTmp, idxTilePrev, sumExclusiveCtaPrev, idxTile);
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
      int sumExclusive = 0;

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
            sumExclusive += refSumThreadAndWarpR.data()[squadReduce.threadCount() + i];
          }
        }
        // Add the sums of preceding threads in this warp to the cumulative sum.
        // Lane 0 reads invalid data.
        int regSumThread = refSumThreadAndWarpR.data()[squad.threadRank()];
        // Perform scan of thread sums
        int sumExclusiveIntraWarp = warpScanExclusive(regSumThread);
        sumExclusive += sumExclusiveIntraWarp;
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Include sum of previous CTAs
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Briefly acquire refSumExclusiveCtaR
        SmemRef refSumExclusiveCtaR = phaseSumExclusiveCtaR.acquireRef();
        // Add the sums of preceding CTAs to the cumulative sum.
        int regSumExclusiveCta = refSumExclusiveCtaR.data();
        sumExclusive += regSumExclusiveCta;
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across elements allocated to this thread
      ////////////////////////////////////////////////////////////////////////////////
      int regSumInclusive[elem_per_thread] = {0};

      // Acquire refInOut for remainder of scope.
      SmemRef refInOutRW = phaseInOutRW.acquireRef();
      squadLoadSmem(squad, regSumInclusive, refInOutRW.data());

      // Perform inclusive scan of register array in current thread.
      regSumInclusive[0] += sumExclusive;
      threadScanInclusive(regSumInclusive);

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      squadStoreSmem(squad, refInOutRW.data(), regSumInclusive);
      // We do *not* release refSmemInOut here, because we will issue a TMA
      // instruction below. Instead, we issue a squad-local syncthreads +
      // fence.proxy.async to sync the shared memory writes with the TMA store.
      squad.syncThreads();

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to global memory using TMA
      ////////////////////////////////////////////////////////////////////////////////
      squadStoreTmaSync(squad, params.ptrOut + idxTile * size_t(tile_size), refInOutRW);
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
}

template <int tile_size, int numLookbackTiles>
__launch_bounds__(squadCountThreads(scanSquads), 1) __global__
  void scan(const __grid_constant__ scanKernelParams params)
{
  // Cache special registers at start of kernel
  SpecialRegisters specialRegisters = getSpecialRegisters();

  // Dispatch for warp-specialization
  squadDispatch(specialRegisters, scanSquads, [&](auto squad) {
    kernelBody<numLookbackTiles, tile_size>(squad, specialRegisters, params);
  });
}

template <int tile_size>
__launch_bounds__(128) __global__ void initTmpStates(int* x, tmp_state_t* tmp, int* out, size_t len, bool do_check)
{
  int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (tile_id * tile_size > len)
  {
    return;
  }
  tmp[tile_id] = {EMPTY, 0};

  if (do_check)
  {
    for (int i = 0; i < tile_size; ++i)
    {
      size_t idx = tile_size * tile_id + i;
      if (idx < len)
      {
        x[idx] = idx;
      }
      else
      {
        return;
      }
    }
  }
}
} // namespace detail::scan

CUB_NAMESPACE_END
