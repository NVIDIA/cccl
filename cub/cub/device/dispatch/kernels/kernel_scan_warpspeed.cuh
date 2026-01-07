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

#include <cub/device/dispatch/kernels/warpspeed/allocators/SmemAllocator.h>
#include <cub/device/dispatch/kernels/warpspeed/look_ahead.h>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemRef.cuh>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemResource.cuh>
#include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh>
#include <cub/device/dispatch/kernels/warpspeed/squad/Squad.h>
#include <cub/device/dispatch/kernels/warpspeed/values.h>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__memory/align_down.h>
#include <cuda/__memory/align_up.h>
#include <cuda/ptx>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename InputT, typename OutputT, typename AccumT>
struct scanKernelParams
{
  const InputT* ptrIn;
  OutputT* ptrOut;
  tile_state_t<AccumT>* ptrTileStates;
  size_t numElem;
  int numStages;
};

// Struct holding all scan kernel resources
template <typename WarpspeedPolicy, typename InputT, typename OutputT, typename AccumT>
struct ScanResources
{
  union InOutT
  {
    // Handle unaligned loads. We have at least 16 extra bytes of padding in every stage for squadLoadBulk.
    InputT in[WarpspeedPolicy::tile_size + ::cuda::ceil_div(16, sizeof(InputT))];
    OutputT out[sizeof(in) / sizeof(OutputT)];
  };
  static_assert(alignof(InOutT) >= alignof(InputT));
  static_assert(alignof(InOutT) >= alignof(OutputT));
  using SumThreadAndWarpT =
    AccumT[WarpspeedPolicy::squadReduce().threadCount() + WarpspeedPolicy::squadReduce().warpCount()];

  SmemResource<InOutT> smemInOut; // will also be used to stage the output (as OutputT) for the bulk copy
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
  using InOutT            = typename ScanResourcesT::InOutT;
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
    SmemResource<InOutT>(syncHandler, smemAllocator, Stages{numStages}),
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

#if __cccl_ptx_isa >= 860

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
  const Tp* ptrGmemStartAlignDown = cuda::align_down(ptrGmem, ::cuda::std::size_t(16));
  const Tp* ptrGmemStartAlignUp   = cuda::align_up(ptrGmem, ::cuda::std::size_t(16));
  const Tp* ptrGmemEnd            = ptrGmem + sizeElem;
  const Tp* ptrGmemEndAlignUp     = cuda::align_up(ptrGmemEnd, ::cuda::std::size_t(16));
  const Tp* ptrGmemEndAlignDown   = cuda::align_down(ptrGmemEnd, ::cuda::std::size_t(16));

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
  void* ptrSmem    = refDestSmem.data().in;
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
    if (beginIndex <= elem_idx && elem_idx < endIndex)
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

namespace ptx = cuda::ptx;

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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void kernelBody(
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

  constexpr int tile_size            = WarpspeedPolicy::tile_size;
  constexpr int num_look_ahead_items = WarpspeedPolicy::num_look_ahead_items;

  // We might try to instantiate the kernel with hughe types which would lead to a small tile size. Ensure its never 0
  constexpr int elemPerThread = WarpspeedPolicy::items_per_thread;
  static_assert(elemPerThread * squadReduce.threadCount() == tile_size, "Invalid tuning policy");

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
  int idxTilePrev = 0;
  AccumT sumExclusiveCtaPrev; // only valid in squadLookback lane_0
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
        squadLoadSmem(squad, regInput, &refInOutRW.data().in[0] + loadInfo.smemStartOffsetElem);

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
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 1; i < squadReduce.warpCount(); ++i)
        {
          if (i < valid_warps)
          {
            regSquadSum = scan_op(regSquadSum, refSumThreadAndWarpW.data()[squadReduce.threadCount() + i]);
          }
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
        storeTileAggregate(params.ptrTileStates, TILE_AGGREGATE, regSquadSum, idxTile);
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

      if (!is_first_tile)
      {
        AccumT regSumExclusiveCta = warpIncrementalLookback<num_look_ahead_items>(
          specialRegisters, params.ptrTileStates, idxTilePrev, sumExclusiveCtaPrev, idxTile, scan_op);
        if (squad.isLeaderThread())
        {
          refSumExclusiveCtaW.data() = regSumExclusiveCta;
        }
        sumExclusiveCtaPrev = regSumExclusiveCta;
        idxTilePrev         = idxTile;
      }
    }

    if (squad == squadScanStore)
    {
      static_assert(tile_size % squadScanStore.threadCount() == 0);

      // Sum of all threads up to but not including this one
      AccumT sumExclusive;

      ////////////////////////////////////////////////////////////////////////////////
      // Include warp and thread sum of current tile
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Acquire refSumThread briefly
        SmemRef refSumThreadAndWarpR = phaseSumThreadAndWarpR.acquireRef();
        // Add the sums of the preceding warps in this CTA to the cumulative
        // sum. These sums have been calculated in squadReduce(). We need
        // the reduce and scan squads to be the same size to do this.
        static_assert(squadReduce.warpCount() == squadScanStore.warpCount());

        // Include warp sums
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < squadScanStore.warpCount(); ++i)
        {
          // We want a predicated unrolled loop here.
          if (i < squad.warpRank())
          {
            if (i == 0)
            {
              // The first iteration initializes sumExclusive
              sumExclusive = refSumThreadAndWarpR.data()[squadReduce.threadCount()];
            }
            else
            {
              // If loaded value belongs to previous warp, include it in sumExclusive.
              sumExclusive = scan_op(sumExclusive, refSumThreadAndWarpR.data()[squadReduce.threadCount() + i]);
            }
          }
        }
        // sumExclusive contains the sum of previous warps.
        // It has a valid value in
        // - tile*::warp{1,2, ..}      (sum of previous warps)
        //
        // It it not yet initialized in
        // - tile*::warp0

        // Add the sums of preceding threads in this warp to the cumulative sum.
        // We perform an exclusive scan of:
        //
        //   {sumT0, sumT1, ..., sumT30, sumT31 }
        //
        // As a result:
        // - lane0 has undefined value
        // - lane1 has sumT0
        // - ...
        // - lane31 has sumT0 + ... + sumT30
        //
        // For lane1, ..., 31, we add the result to sumExclusive.
        //
        // If the warp contains partial data, we pass invalid elements to
        // scan_op, and sumExclusiveIntraWarp is invalid when the inputs were
        // invalid.
        AccumT regSumThread          = refSumThreadAndWarpR.data()[squad.threadRank()];
        AccumT sumExclusiveIntraWarp = warpScanExclusive(regSumThread, scan_op);

        if (squad.warpRank() == 0)
        {
          // Warp0 does not yet have a valid value for sumExclusive. We set it
          // here. This ensures that lane1,..,31 of tile0::warp0 have a valid
          // value for sumExclusive.
          sumExclusive = sumExclusiveIntraWarp;
        }
        else if (specialRegisters.laneIdx != 0)
        {
          // lane0 has an undefined value for sumIntraWarp. Other lanes update
          // sumExclusive using sumIntraWarp.
          sumExclusive = scan_op(sumExclusive, sumExclusiveIntraWarp);
        }
      }
      // sumExclusive contains the sum of previous warps and sum of previous threads.
      //
      // - tile*::warp0::lane{1, .., 31}  (sum of previous threads)
      // - tile*::warp{1,2, ..}           (sum of previous warps + sum of previous threads)
      //
      // It has an undefined value in
      // - tile*::warp0::lane0

      ////////////////////////////////////////////////////////////////////////////////
      // Include sum of previous tiles
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Briefly acquire refSumExclusiveCtaR (we have to do this for the first tile as well to prevent a hang)
        SmemRef refSumExclusiveCtaR = phaseSumExclusiveCtaR.acquireRef();

        if (!is_first_tile)
        {
          // Add the sums of preceding CTAs to the cumulative sum.
          AccumT regSumExclusiveCta = refSumExclusiveCtaR.data();
          // sumExclusive is invalid in warp_0/thread_0, so only include it in other threads/warps
          sumExclusive = squad.threadRank() == 0 ? regSumExclusiveCta : scan_op(sumExclusive, regSumExclusiveCta);
        }
      }

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
      // sumExclusive contains the following values:
      //
      // - tile0::warp0::lane0            (init_value)
      // - tile0::warp0::lane{1, .., 31}  (init_value + sum of previous threads)
      // - tile0::warp{1,2, ..}           (init_value + sum of previous warps + sum of previous threads)
      // - tile*::warp0::lane0            (sum of previous CTAs)
      // - tile*::warp0::lane{1, .., 31}  (sum of previous CTAs + sum of previous threads)
      // - tile*::warp{1,2, ..}           (sum of previous CTAs + sum of previous warps + sum of previous threads)
      //
      // If no init value is provided, then sumExclusive has an undefined value in
      // - tile0::warp0::lane0

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across elements allocated to this thread
      ////////////////////////////////////////////////////////////////////////////////
      AccumT regSumInclusive[elemPerThread] = {{}};

      // Acquire refInOut for remainder of scope.
      SmemRef refInOutRW = phaseInOutRW.acquireRef();

      // We are always loading a full tile even for the last one. That will call scan_op on invalid data
      // loading partial tiles here regresses perf for about 10-15%
      squadLoadSmem(squad, regSumInclusive, &refInOutRW.data().in[0] + loadInfo.smemStartOffsetElem);

      // Perform inclusive scan of register array in current thread.
      // warp_0/thread_0 in the first tile when there is no initial value, we MUST NOT use sumExclusive
      const bool use_prefix = hasInit ? true : !(is_first_tile && squad.threadRank() == 0);
      if constexpr (isInclusive)
      {
        ThreadScanInclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive, use_prefix);
      }
      else
      {
        ThreadScanExclusive(regSumInclusive, regSumInclusive, scan_op, sumExclusive, use_prefix);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      // Sync before storing to avoid data races on SMEM
      squad.syncThreads();

      OutputT* smem_output_tile = refInOutRW.data().out;
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
        const int elem_per_chunk = ::cuda::std::size(refInOutRW.data().out);
        for (int chunk_offset = 0; chunk_offset < static_cast<int>(valid_items); chunk_offset += elem_per_chunk)
        {
          const int chunk_size     = ::cuda::std::min(static_cast<int>(valid_items) - chunk_offset, elem_per_chunk);
          CpAsyncOobInfo storeInfo = prepareCpAsyncOob(params.ptrOut + idxTileBase + chunk_offset, chunk_size);
          OutputT* smemBuf         = smem_output_tile + storeInfo.smemStartOffsetElem;

          // only stage elements of the current chunk to SMEM
          // storeInfo.smemStartOffsetElem < 16 and smem_output_tile contains extra 16 bytes, so we should fit
          _CCCL_ASSERT((storeInfo.smemStartOffsetElem + elem_per_chunk) * sizeof(OutputT) <= res.smemInOut.mSizeBytes,
                       "");
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

#endif // __cccl_ptx_isa >= 860

template <typename ActivePolicy, class = void>
inline constexpr int get_scan_block_threads = 1;

template <typename ActivePolicy>
inline constexpr int get_scan_block_threads<ActivePolicy, ::cuda::std::void_t<typename ActivePolicy::WarpspeedPolicy>> =
  ActivePolicy::WarpspeedPolicy::num_total_threads;

template <typename WarpspeedPolicy,
          bool ForceInclusive,
          typename RealInitValueT,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_scan_lookahead_body(
  const scanKernelParams<InputT, OutputT, AccumT> params, ScanOpT scan_op, const InitValueT& init_value)
{
#if __cccl_ptx_isa >= 860
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

  squadDispatch(specialRegisters, scanSquads, [&](Squad squad) {
    kernelBody<WarpspeedPolicy, InputT, OutputT, AccumT, ScanOpT, RealInitValueT, ForceInclusive>(
      squad, specialRegisters, params, ::cuda::std::move(scan_op), static_cast<RealInitValueT>(init_value));
  });
#endif // __cccl_ptx_isa >= 860
}

template <typename AccumT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
device_scan_init_lookahead_body(tile_state_t<AccumT>* tile_states, const size_t num_temp_states)
{
  const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (tile_id >= num_temp_states)
  {
    return;
  }
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  // we strive to initialize the padding bits to avoid compute-sanitizer's initcheck to report reading uninitialized
  // data when reading the tile state. We use a single atomic load/store up until 16 bytes.
  static_assert(EMPTY == 0); // so we can zero init each tile state
  if constexpr (sizeof(tile_state_t<AccumT>) == 2)
  {
    *reinterpret_cast<uint16_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(tile_state_t<AccumT>) == 4)
  {
    *reinterpret_cast<uint32_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(tile_state_t<AccumT>) == 8)
  {
    *reinterpret_cast<uint64_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(tile_state_t<AccumT>) == 16)
  {
    *reinterpret_cast<uint4*>(tile_states + tile_id) = {};
  }
  else
  {
    tile_states[tile_id] = tile_state_t<AccumT>{};
  }
}
} // namespace detail::scan

CUB_NAMESPACE_END
