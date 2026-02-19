// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/detail/warpspeed/allocators/smem_allocator.cuh>
#include <cub/detail/warpspeed/look_ahead.cuh>
#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/resource/smem_resource.cuh>
#include <cub/detail/warpspeed/special_registers.cuh>
#include <cub/detail/warpspeed/squad/load_store.cuh>
#include <cub/detail/warpspeed/squad/squad.cuh>
#include <cub/detail/warpspeed/values.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__memory/align_down.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__ptx/instructions/clusterlaunchcontrol.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <int NumSquads,
          int NumReduceWarps,
          int NumScanStorWarps,
          int NumLoadWarps,
          int NumSchedWarps,
          int NumLookAheadWarps,
          int NumLookAheadItems,
          int NumTotalThreads,
          int ItemsPerThread,
          int TileSize>
struct warpspeedKernelPolicy
{
  static constexpr int num_squads = NumSquads;

  // The squads cannot be static constexpr variables, as those are not device accessible
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr warpspeed::SquadDesc squadReduce() noexcept
  {
    return warpspeed::SquadDesc{0, NumReduceWarps};
  }
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr warpspeed::SquadDesc squadScanStore() noexcept
  {
    return warpspeed::SquadDesc{1, NumScanStorWarps};
  }
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr warpspeed::SquadDesc squadLoad() noexcept
  {
    return warpspeed::SquadDesc{2, NumLoadWarps};
  }
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr warpspeed::SquadDesc squadSched() noexcept
  {
    return warpspeed::SquadDesc{3, NumSchedWarps};
  }
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr warpspeed::SquadDesc squadLookback() noexcept
  {
    return warpspeed::SquadDesc{4, NumLookAheadWarps};
  }

  static constexpr int num_look_ahead_items = NumLookAheadItems;
  static constexpr int num_total_threads    = NumTotalThreads;
  static constexpr int items_per_thread     = ItemsPerThread;
  static constexpr int tile_size            = TileSize;

  static constexpr int num_reduce_warps     = NumReduceWarps;
  static constexpr int num_scan_stor_warps  = NumScanStorWarps;
  static constexpr int num_load_warps       = NumLoadWarps;
  static constexpr int num_sched_warps      = NumSchedWarps;
  static constexpr int num_look_ahead_warps = NumLookAheadWarps;
};

template <typename InputT, typename OutputT, typename AccumT>
struct scanKernelParams
{
  const InputT* ptrIn;
  OutputT* ptrOut;
  warpspeed::tile_state_t<AccumT>* ptrTileStates;
  ::cuda::std::size_t numElem;
  int numStages;
};

// Struct holding all scan kernel resources
template <typename WarpspeedPolicy, typename InputT, typename OutputT, typename AccumT>
struct ScanResources
{
  // align to at least 16 bytes (InputT/OutputT may be aligned higher) so each stage starts correctly aligned
  struct alignas(::cuda::std::max({::cuda::std::size_t{16}, alignof(InputT), alignof(OutputT)})) InOutT
  {
    // the tile_size size is a multiple of the warp size, and thus for sure a multiple of 16
    static_assert(WarpspeedPolicy::tile_size % 16 == 0, "tile_size must be multiple of 16");

    // therefore, unaligned inputs need exactly 16 bytes extra for overcopying (tail padding = 16 - head padding)
    ::cuda::std::byte inout[WarpspeedPolicy::tile_size * sizeof(InputT) + 16];
  };
  static_assert(alignof(InOutT) >= alignof(InputT));
  static_assert(alignof(InOutT) >= alignof(OutputT));
  using SumThreadAndWarpT =
    AccumT[WarpspeedPolicy::squadReduce().threadCount() + WarpspeedPolicy::squadReduce().warpCount()];

  warpspeed::SmemResource<InOutT> smemInOut; // will also be used to stage the output (as OutputT) for the bulk copy
  warpspeed::SmemResource<uint4> smemNextBlockIdx;
  warpspeed::SmemResource<AccumT> smemSumExclusiveCta;
  warpspeed::SmemResource<SumThreadAndWarpT> smemSumThreadAndWarp;
};

struct ScanResourcesRaw
{
  warpspeed::SmemResourceRaw smemInOut;
  warpspeed::SmemResourceRaw smemNextBlockIdx;
  warpspeed::SmemResourceRaw smemSumExclusiveCta;
  warpspeed::SmemResourceRaw smemSumThreadAndWarp;
};

struct scan_stage_counts
{
  int num_block_idx_stages;
  int num_sum_exclusive_cta_stages;
};

_CCCL_API constexpr scan_stage_counts make_scan_stage_counts(int num_stages)
{
  int num_block_idx_stages = num_stages - 1;
  num_block_idx_stages     = num_block_idx_stages < 1 ? 1 : num_block_idx_stages;
  return {num_block_idx_stages, 2};
}

template <typename WarpspeedPolicy>
_CCCL_API constexpr scan_warpspeed_policy make_scan_warpspeed_policy()
{
  return scan_warpspeed_policy{
    WarpspeedPolicy::num_reduce_warps,
    WarpspeedPolicy::num_scan_stor_warps,
    WarpspeedPolicy::num_load_warps,
    WarpspeedPolicy::num_sched_warps,
    WarpspeedPolicy::num_look_ahead_warps,
    WarpspeedPolicy::num_look_ahead_items,
    WarpspeedPolicy::num_total_threads,
    WarpspeedPolicy::items_per_thread,
    WarpspeedPolicy::tile_size};
}

template <typename Policy,
          typename SmemInOutT,
          typename SmemNextBlockIdxT,
          typename SmemSumExclusiveCtaT,
          typename SmemSumThreadAndWarpT>
_CCCL_API constexpr void setup_scan_resources(
  const Policy& policy,
  warpspeed::SyncHandler& syncHandler,
  warpspeed::SmemAllocator& smemAllocator,
  SmemInOutT& smemInOut,
  SmemNextBlockIdxT& smemNextBlockIdx,
  SmemSumExclusiveCtaT& smemSumExclusiveCta,
  SmemSumThreadAndWarpT& smemSumThreadAndWarp)
{
  const warpspeed::SquadDesc scanSquads[Policy::num_squads] = {
    policy.squadReduce(),
    policy.squadScanStore(),
    policy.squadLoad(),
    policy.squadSched(),
    policy.squadLookback(),
  };

  smemInOut.addPhase(syncHandler, smemAllocator, policy.squadLoad());
  smemInOut.addPhase(syncHandler, smemAllocator, {policy.squadReduce(), policy.squadScanStore()});

  smemNextBlockIdx.addPhase(syncHandler, smemAllocator, policy.squadSched());
  smemNextBlockIdx.addPhase(syncHandler, smemAllocator, scanSquads);

  smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, policy.squadLookback());
  smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, policy.squadScanStore());

  smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, policy.squadReduce());
  smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, policy.squadScanStore());
}

// Function to allocate resources.

template <typename WarpspeedPolicy, typename InputT, typename OutputT, typename AccumT>
[[nodiscard]] _CCCL_API constexpr ScanResources<WarpspeedPolicy, InputT, OutputT, AccumT>
allocResources(warpspeed::SyncHandler& syncHandler, warpspeed::SmemAllocator& smemAllocator, int numStages)
{
  using ScanResourcesT    = ScanResources<WarpspeedPolicy, InputT, OutputT, AccumT>;
  using InOutT            = typename ScanResourcesT::InOutT;
  using SumThreadAndWarpT = typename ScanResourcesT::SumThreadAndWarpT;

  const auto counts = make_scan_stage_counts(numStages);

  ScanResourcesT res = {
    warpspeed::SmemResource<InOutT>(syncHandler, smemAllocator, warpspeed::Stages{numStages}),
    warpspeed::SmemResource<uint4>(syncHandler, smemAllocator, warpspeed::Stages{counts.num_block_idx_stages}),
    warpspeed::SmemResource<AccumT>(syncHandler, smemAllocator, warpspeed::Stages{counts.num_sum_exclusive_cta_stages}),
    warpspeed::SmemResource<SumThreadAndWarpT>(syncHandler, smemAllocator, warpspeed::Stages{numStages}),
  };

  setup_scan_resources(
    make_scan_warpspeed_policy<WarpspeedPolicy>(),
    syncHandler,
    smemAllocator,
    res.smemInOut,
    res.smemNextBlockIdx,
    res.smemSumExclusiveCta,
    res.smemSumThreadAndWarp);

  return res;
}

#if __cccl_ptx_isa >= 860

_CCCL_DEVICE_API inline void squadGetNextBlockIdx(const warpspeed::Squad& squad, warpspeed::SmemRef<uint4>& refDestSmem)
{
  if (squad.isLeaderThread())
  {
    ::cuda::ptx::clusterlaunchcontrol_try_cancel(&refDestSmem.data(), refDestSmem.ptrCurBarrierRelease());
  }
  refDestSmem.squadIncreaseTxCount(squad, refDestSmem.sizeBytes());
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpReduce(const Tp input, ScanOpT& scan_op)
{
  using warp_reduce_t = WarpReduce<Tp>;

  // TODO (elstehle): Do proper temporary storage allocation in case WarpReduce may rely on it
  static_assert(sizeof(typename warp_reduce_t::TempStorage) <= 4,
                "WarpReduce with non-trivial temporary storage is not supported yet in this kernel.");

  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpReducePartial(const Tp input, ScanOpT& scan_op, const int num_items)
{
  using warp_reduce_t = WarpReduce<Tp>;

  // TODO (elstehle): Do proper temporary storage allocation in case WarpReduce may rely on it
  static_assert(sizeof(typename warp_reduce_t::TempStorage) <= 4,
                "WarpReduce with non-trivial temporary storage is not supported yet in this kernel.");

  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op, num_items);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpScanExclusive(const Tp regInput, ScanOpT& scan_op)
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
  warpspeed::Squad squad,
  warpspeed::SpecialRegisters specialRegisters,
  const scanKernelParams<InputT, OutputT, AccumT>& params,
  ScanOpT scan_op,
  RealInitValueT real_init_value)
{
  ////////////////////////////////////////////////////////////////////////////////
  // Tuning dependent variables
  ////////////////////////////////////////////////////////////////////////////////
  static constexpr warpspeed::SquadDesc squadReduce    = WarpspeedPolicy::squadReduce();
  static constexpr warpspeed::SquadDesc squadScanStore = WarpspeedPolicy::squadScanStore();
  static constexpr warpspeed::SquadDesc squadLoad      = WarpspeedPolicy::squadLoad();
  static constexpr warpspeed::SquadDesc squadSched     = WarpspeedPolicy::squadSched();
  static constexpr warpspeed::SquadDesc squadLookback  = WarpspeedPolicy::squadLookback();

  constexpr int tile_size            = WarpspeedPolicy::tile_size;
  constexpr int num_look_ahead_items = WarpspeedPolicy::num_look_ahead_items;

  // We might try to instantiate the kernel with hughe types which would lead to a small tile size. Ensure its never 0
  constexpr int elemPerThread = WarpspeedPolicy::items_per_thread;
  static_assert(elemPerThread * squadReduce.threadCount() == tile_size, "Invalid tuning policy");

  ////////////////////////////////////////////////////////////////////////////////
  // Resources
  ////////////////////////////////////////////////////////////////////////////////
  warpspeed::SyncHandler syncHandler{};
  warpspeed::SmemAllocator smemAllocator{};

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
    warpspeed::SmemStage stageNextBlockIdx     = res.smemNextBlockIdx.popStage();
    warpspeed::SmemStage stageInOut            = res.smemInOut.popStage();
    warpspeed::SmemStage stageSumThreadAndWarp = res.smemSumThreadAndWarp.popStage();
    warpspeed::SmemStage stageSumExclusiveCta  = res.smemSumExclusiveCta.popStage();

    // Split the stages into phases. Each resource goes through phases where it
    // is writeable by a set of threads and readable by a set of threads. To
    // acquire and release a phase, we need to arrive and wait on certain
    // barriers. The selection of the barriers is handled under the hood.
    auto [phaseNextBlockIdxW, phaseNextBlockIdxR]         = warpspeed::bindPhases<2>(stageNextBlockIdx);
    auto [phaseInOutW, phaseInOutRW]                      = warpspeed::bindPhases<2>(stageInOut);
    auto [phaseSumThreadAndWarpW, phaseSumThreadAndWarpR] = warpspeed::bindPhases<2>(stageSumThreadAndWarp);
    auto [phaseSumExclusiveCtaW, phaseSumExclusiveCtaR]   = warpspeed::bindPhases<2>(stageSumExclusiveCta);

    // We need to handle the first and the last -partial- tile differently
    const bool is_first_tile = idxTile == 0;

    if (squad == squadSched)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load next tile index
      ////////////////////////////////////////////////////////////////////////////////
      warpspeed::SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
      squadGetNextBlockIdx(squad, refNextBlockIdxW);
    }

    const ::cuda::std::size_t idxTileBase = idxTile * ::cuda::std::size_t(tile_size);
    _CCCL_ASSERT(idxTileBase < params.numElem, "");
    const int valid_items =
      static_cast<int>(cuda::std::min(params.numElem - idxTileBase, ::cuda::std::size_t(tile_size)));
    const bool is_last_tile = valid_items < tile_size;
    warpspeed::CpAsyncOobInfo loadInfo =
      warpspeed::prepareCpAsyncOob(const_cast<InputT*>(params.ptrIn) + idxTileBase, valid_items);

    if (squad == squadLoad)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load current tile
      ////////////////////////////////////////////////////////////////////////////////
      warpspeed::SmemRef refInOutW = phaseInOutW.acquireRef();
      warpspeed::squadLoadBulk(squad, refInOutW, loadInfo);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Get next tile index from shared memory (all squads)
    ////////////////////////////////////////////////////////////////////////////////
    uint4 regNextBlockIdx{};
    {
      warpspeed::SmemRef refNextBlockIdxR = phaseNextBlockIdxR.acquireRef();
      regNextBlockIdx                     = refNextBlockIdxR.data();
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
        warpspeed::SmemRef refInOutRW = phaseInOutRW.acquireRef();
        // Load data
        AccumT regInput[elemPerThread];
        // refInOutRW.data() + loadInfo.smemStartSkipBytes points to the first element of the tile.
        // in the last tile, we load some invalid elements, but don't process them later
        warpspeed::squadLoadSmem(
          squad, regInput, reinterpret_cast<const InputT*>(&refInOutRW.data().inout[0] + loadInfo.smemStartSkipBytes));

        ////////////////////////////////////////////////////////////////////////////////
        // Reduce across thread and warp
        ////////////////////////////////////////////////////////////////////////////////
        if (is_last_tile)
        {
          // TODO(bgruber): for operators where we know the identity we can probably optimize further here
          regThreadSum = ThreadReducePartial(regInput, scan_op, valid_items_this_thread);
          regWarpSum   = warpReducePartial(regThreadSum, scan_op, valid_threads_this_warp);
        }
        else
        {
          regThreadSum = ThreadReduce(regInput, scan_op);
          regWarpSum   = warpReduce(regThreadSum, scan_op);
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store warp sump to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      warpspeed::SmemRef refSumThreadAndWarpW = phaseSumThreadAndWarpW.acquireRef();

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
        warpspeed::storeTileAggregate(params.ptrTileStates, warpspeed::scan_state::tile_aggregate, regSquadSum, idxTile);
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
      warpspeed::SmemRef refSumExclusiveCtaW = phaseSumExclusiveCtaW.acquireRef();

      if (!is_first_tile)
      {
        AccumT regSumExclusiveCta = warpspeed::warpIncrementalLookback<num_look_ahead_items>(
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
        warpspeed::SmemRef refSumThreadAndWarpR = phaseSumThreadAndWarpR.acquireRef();
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
        warpspeed::SmemRef refSumExclusiveCtaR = phaseSumExclusiveCtaR.acquireRef();

        if (!is_first_tile)
        {
          // Add the sums of preceding CTAs to the cumulative sum.
          AccumT regSumExclusiveCta = refSumExclusiveCtaR.data();
          // sumExclusive is invalid in warp_0/thread_0, so only include it in other threads/warps
          sumExclusive = squad.threadRank() == 0 ? regSumExclusiveCta : scan_op(sumExclusive, regSumExclusiveCta);
        }
      }

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
      warpspeed::SmemRef refInOutRW = phaseInOutRW.acquireRef();

      // We are always loading a full tile even for the last one. That will call scan_op on invalid data
      // loading partial tiles here regresses perf for about 10-15%
      warpspeed::squadLoadSmem(
        squad,
        regSumInclusive,
        reinterpret_cast<const InputT*>(&refInOutRW.data().inout[0] + loadInfo.smemStartSkipBytes));

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

      ::cuda::std::byte* smem_output_tile = refInOutRW.data().inout;
      if constexpr (sizeof(OutputT) <= sizeof(InputT))
      {
        warpspeed::CpAsyncOobInfo storeInfo = warpspeed::prepareCpAsyncOob(params.ptrOut + idxTileBase, valid_items);

        warpspeed::squadStoreSmem(
          squad, reinterpret_cast<OutputT*>(smem_output_tile + storeInfo.smemStartSkipBytes), regSumInclusive);
        // We do *not* release refSmemInOut here, because we will issue a TMA
        // instruction below. Instead, we issue a squad-local syncthreads +
        // fence.proxy.async to sync the shared memory writes with the TMA store.
        squad.syncThreads();

        ////////////////////////////////////////////////////////////////////////////////
        // Store result to global memory using TMA
        ////////////////////////////////////////////////////////////////////////////////
        warpspeed::squadStoreBulkSync(squad, storeInfo, smem_output_tile);
      }
      else
      {
        // otherwise, issue multiple bulk copies in chunks of the input tile size
        // TODO(bgruber): I am sure this could be implemented a lot more efficiently
        static constexpr int elem_per_chunk =
          static_cast<int>(WarpspeedPolicy::tile_size * sizeof(InputT) / sizeof(OutputT));
        for (int chunk_offset = 0; chunk_offset < valid_items; chunk_offset += elem_per_chunk)
        {
          const int chunk_size = ::cuda::std::min(valid_items - chunk_offset, elem_per_chunk);
          warpspeed::CpAsyncOobInfo storeInfo =
            warpspeed::prepareCpAsyncOob(params.ptrOut + idxTileBase + chunk_offset, chunk_size);

          // only stage elements of the current chunk to SMEM
          // storeInfo.smemStartSkipBytes < 16 and smem_output_tile contains extra 16 bytes, so we should fit
          _CCCL_ASSERT(storeInfo.smemStartSkipBytes + elem_per_chunk * sizeof(OutputT) <= res.smemInOut.mSizeBytes, "");
          warpspeed::squadStoreSmemPartial(
            squad,
            reinterpret_cast<OutputT*>(smem_output_tile + storeInfo.smemStartSkipBytes), // different in each iteration
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
          warpspeed::squadStoreBulkSync(squad, storeInfo, smem_output_tile);

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
  warpspeed::SpecialRegisters specialRegisters = warpspeed::getSpecialRegisters();

  // Dispatch for warp-specialization
  static constexpr warpspeed::SquadDesc scanSquads[WarpspeedPolicy::num_squads] = {
    WarpspeedPolicy::squadReduce(),
    WarpspeedPolicy::squadScanStore(),
    WarpspeedPolicy::squadLoad(),
    WarpspeedPolicy::squadSched(),
    WarpspeedPolicy::squadLookback(),
  };

  // we need to force inline the lambda, but clang in CUDA mode only likes the GNU syntax
  warpspeed::squadDispatch(specialRegisters, scanSquads, [&](warpspeed::Squad squad) _CCCL_FORCEINLINE_LAMBDA {
    kernelBody<WarpspeedPolicy, InputT, OutputT, AccumT, ScanOpT, RealInitValueT, ForceInclusive>(
      squad, specialRegisters, params, ::cuda::std::move(scan_op), static_cast<RealInitValueT>(init_value));
  });
#endif // __cccl_ptx_isa >= 860
}

template <typename AccumT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
device_scan_init_lookahead_body(warpspeed::tile_state_t<AccumT>* tile_states, const int num_temp_states)
{
  const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (tile_id >= num_temp_states)
  {
    return;
  }
  // we strive to initialize the padding bits to avoid compute-sanitizer's initcheck to report reading uninitialized
  // data when reading the tile state. We use a single atomic load/store up until 16 bytes.
  static_assert(warpspeed::scan_state::empty == 0); // so we can zero init each tile state
  if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 2)
  {
    *reinterpret_cast<::cuda::std::uint16_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 4)
  {
    *reinterpret_cast<::cuda::std::uint32_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 8)
  {
    *reinterpret_cast<::cuda::std::uint64_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 16)
  {
    *reinterpret_cast<uint4*>(tile_states + tile_id) = {};
  }
  else
  {
    tile_states[tile_id] = warpspeed::tile_state_t<AccumT>{};
  }
}

_CCCL_API constexpr auto smem_for_stages(
  const scan_warpspeed_policy& policy,
  int num_stages,
  int input_size,
  int input_align,
  int output_size,
  int output_align,
  int accum_size,
  int accum_align) -> int
{
  warpspeed::SyncHandler syncHandler{};
  warpspeed::SmemAllocator smemAllocator{};
  (void) output_size;
  const auto counts = make_scan_stage_counts(num_stages);

  const int align_inout     = ::cuda::std::max({16, input_align, output_align});
  const int inout_bytes     = policy.tile_size * input_size + 16;
  const auto reduce_squad   = policy.squadReduce();
  const int sum_thread_warp = (reduce_squad.threadCount() + reduce_squad.warpCount()) * accum_size;

  void* inout_base = smemAllocator.alloc(static_cast<::cuda::std::uint32_t>(inout_bytes * num_stages), align_inout);
  void* next_block_idx_base = smemAllocator.alloc(
    static_cast<::cuda::std::uint32_t>(sizeof(uint4) * counts.num_block_idx_stages), alignof(uint4));
  void* sum_exclusive_base = smemAllocator.alloc(
    static_cast<::cuda::std::uint32_t>(accum_size * counts.num_sum_exclusive_cta_stages), accum_align);
  void* sum_thread_warp_base =
    smemAllocator.alloc(static_cast<::cuda::std::uint32_t>(sum_thread_warp * num_stages), accum_align);

  ScanResourcesRaw res = {
    warpspeed::SmemResourceRaw{syncHandler, inout_base, inout_bytes, inout_bytes, num_stages},
    warpspeed::SmemResourceRaw{
      syncHandler,
      next_block_idx_base,
      static_cast<int>(sizeof(uint4)),
      static_cast<int>(sizeof(uint4)),
      counts.num_block_idx_stages},
    warpspeed::SmemResourceRaw{
      syncHandler, sum_exclusive_base, accum_size, accum_size, counts.num_sum_exclusive_cta_stages},
    warpspeed::SmemResourceRaw{syncHandler, sum_thread_warp_base, sum_thread_warp, sum_thread_warp, num_stages},
  };

  setup_scan_resources(
    policy,
    syncHandler,
    smemAllocator,
    res.smemInOut,
    res.smemNextBlockIdx,
    res.smemSumExclusiveCta,
    res.smemSumThreadAndWarp);
  syncHandler.mHasInitialized = true; // avoid assertion in destructor
  return static_cast<int>(smemAllocator.sizeBytes());
}

template <typename InputT, typename OutputT, typename AccumT>
_CCCL_API constexpr auto smem_for_stages(const scan_warpspeed_policy& policy, int num_stages) -> int
{
  return smem_for_stages(
    policy,
    num_stages,
    static_cast<int>(sizeof(InputT)),
    static_cast<int>(alignof(InputT)),
    static_cast<int>(sizeof(OutputT)),
    static_cast<int>(alignof(OutputT)),
    static_cast<int>(sizeof(AccumT)),
    static_cast<int>(alignof(AccumT)));
}

_CCCL_API constexpr bool use_warpspeed(
  const scan_warpspeed_policy& policy,
  int input_size,
  int input_align,
  int output_size,
  int output_align,
  int accum_size,
  int accum_align,
  bool input_contiguous,
  bool output_contiguous,
  bool input_trivially_copyable,
  bool output_trivially_copyable,
  bool output_default_constructible)
{
#if defined(__CUDA_ARCH__) && __cccl_ptx_isa < 860
  (void) policy;
  (void) input_size;
  (void) input_align;
  (void) output_size;
  (void) output_align;
  (void) accum_size;
  (void) accum_align;
  (void) input_contiguous;
  (void) output_contiguous;
  (void) input_trivially_copyable;
  (void) output_trivially_copyable;
  (void) output_default_constructible;
  return false;
#else
  if (!input_contiguous || !output_contiguous || !input_trivially_copyable || !output_trivially_copyable
      || !output_default_constructible)
  {
    return false;
  }

  return smem_for_stages(policy, 1, input_size, input_align, output_size, output_align, accum_size, accum_align)
      <= static_cast<int>(max_smem_per_block);
#endif
}

template <typename InputIteratorT, typename OutputIteratorT, typename AccumT>
_CCCL_API constexpr bool use_warpspeed(const scan_warpspeed_policy& policy)
{
  using InputT  = it_value_t<InputIteratorT>;
  using OutputT = it_value_t<OutputIteratorT>;
  return use_warpspeed(
    policy,
    static_cast<int>(sizeof(InputT)),
    static_cast<int>(alignof(InputT)),
    static_cast<int>(sizeof(OutputT)),
    static_cast<int>(alignof(OutputT)),
    static_cast<int>(sizeof(AccumT)),
    static_cast<int>(alignof(AccumT)),
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>,
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT>,
    ::cuda::std::is_trivially_copyable_v<InputT>,
    ::cuda::std::is_trivially_copyable_v<OutputT>,
    ::cuda::std::is_default_constructible_v<OutputT>);
}

#if 0
// we check the required shared memory inside a template, so the error message shows the amount in case of failure
template <int RequiredSharedMemory>
_CCCL_API constexpr void verify_smem()
{
  static_assert(RequiredSharedMemory <= max_smem_per_block,
                "Single stage configuration exceeds architecture independent SMEM (48KiB)");
}
#endif

template <typename WarpspeedPolicy, typename InputIteratorT, typename OutputIteratorT, typename AccumT>
_CCCL_API constexpr auto one_stage_fits_48KiB_SMEM() -> bool
{
  using InputT                           = it_value_t<InputIteratorT>;
  using OutputT                          = it_value_t<OutputIteratorT>;
  constexpr scan_warpspeed_policy policy = make_scan_warpspeed_policy<WarpspeedPolicy>();
  constexpr int smem_size_1_stage        = smem_for_stages<InputT, OutputT, AccumT>(policy, 1);
// We can turn this on to report if a single stage of the warpspeed scan would exceed 48KiB if SMEM.
#if 0
  verify_smem<smem_size_1_stage>();
#endif
  return smem_size_1_stage <= max_smem_per_block;
}

template <typename Policy, typename InputIterator, typename OutputIterator, typename AccumT, typename = void>
inline constexpr bool scan_use_warpspeed = false;

// We need `cuda::std::is_constant_evaluated` for the compile-time SMEM computation. And we need PTX ISA 8.6.
// MSVC + nvcc < 13.1 just fails to compile `cub.test.device.scan.lid_1.types_0` with `Internal error` and nothing else.
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) && __cccl_ptx_isa >= 860 \
  && !(_CCCL_COMPILER(MSVC) && _CCCL_CUDA_COMPILER(NVCC, <, 13, 1))
template <typename Policy, typename InputIteratorT, typename OutputIteratorT, typename AccumT>
inline constexpr bool scan_use_warpspeed<Policy,
                                         InputIteratorT,
                                         OutputIteratorT,
                                         AccumT,
                                         ::cuda::std::void_t<typename Policy::WarpspeedPolicy>> =
  // for bulk copy: input and output iterators must be contiguous and their value types must be trivially copyable
  THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>
  && ::cuda::std::is_trivially_copyable_v<it_value_t<InputIteratorT>>
  && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT>
  && ::cuda::std::is_trivially_copyable_v<it_value_t<OutputIteratorT>>
  // for bulk copy store: we need to prepare a buffer of output types in SMEM
  && ::cuda::std::is_default_constructible_v<it_value_t<OutputIteratorT>>
  // need to fit one stage into 48KiB so binaries stay forward compatible with future GPUs
  && one_stage_fits_48KiB_SMEM<typename Policy::WarpspeedPolicy, InputIteratorT, OutputIteratorT, AccumT>();
#endif // defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) && __cccl_ptx_isa >= 860
} // namespace detail::scan

CUB_NAMESPACE_END
