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
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__ptx/instructions/clusterlaunchcontrol.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
namespace __cub_detail  = CUB_NS_QUALIFIER::detail;
namespace __scan_detail = CUB_NS_QUALIFIER::detail::scan;

_CCCL_HOST_DEVICE_API constexpr int num_total_threads(const ScanLookaheadPolicy& policy)
{
  const auto num_total_warps = 2 * policy.reduce_and_scan_warps + 1 /*num_load_warps*/
                             + 1 /*num_sched_warps*/ + 1 /*num_lookahead_warps*/;
  return num_total_warps * warp_threads;
}

template <typename InputT, typename OutputT, typename AccumT>
struct scanKernelParams
{
  const InputT* ptrIn;
  OutputT* ptrOut;
  warpspeed::tile_state_t<AccumT>* ptrTileStates;
  ::cuda::std::size_t numElem;
  int numStages;
};

// holds all scan kernel resources
template <typename PolicySelector, typename InputT, typename OutputT, typename AccumT>
struct ScanResources
{
  static constexpr ScanLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;

  // align to at least 16 bytes (InputT/OutputT may be aligned higher) so each stage starts correctly aligned
  struct alignas(::cuda::std::max({::cuda::std::size_t{16}, alignof(InputT), alignof(OutputT)})) in_out_t
  {
    // the tile_size size is a multiple of the warp size, and thus for sure a multiple of 16
    static_assert(policy.tile_size() % 16 == 0, "tile_size must be multiple of 16");

    // therefore, unaligned inputs need exactly 16 bytes extra for overcopying (tail padding = 16 - head padding)
    ::cuda::std::byte inout[policy.tile_size() * sizeof(InputT) + 16];
  };
  static_assert(alignof(in_out_t) >= alignof(InputT));
  static_assert(alignof(in_out_t) >= alignof(OutputT));
  using thread_and_warp_aggr_t = AccumT[squad_reduce(policy).threadCount() + squad_reduce(policy).warpCount()];

  warpspeed::SmemResource<in_out_t> smemInOut; // will also be used to stage the output (as OutputT) for the bulk copy
  warpspeed::SmemResource<uint4> smemNextBlockIdx;
  warpspeed::SmemResource<AccumT> smemAggrExclusiveCta;
  warpspeed::SmemResource<thread_and_warp_aggr_t> smemThreadAndWarpAggr;
};

template <typename PolicySelector, typename InputT, typename OutputT, typename AccumT>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
allocResources(warpspeed::SyncHandler& syncHandler, warpspeed::SmemAllocator& smemAllocator, int numStages)
  -> ScanResources<PolicySelector, InputT, OutputT, AccumT>
{
  using ScanResourcesT         = ScanResources<PolicySelector, InputT, OutputT, AccumT>;
  using in_out_t               = typename ScanResourcesT::in_out_t;
  using thread_and_warp_aggr_t = typename ScanResourcesT::thread_and_warp_aggr_t;

  constexpr auto policy = current_policy<PolicySelector>().lookahead;

  const int num_block_idx_stages =
    policy.block_idx_stages > 0 ? policy.block_idx_stages : ::cuda::std::max(1, numStages + policy.block_idx_stages);
  const int num_aggr_exclusive_cta_stages =
    policy.lookahead_stages > 0 ? policy.lookahead_stages : ::cuda::std::max(1, numStages + policy.lookahead_stages);

  ScanResourcesT res = {
    warpspeed::SmemResource<in_out_t>(syncHandler, smemAllocator, warpspeed::Stages{numStages}),
    warpspeed::SmemResource<uint4>(syncHandler, smemAllocator, warpspeed::Stages{num_block_idx_stages}),
    warpspeed::SmemResource<AccumT>(syncHandler, smemAllocator, warpspeed::Stages{num_aggr_exclusive_cta_stages}),
    warpspeed::SmemResource<thread_and_warp_aggr_t>(syncHandler, smemAllocator, warpspeed::Stages{numStages}),
  };

  setup_scan_resources(
    policy,
    syncHandler,
    smemAllocator,
    res.smemInOut,
    res.smemNextBlockIdx,
    res.smemAggrExclusiveCta,
    res.smemThreadAndWarpAggr);

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
  static_assert(::cuda::std::is_same_v<typename warp_reduce_t::TempStorage, Uninitialized<NullType>>,
                "WarpReduce for a full warp must not require temporary storage");
  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpReducePartial(const Tp input, ScanOpT& scan_op, const int num_items)
{
  using warp_reduce_t = WarpReduce<Tp>;
  static_assert(::cuda::std::is_same_v<typename warp_reduce_t::TempStorage, Uninitialized<NullType>>,
                "WarpReduce for a full warp must not require temporary storage");
  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op, num_items);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpScanExclusive(const Tp regInput, ScanOpT& scan_op)
{
  using warp_scan_t = WarpScan<Tp>;
  static_assert(::cuda::std::is_same_v<typename warp_scan_t::TempStorage, Uninitialized<NullType>>,
                "WarpScan for a full warp must not require temporary storage");
  typename warp_scan_t::TempStorage temp_storage;
  Tp result;
  warp_scan_t{temp_storage}.ExclusiveScan(regInput, result, scan_op);
  return result;
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE Tp
warpScanExclusivePartial(Tp regInput, ScanOpT& scan_op, const int num_items, bool this_lane_is_valid)
{
  // if we have an identity, just fill the out-of-bounds items with it and use the full warp scan, since it's faster
  if constexpr (cuda::has_identity_element_v<ScanOpT, Tp>)
  {
    if (!this_lane_is_valid)
    {
      regInput = cuda::identity_element<ScanOpT, Tp>();
    }
    return warpScanExclusive(regInput, scan_op);
  }
  else
  {
    using warp_scan_t = WarpScan<Tp>;
    static_assert(::cuda::std::is_same_v<typename warp_scan_t::TempStorage, Uninitialized<NullType>>,
                  "WarpScan for a full warp must not require temporary storage");
    Tp result;
    typename warp_scan_t::TempStorage temp_storage;
    warp_scan_t{temp_storage}.ExclusiveScanPartial(regInput, result, scan_op, num_items);
    return result;
  }
}

template <typename ScanOpT, typename Tp, size_t ElemPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void fillWithIdentity(Tp (&regAggrInclusive)[ElemPerThread], int valid_items)
{
  // if we are in the last tile and have an identity, fill the invalid array items with it
  if constexpr (::cuda::has_identity_element_v<ScanOpT, Tp>)
  {
    for (int i = 0; i < ElemPerThread; ++i)
    {
      if (i >= valid_items)
      {
        regAggrInclusive[i] = ::cuda::identity_element<ScanOpT, Tp>();
      }
    }
  }
}

template <bool IsInclusive, bool IsLastTile, typename Tp, size_t ElemPerThread, typename ScanOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
threadScanPartial(Tp (&regAggrInclusive)[ElemPerThread], ScanOpT& scan_op, Tp prefix, bool use_prefix, int valid_items)
{
  // skip the partial scan if we have an identity
  if constexpr (IsLastTile && !::cuda::has_identity_element_v<ScanOpT, Tp>)
  {
    if constexpr (IsInclusive)
    {
      __cub_detail::ThreadScanInclusivePartial(
        regAggrInclusive, regAggrInclusive, scan_op, valid_items, prefix, use_prefix);
    }
    else
    {
      __cub_detail::ThreadScanExclusivePartial(
        regAggrInclusive, regAggrInclusive, scan_op, valid_items, prefix, use_prefix);
    }
  }
  else
  {
    if constexpr (IsInclusive)
    {
      __cub_detail::ThreadScanInclusive(regAggrInclusive, regAggrInclusive, scan_op, prefix, use_prefix);
    }
    else
    {
      __cub_detail::ThreadScanExclusive(regAggrInclusive, regAggrInclusive, scan_op, prefix, use_prefix);
    }
  }
}

// Similar to CUB agents, this closure just aggregates common constants so the device functions implementing the
// lookahead scan kernel can have lighter signatures. Each squad uses an instance of this to provide its context. In
// principle, it does not hold any mutable state. But it refers to the shared scan resources (SMEM + barriers etc.).
template <typename PolicySelector,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename RealInitValueT,
          bool ForceInclusive,
          bool StableReductionOrder = false>
struct lookahead_scan_closure
{
  static constexpr ScanLookaheadPolicy policy          = current_policy<PolicySelector>().lookahead;
  static constexpr warpspeed::SquadDesc squadReduce    = squad_reduce(policy);
  static constexpr warpspeed::SquadDesc squadScanStore = squad_scan_store(policy);
  static constexpr warpspeed::SquadDesc squadLoad      = squad_load(policy);
  static constexpr warpspeed::SquadDesc squadSched     = squad_sched(policy);
  static constexpr warpspeed::SquadDesc squadLookahead = squad_lookahead(policy);

  static constexpr ::cuda::std::array<warpspeed::SquadDesc, 5> scanSquads = {
    squad_reduce(policy),
    squad_scan_store(policy),
    squad_load(policy),
    squad_sched(policy),
    squad_lookahead(policy),
  };

  static constexpr int tile_size                  = policy.tile_size();
  static constexpr int lookahead_items_per_thread = policy.lookahead_items_per_thread;

  // We might try to instantiate the kernel with huge types which would lead to a small tile size. Ensure its never 0
  static constexpr int elemPerThread = policy.items_per_thread;
  static_assert(elemPerThread * squadReduce.threadCount() == tile_size, "Invalid tuning policy");

  // Inclusive scan if no init_value type is provided
  static constexpr bool hasInit     = !::cuda::std::is_same_v<RealInitValueT, NullType>;
  static constexpr bool isInclusive = ForceInclusive || !hasInit;

  using scan_resources_t       = ScanResources<PolicySelector, InputT, OutputT, AccumT>;
  using in_out_t               = typename scan_resources_t::in_out_t;
  using thread_and_warp_aggr_t = typename scan_resources_t::thread_and_warp_aggr_t;

  const warpspeed::SpecialRegisters specialRegisters;
  const scanKernelParams<InputT, OutputT, AccumT> params;
  mutable ScanOpT scan_op; // mutable, so we can support non-const operator()
  const RealInitValueT real_init_value;
  scan_resources_t& res; // this is the only shared mutable state

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  load_next_tile_index(const warpspeed::Squad& squad, warpspeed::SmemPhase<uint4>& phaseNextBlockIdxW) const
  {
    warpspeed::SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
    squadGetNextBlockIdx(squad, refNextBlockIdxW);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void load_current_tile(
    const warpspeed::Squad& squad,
    warpspeed::SmemPhase<in_out_t>& phaseInOutW,
    const warpspeed::CpAsyncOobInfo<InputT>& loadInfo) const
  {
    warpspeed::SmemRef refInOutW = phaseInOutW.acquireRef();
    warpspeed::squadLoadBulk(squad, refInOutW, loadInfo);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void lookahead(
    const warpspeed::Squad& squad,
    warpspeed::SmemPhase<AccumT>& phaseAggrExclusiveCtaW,
    bool is_first_tile,
    int& idxTilePrev,
    AccumT& AggrExclusiveCtaPrev,
    int idxTile) /*const*/ // FIXME(bgruber): this const causes a large SASS diff
  {
    warpspeed::SmemRef refAggrExclusiveCtaW = phaseAggrExclusiveCtaW.acquireRef();

    if (!is_first_tile)
    {
      if constexpr (StableReductionOrder)
      {
        // The stable-order version updates idxTilePrev/AggrExclusiveCtaPrev itself
        AccumT regAggrExclusiveCta = warpspeed::warpIncrementalLookaheadStable<lookahead_items_per_thread>(
          specialRegisters, params.ptrTileStates, idxTilePrev, AggrExclusiveCtaPrev, idxTile, scan_op);
        if (squad.isLeaderThread())
        {
          refAggrExclusiveCtaW.data() = regAggrExclusiveCta;
        }
      }
      else
      {
        AccumT regAggrExclusiveCta = warpspeed::warpIncrementalLookahead<lookahead_items_per_thread>(
          specialRegisters, params.ptrTileStates, idxTilePrev, AggrExclusiveCtaPrev, idxTile, scan_op);
        if (squad.isLeaderThread())
        {
          refAggrExclusiveCtaW.data() = regAggrExclusiveCta;
        }
        AggrExclusiveCtaPrev = regAggrExclusiveCta;
        idxTilePrev          = idxTile;
      }
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void reduce_tile(
    const warpspeed::Squad& squad,
    warpspeed::SmemPhase<in_out_t>& phaseInOutRW,
    warpspeed::SmemPhase<thread_and_warp_aggr_t>& phaseThreadAndWarpAggrW,
    int valid_items,
    bool is_first_tile,
    bool is_last_tile, // TODO(bgruber): should we dispatch on is_last_tile outside this function and compile it twice?
    const warpspeed::CpAsyncOobInfo<InputT>& loadInfo,
    int idxTile) const
  {
    const int valid_items_this_thread =
      cuda::std::clamp(valid_items - squad.threadRank() * elemPerThread, 0, +elemPerThread);
    const int valid_threads_this_warp =
      cuda::std::clamp(::cuda::ceil_div(valid_items, elemPerThread) - squad.warpRank() * 32, 0, 32);
    const int valid_warps = ::cuda::ceil_div(valid_items, elemPerThread * 32);
    _CCCL_ASSERT(0 < valid_warps && valid_warps <= squad.warpCount(), "");

    // Load tile from shared memory and reduce across thread and warp
    AccumT regThreadAggr;
    AccumT regWarpAggr;
    {
      // Acquire phaseInOutRW only in this short scope
      warpspeed::SmemRef refInOutRW = phaseInOutRW.acquireRef();
      AccumT regInput[elemPerThread];
      const auto* smem_data_start =
        reinterpret_cast<const InputT*>(&refInOutRW.data().inout[0] + loadInfo.smemStartSkipBytes);
      // in the last tile, we load some invalid elements, but don't process them later
      warpspeed::squadLoadSmem(squad, regInput, smem_data_start);

      // Reduce across thread and warp
      if (is_last_tile)
      {
        // TODO(bgruber): for operators where we know the identity we can probably optimize this better
        regThreadAggr = __cub_detail::ThreadReducePartial(regInput, scan_op, valid_items_this_thread);
        regWarpAggr   = __scan_detail::warpReducePartial(regThreadAggr, scan_op, valid_threads_this_warp);
      }
      else
      {
        regThreadAggr = CUB_NS_QUALIFIER::ThreadReduce(regInput, scan_op);
        regWarpAggr   = __scan_detail::warpReduce(regThreadAggr, scan_op);
      }
    }

    // Store warp aggregate to shared memory
    warpspeed::SmemRef refThreadAndWarpAggrW = phaseThreadAndWarpAggrW.acquireRef();
    if (squad.isLeaderThreadOfWarp())
    {
      refThreadAndWarpAggrW.data()[squadReduce.threadCount() + squad.warpRank()] = regWarpAggr;
    }
    squad.syncThreads();

    // Reduce across squad
    // We need to accumulate the first element by hand because of the potential initial element and partial tiles
    AccumT regSquadAggr;
    if constexpr (hasInit)
    {
      if (is_first_tile)
      {
        regSquadAggr = scan_op(real_init_value, refThreadAndWarpAggrW.data()[squadReduce.threadCount()]);
      }
      else
      {
        regSquadAggr = refThreadAndWarpAggrW.data()[squadReduce.threadCount()];
      }
    }
    else
    {
      regSquadAggr = refThreadAndWarpAggrW.data()[squadReduce.threadCount()];
    }

    if (is_last_tile)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < squadReduce.warpCount(); ++i)
      {
        if (i < valid_warps)
        {
          regSquadAggr = scan_op(regSquadAggr, refThreadAndWarpAggrW.data()[squadReduce.threadCount() + i]);
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < squadReduce.warpCount(); ++i)
      {
        regSquadAggr = scan_op(regSquadAggr, refThreadAndWarpAggrW.data()[squadReduce.threadCount() + i]);
      }
    }

    // Store tile aggregate for lookahead
    if (squad.isLeaderThread())
    {
      warpspeed::storeTileAggregate(params.ptrTileStates, warpspeed::scan_state::tile_aggregate, regSquadAggr, idxTile);
    }

    // Store thread aggregate
    refThreadAndWarpAggrW.data()[squad.threadRank()] = regThreadAggr;
  }

  template <bool IsLastTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void scan_and_store_tile(
    const warpspeed::Squad& squad,
    warpspeed::SmemPhase<thread_and_warp_aggr_t>& phaseThreadAndWarpAggrR,
    warpspeed::SmemPhase<AccumT>& phaseAggrExclusiveCtaR,
    warpspeed::SmemPhase<in_out_t>& phaseInOutRW,
    bool is_first_tile,
    int valid_items,
    const warpspeed::CpAsyncOobInfo<InputT>& loadInfo,
    ::cuda::std::size_t idxTileBase) /*const*/ // FIXME(bgruber): this const causes a large SASS diff
  {
    // need to init these to silence nvcc warning about reading uninitialized data
    [[maybe_unused]] int valid_items_this_thread = 0;
    [[maybe_unused]] int valid_threads_this_warp = 0;
    [[maybe_unused]] int valid_warps             = 0;
    if constexpr (IsLastTile)
    {
      valid_items_this_thread = ::cuda::std::clamp(valid_items - squad.threadRank() * elemPerThread, 0, +elemPerThread);
      valid_threads_this_warp =
        ::cuda::std::clamp(::cuda::ceil_div(valid_items, elemPerThread) - squad.warpRank() * 32, 0, 32);
      valid_warps = ::cuda::ceil_div(valid_items, elemPerThread * 32);
      _CCCL_ASSERT(0 < valid_warps && valid_warps <= squad.warpCount(), "");
    }

    // Fill the registers with the scan identity, if there is one, before acquiring/waiting on any resources
    AccumT regAggrInclusive[elemPerThread];
    if constexpr (IsLastTile)
    {
      fillWithIdentity<ScanOpT>(regAggrInclusive, valid_items_this_thread);
    }

    // Aggregate of all threads up to but not including this one
    AccumT aggrExclusive;

    // Include warp and thread aggregates of current tile
    {
      // acquire the thread and warp aggregates only for as long as we need them
      warpspeed::SmemRef refThreadAndWarpAggrR = phaseThreadAndWarpAggrR.acquireRef();
      // Add the aggregates of the preceding warps in this CTA to the cumulative aggregate. These have been calculated
      // in reduce squad. We need the reduce and scan squads to be the same size to do this.
      static_assert(squadReduce.warpCount() == squadScanStore.warpCount());

      // Include warp aggregates
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < squadScanStore.warpCount(); ++i)
      {
        // We want a predicated unrolled loop here.
        bool include_warp = i < squad.warpRank();
        if constexpr (IsLastTile)
        {
          include_warp &= i < valid_warps;
        }
        if (include_warp)
        {
          if (i == 0)
          {
            // The first iteration initializes aggrExclusive
            aggrExclusive = refThreadAndWarpAggrR.data()[squadReduce.threadCount()];
          }
          else
          {
            // If loaded value belongs to previous warp, include it in aggrExclusive.
            aggrExclusive = scan_op(aggrExclusive, refThreadAndWarpAggrR.data()[squadReduce.threadCount() + i]);
          }
        }
      }
      // aggrExclusive contains the aggregates of previous warps.
      // It has a valid value in
      // - tile*::warp{1,2, ..}      (aggregate of previous warps)
      //
      // It is not yet initialized in
      // - tile*::warp0

      // Add the aggregates of preceding threads in this warp to the cumulative aggregate.
      // We perform an exclusive scan of:
      //
      //   {aggrT0, aggrT1, ..., aggrT30, aggrT31 }
      //
      // As a result:
      // - lane0 has undefined value
      // - lane1 has aggrT0
      // - ...
      // - lane31 has aggrT0 + ... + aggrT30
      //
      // For lane1, ..., 31, we add the result to aggrExclusive.
      //
      // If the warp contains partial data, we pass invalid elements to
      // scan_op, and aggrExclusiveIntraWarp is invalid when the inputs were
      // invalid.
      AccumT regAggrThread = refThreadAndWarpAggrR.data()[squad.threadRank()];
      AccumT aggrExclusiveIntraWarp;
      if constexpr (IsLastTile) // this branch would cost up to 4% BW for I8 and I16 if it were at
                                // runtime
      {
        aggrExclusiveIntraWarp = __scan_detail::warpScanExclusivePartial(
          regAggrThread,
          scan_op,
          valid_threads_this_warp,
          specialRegisters.laneIdx < static_cast<uint32_t>(valid_threads_this_warp));
      }
      else
      {
        aggrExclusiveIntraWarp = __scan_detail::warpScanExclusive(regAggrThread, scan_op);
      }

      if (squad.warpRank() == 0)
      {
        // Warp0 does not yet have a valid value for aggrExclusive. We set it
        // here. This ensures that lane1,..,31 of tile0::warp0 have a valid
        // value for aggrExclusive.
        aggrExclusive = aggrExclusiveIntraWarp;
      }
      else
      {
        // lane0 has an undefined value for aggrExclusiveIntraWarp, so skip it
        bool includeIntraWarpAggr = specialRegisters.laneIdx != 0;
        if constexpr (IsLastTile)
        {
          includeIntraWarpAggr &= specialRegisters.laneIdx < static_cast<uint32_t>(valid_threads_this_warp);
        }

        if (includeIntraWarpAggr)
        {
          aggrExclusive = scan_op(aggrExclusive, aggrExclusiveIntraWarp);
        }
      }
    }
    // aggrExclusive contains the aggregates of previous warps and threads.
    //
    // - tile*::warp0::lane{1, .., 31}  (aggr of previous threads)
    // - tile*::warp{1,2, ..}           (aggr of previous warps + aggr of previous threads)
    //
    // It has an undefined value in
    // - tile*::warp0::lane0

    // Include aggregate of previous tiles
    {
      // important: we have to acquire the resource for the first tile as well to prevent a hang
      warpspeed::SmemRef refAggrExclusiveCtaR = phaseAggrExclusiveCtaR.acquireRef();

      if (!is_first_tile)
      {
        // Add the aggregates of preceding CTAs to the cumulative aggregate.
        AccumT regAggrExclusiveCta = refAggrExclusiveCtaR.data();
        // aggrExclusive is invalid in warp_0/thread_0, so only include it in other threads/warps. Skip it, when this
        // thread has no valid items to process in the last tile.
        if (squad.threadRank() == 0 || (IsLastTile && valid_items_this_thread == 0))
        {
          aggrExclusive = regAggrExclusiveCta;
        }
        else
        {
          aggrExclusive = scan_op(regAggrExclusiveCta, aggrExclusive);
        }
      }
    }

    if constexpr (hasInit)
    {
      if (is_first_tile)
      {
        // The first thread cannot use scan_op because aggrExclusive holds garbage data. Also skip sumExclusive when
        // this thread has no valid items to process in the last tile.
        if (squad.threadRank() == 0 || (IsLastTile && valid_items_this_thread == 0))
        {
          aggrExclusive = static_cast<AccumT>(real_init_value);
        }
        else
        {
          aggrExclusive = scan_op(static_cast<AccumT>(real_init_value), aggrExclusive);
        }
      }
    }
    // aggrExclusive contains the following values:
    //
    // - tile0::warp0::lane0            (init_value)
    // - tile0::warp0::lane{1, .., 31}  (init_value + aggr of previous threads)
    // - tile0::warp{1,2, ..}           (init_value + aggr of previous warps + aggr of previous threads)
    // - tile*::warp0::lane0            (aggr of previous CTAs)
    // - tile*::warp0::lane{1, .., 31}  (aggr of previous CTAs + aggr of previous threads)
    // - tile*::warp{1,2, ..}           (aggr of previous CTAs + aggr of previous warps + aggr of previous
    // threads)
    //
    // If no init value is provided, then aggrExclusive has an undefined value in
    // - tile0::warp0::lane0

    // Scan across elements allocated to this thread

    // Acquire refInOut for remainder of scope.
    warpspeed::SmemRef refInOutRW = phaseInOutRW.acquireRef();

    // We are always loading a full tile even for the last tile, so we are loading invalid data
    warpspeed::squadLoadSmem(
      squad,
      regAggrInclusive,
      reinterpret_cast<const InputT*>(&refInOutRW.data().inout[0] + loadInfo.smemStartSkipBytes));

    // Perform inclusive scan of register array in current thread.
    // warp_0/thread_0 in the first tile when there is no initial value, we MUST NOT use aggrExclusive
    const bool use_prefix = hasInit ? true : !(is_first_tile && squad.threadRank() == 0);
    threadScanPartial<isInclusive, IsLastTile>(
      regAggrInclusive, scan_op, aggrExclusive, use_prefix, valid_items_this_thread);

    // Sync before storing to avoid data races on SMEM
    squad.syncThreads();

    // Store result to shared memory
    ::cuda::std::byte* smem_output_tile = refInOutRW.data().inout;
    if constexpr (sizeof(OutputT) <= sizeof(InputT))
    {
      warpspeed::CpAsyncOobInfo storeInfo = warpspeed::prepareCpAsyncOob(params.ptrOut + idxTileBase, valid_items);

      warpspeed::squadStoreSmem(
        squad, reinterpret_cast<OutputT*>(smem_output_tile + storeInfo.smemStartSkipBytes), regAggrInclusive);
      // We do *not* release refSmemInOut here, because we will issue a TMA
      // instruction below. Instead, we issue a squad-local syncthreads +
      // fence.proxy.async to sync the shared memory writes with the TMA store.
      squad.syncThreads();

      // Store result to global memory using TMA
      warpspeed::squadStoreBulkSync(squad, storeInfo, smem_output_tile);
    }
    else
    {
      // otherwise, issue multiple bulk copies in chunks of the input tile size
      // TODO(bgruber): I am sure this could be implemented a lot more efficiently
      static constexpr int elem_per_chunk = static_cast<int>(policy.tile_size() * sizeof(InputT) / sizeof(OutputT));
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
          reinterpret_cast<OutputT*>(smem_output_tile + storeInfo.smemStartSkipBytes), // different in each
                                                                                       // iteration
          regAggrInclusive,
          chunk_offset,
          chunk_offset + chunk_size);

        // We do *not* release refSmemInOut here, because we will issue a TMA
        // instruction below. Instead, we issue a squad-local syncthreads +
        // fence.proxy.async to sync the shared memory writes with the TMA store.
        squad.syncThreads();

        // Store result to global memory using TMA
        warpspeed::squadStoreBulkSync(squad, storeInfo, smem_output_tile);

        squad.syncThreads();
      }
    }

    // Release refInOut. No need to do any cross-proxy fencing here, because
    // the TMA store in this warp and the TMA load in the load warp are both
    // async proxy.
  }

  // This function is a straight-line implementation of the warp-specialized kernel.
  //
  // It is called from the __global__ kernel body with a squad argument that is the active squad on the current thread.
  //
  // Using this structure, all code that is not executed by the current squad is DCE (dead-code-eliminated) by the
  // compiler and all warp-specialization dispatch is performed once at the start of the kernel and not in any of the
  // hot loops (even if that may seem the case from a first glance at the code).
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void dispatch_squad(warpspeed::Squad squad) // const // TODO(bgruber): enable const
  {
    // Start with the tile indicated by blockIdx.x
    int idxTile = specialRegisters.blockIdxX;
    // Lookahead-specific variables:
    int idxTilePrev = 0;
    AccumT AggrExclusiveCtaPrev; // only valid in squadLookahead lane_0

    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // Loop over tiles
#  pragma unroll 1
    while (true)
    {
      // Get stages. When these objects go out of scope, the stage of the resource is automatically incremented.
      warpspeed::SmemStage stageNextBlockIdx      = res.smemNextBlockIdx.nextStage();
      warpspeed::SmemStage stageInOut             = res.smemInOut.nextStage();
      warpspeed::SmemStage stageThreadAndWarpAggr = res.smemThreadAndWarpAggr.nextStage();
      warpspeed::SmemStage stageAggrExclusiveCta  = res.smemAggrExclusiveCta.nextStage();

      // Split the stages into phases. Each resource goes through phases where it is writeable by a set of threads and
      // readable by a set of threads. To acquire and release a phase, we need to arrive and wait on certain barriers.
      // The selection of the barriers is handled under the hood.
      auto [phaseNextBlockIdxW, phaseNextBlockIdxR]           = warpspeed::bindPhases<2>(stageNextBlockIdx);
      auto [phaseInOutW, phaseInOutRW]                        = warpspeed::bindPhases<2>(stageInOut);
      auto [phaseThreadAndWarpAggrW, phaseThreadAndWarpAggrR] = warpspeed::bindPhases<2>(stageThreadAndWarpAggr);
      auto [phaseAggrExclusiveCtaW, phaseAggrExclusiveCtaR]   = warpspeed::bindPhases<2>(stageAggrExclusiveCta);

      // We need to handle the first and the last -partial- tile differently
      const bool is_first_tile = idxTile == 0;

      if (squad == squadSched)
      {
        load_next_tile_index(squad, phaseNextBlockIdxW);
      }

      const ::cuda::std::size_t idxTileBase = idxTile * ::cuda::std::size_t(tile_size);
      _CCCL_ASSERT(idxTileBase < params.numElem, "");
      const int valid_items =
        static_cast<int>(cuda::std::min(params.numElem - idxTileBase, ::cuda::std::size_t(tile_size)));
      const bool is_last_tile = valid_items < tile_size;
      const warpspeed::CpAsyncOobInfo loadInfo =
        warpspeed::prepareCpAsyncOob(const_cast<InputT*>(params.ptrIn) + idxTileBase, valid_items);

      if (squad == squadLoad)
      {
        load_current_tile(squad, phaseInOutW, loadInfo);
      }

      // Get next tile index from shared memory (all squads)
      uint4 regNextBlockIdx{};
      {
        warpspeed::SmemRef refNextBlockIdxR = phaseNextBlockIdxR.acquireRef();
        regNextBlockIdx                     = refNextBlockIdxR.data();
        refNextBlockIdxR.setFenceLdsToAsyncProxy();
      }
      bool nextIdxTileValid = ::cuda::ptx::clusterlaunchcontrol_query_cancel_is_canceled(regNextBlockIdx);

      if (squad == squadReduce)
      {
        reduce_tile(
          squad, phaseInOutRW, phaseThreadAndWarpAggrW, valid_items, is_first_tile, is_last_tile, loadInfo, idxTile);
      }

      if (squad == squadLookahead)
      {
        lookahead(squad, phaseAggrExclusiveCtaW, is_first_tile, idxTilePrev, AggrExclusiveCtaPrev, idxTile);
      }

      if (squad == squadScanStore)
      {
        static_assert(tile_size % squadScanStore.threadCount() == 0);
        if (is_last_tile)
        {
          scan_and_store_tile<true>(
            squad,
            phaseThreadAndWarpAggrR,
            phaseAggrExclusiveCtaR,
            phaseInOutRW,
            is_first_tile,
            valid_items,
            loadInfo,
            idxTileBase);
        }
        else
        {
          scan_and_store_tile<false>(
            squad,
            phaseThreadAndWarpAggrR,
            phaseAggrExclusiveCtaR,
            phaseInOutRW,
            is_first_tile,
            valid_items,
            loadInfo,
            idxTileBase);
        }
      }

      // All squads: Check loop condition and update next tile index
      if (!nextIdxTileValid)
      {
        break;
      }
      idxTile = ::cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(regNextBlockIdx);
    }

    // epilogue: after the load squad finished, we can start ramping up the next kernel
    if (squad == squadLoad)
    {
      _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
    }
  }
};

#endif // __cccl_ptx_isa >= 860

template <typename PolicySelector,
          bool ForceInclusive,
          typename RealInitValueT,
          bool StableReductionOrder,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_scan_lookahead_body(
  const scanKernelParams<InputT, OutputT, AccumT>& params, const ScanOpT& scan_op, const InitValueT& init_value)
{
#if __cccl_ptx_isa >= 860
  // Cache special registers at the start of kernel, since getting them takes a few cycles
  warpspeed::SpecialRegisters specialRegisters = warpspeed::getSpecialRegisters();

  static constexpr ScanLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;

  // Set up the shared memory resources
  auto res = [&] {
    warpspeed::SyncHandler syncHandler{};
    warpspeed::SmemAllocator smemAllocator{};
    auto r = allocResources<PolicySelector, InputT, OutputT, AccumT>(syncHandler, smemAllocator, params.numStages);
    syncHandler.clusterInitSync<num_total_threads(policy)>(specialRegisters);
    return r;
  }();

  // Dispatch each warp to its respective squad
  using closure_t = lookahead_scan_closure<
    PolicySelector,
    InputT,
    OutputT,
    AccumT,
    ScanOpT,
    RealInitValueT,
    ForceInclusive,
    StableReductionOrder>;
  warpspeed::squadDispatch(
    specialRegisters, closure_t::scanSquads, [&](warpspeed::Squad squad) _CCCL_FORCEINLINE_LAMBDA {
      // we load the initial value after the squad dispatch, so only the squads needing it emit an LDG
      closure_t{specialRegisters, params, scan_op, static_cast<RealInitValueT>(init_value), res}.dispatch_squad(squad);
    });
#endif // __cccl_ptx_isa >= 860
}

template <typename AccumT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
device_scan_init_lookahead_body(warpspeed::tile_state_t<AccumT>* tile_states, const int num_temp_states)
{
  const int tile_id = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
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

template <typename InputT, typename OutputT, typename AccumT>
_CCCL_HOST_DEVICE_API constexpr auto smem_for_stages(const ScanLookaheadPolicy& policy, int num_stages) -> int
{
  return smem_for_stages(
    policy,
    num_stages,
    static_cast<int>(sizeof(InputT)),
    static_cast<int>(alignof(InputT)),
    static_cast<int>(alignof(OutputT)),
    static_cast<int>(sizeof(AccumT)),
    static_cast<int>(alignof(AccumT)));
}
} // namespace detail::scan

CUB_NAMESPACE_END
