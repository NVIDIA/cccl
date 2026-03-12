// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_scan.cuh>

#include <cuda/std/__functional/invoke.h>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  if !USES_WARPSPEED()
#    include <look_back_helper.cuh>
#  endif // !USES_WARPSPEED()

template <typename AccumT>
struct policy_hub_t
{
  struct MaxPolicy : cub::ChainedPolicy<300, MaxPolicy, MaxPolicy>
  {
#  if USES_WARPSPEED()
    struct WarpspeedPolicy
    {
      static constexpr int num_reduce_and_scan_warps = TUNE_NUM_REDUCE_SCAN_WARPS;
      static constexpr int num_look_ahead_items      = TUNE_NUM_LOOKBACK_ITEMS;
      static constexpr int items_per_thread          = TUNE_ITEMS_PLUS_ONE - 1;

      // the rest are fixed or derived definitions

      static constexpr int num_squads           = 5;
      static constexpr int num_threads_per_warp = 32;
      static constexpr int num_load_warps       = 1;
      static constexpr int num_sched_warps      = 1;
      static constexpr int num_look_ahead_warps = 1;

      static constexpr int num_total_warps =
        2 * num_reduce_and_scan_warps + num_load_warps + num_sched_warps + num_look_ahead_warps;
      static constexpr int num_total_threads = num_total_warps * num_threads_per_warp;

      static constexpr int squad_reduce_thread_count = num_reduce_and_scan_warps * num_threads_per_warp;

      static constexpr int tile_size = items_per_thread * squad_reduce_thread_count;

      using SquadDesc = cub::detail::warpspeed::SquadDesc;

      // The squads cannot be static constexpr variables, as those are not device accessible
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadReduce() noexcept
      {
        return SquadDesc{0, num_reduce_and_scan_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadScanStore() noexcept
      {
        return SquadDesc{1, num_reduce_and_scan_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadLoad() noexcept
      {
        return SquadDesc{2, num_load_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadSched() noexcept
      {
        return SquadDesc{3, num_sched_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadLookback() noexcept
      {
        return SquadDesc{4, num_look_ahead_warps};
      }
    };
#  else // USES_WARPSPEED()
#    if TUNE_TRANSPOSE == 0
#      define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#      define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#    else // TUNE_TRANSPOSE == 1
#      define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#      define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#    endif // TUNE_TRANSPOSE

#    if TUNE_LOAD == 0
#      define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#    elif TUNE_LOAD == 1
#      define TUNE_LOAD_MODIFIER cub::LOAD_CA
#    endif // TUNE_LOAD

    template <int NOMINAL_BLOCK_THREADS_4B,
              int NOMINAL_ITEMS_PER_THREAD_4B,
              typename ComputeT,
              cub::BlockLoadAlgorithm LOAD_ALGORITHM,
              cub::CacheLoadModifier LOAD_MODIFIER,
              cub::BlockStoreAlgorithm STORE_ALGORITHM,
              cub::BlockScanAlgorithm SCAN_ALGORITHM>
    using agent_policy_t = cub::AgentScanPolicy<
      NOMINAL_BLOCK_THREADS_4B,
      NOMINAL_ITEMS_PER_THREAD_4B,
      ComputeT,
      LOAD_ALGORITHM,
      LOAD_MODIFIER,
      STORE_ALGORITHM,
      SCAN_ALGORITHM,
      cub::detail::MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>,
      delay_constructor_t>;

    using ScanPolicyT =
      agent_policy_t<TUNE_THREADS,
                     TUNE_ITEMS,
                     AccumT,
                     TUNE_LOAD_ALGORITHM,
                     TUNE_LOAD_MODIFIER,
                     TUNE_STORE_ALGORITHM,
                     cub::BLOCK_SCAN_WARP_SCANS>;
#  endif // USES_WARPSPEED()
  };
};
#endif // TUNE_BASE

template <typename T, typename OffsetT>
static void basic(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using init_t         = T;
  using wrapped_init_t = cub::detail::InputValue<init_t>;
  using accum_t        = ::cuda::std::__accumulator_t<op_t, init_t, T>;
  using input_it_t     = const T*;
  using output_it_t    = T*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;
#if USES_WARPSPEED()
  static_assert(sizeof(offset_t) == sizeof(size_t)); // warpspeed scan uses size_t internally
#endif // USES_WARPSPEED()

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::
    DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  if (sizeof(offset_t) == 4 && elements > std::numeric_limits<offset_t>::max())
  {
    state.skip("Skipping: input size exceeds 32-bit offset type capacity.");
    return;
  }

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  T* d_input  = thrust::raw_pointer_cast(input.data());
  T* d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(
    nullptr,
    tmp_size,
    d_input,
    d_output,
    op_t{},
    wrapped_init_t{T{}},
    static_cast<offset_t>(input.size()),
    0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      d_input,
      d_output,
      op_t{},
      wrapped_init_t{T{}},
      static_cast<offset_t>(input.size()),
      launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_types, scan_offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
