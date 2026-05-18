// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_partition.cuh>

#include <thrust/count.h>

#include <cuda/std/algorithm>
#include <cuda/std/type_traits>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  else // TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename InputT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::select::select_if_policy
  {
    return {TUNE_THREADS_PER_BLOCK,
            TUNE_ITEMS_PER_THREAD,
            TUNE_LOAD_ALGORITHM,
            TUNE_LOAD_MODIFIER,
            cub::BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy};
  }
};
#endif // TUNE_BASE

template <typename FlagsItT, typename T, typename OffsetT>
void init_output_partition_buffer(
  FlagsItT d_flags,
  OffsetT num_items,
  T* d_out,
  cub::detail::select::partition_distinct_output_t<T*, T*>& d_partition_out_buffer)
{
  const auto selected_elements = thrust::count(d_flags, d_flags + num_items, true);
  d_partition_out_buffer = cub::detail::select::partition_distinct_output_t<T*, T*>{d_out, d_out + selected_elements};
}

template <typename FlagsItT, typename T, typename OffsetT>
void init_output_partition_buffer(FlagsItT, OffsetT, T* d_out, T*& d_partition_out_buffer)
{
  d_partition_out_buffer = d_out;
}

template <typename T, typename OffsetT, typename UseDistinctPartitionT>
void flagged(nvbench::state& state, nvbench::type_list<T, OffsetT, UseDistinctPartitionT>)
{
  using offset_t                             = OffsetT;
  constexpr bool use_distinct_out_partitions = UseDistinctPartitionT::value;
  using output_it_t                          = typename ::cuda::std::
    conditional<use_distinct_out_partitions, cub::detail::select::partition_distinct_output_t<T*, T*>, T*>::type;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  auto generator = generate(elements, entropy);

  thrust::device_vector<T> in       = generator;
  thrust::device_vector<bool> flags = generator;
  thrust::device_vector<offset_t> num_selected(1);
  thrust::device_vector<T> out(elements);

  const T* d_in            = thrust::raw_pointer_cast(in.data());
  const bool* d_flags      = thrust::raw_pointer_cast(flags.data());
  offset_t* d_num_selected = thrust::raw_pointer_cast(num_selected.data());
  output_it_t d_out{};
  init_output_partition_buffer(flags.cbegin(), elements, thrust::raw_pointer_cast(out.data()), d_out);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<bool>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<offset_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<T>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DevicePartition::Flagged,
      "Flagged failed",
      d_in,
      d_flags,
      d_out,
      d_num_selected,
      static_cast<offset_t>(elements),
      env);
  });
}

using ::cuda::std::false_type;
using ::cuda::std::true_type;
#ifdef TUNE_DistinctPartitions
using distinct_partitions = nvbench::type_list<TUNE_DistinctPartitions>; // expands to "false_type" or "true_type"
#else // !defined(TUNE_DistinctPartitions)
using distinct_partitions = nvbench::type_list<false_type, true_type>;
#endif // TUNE_DistinctPartitions

NVBENCH_BENCH_TYPES(flagged, NVBENCH_TYPE_AXES(fundamental_types, offset_types, distinct_partitions))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "DistinctPartitions{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
