// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_select.cuh>

#include <cuda/std/__algorithm_>

#include <limits>

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
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using SelectIfPolicyT =
      cub::AgentSelectIfPolicy<TUNE_THREADS_PER_BLOCK,
                               TUNE_ITEMS_PER_THREAD,
                               TUNE_LOAD_ALGORITHM,
                               TUNE_LOAD_MODIFIER,
                               cub::BLOCK_SCAN_WARP_SCANS,
                               delay_constructor_t>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT, typename InPlace>
static void unique(nvbench::state& state, nvbench::type_list<T, OffsetT, InPlace>)
{
  using input_it_t        = const T*;
  using flag_it_t         = cub::NullType*;
  using output_it_t       = T*;
  using num_selected_it_t = OffsetT*;
  using select_op_t       = cub::NullType;
  using equality_op_t     = ::cuda::std::equal_to<>;
  using offset_t          = OffsetT;
  constexpr cub::SelectImpl selection_option =
    InPlace::value ? cub::SelectImpl::SelectPotentiallyInPlace : cub::SelectImpl::Select;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchSelectIf<
    input_it_t,
    flag_it_t,
    output_it_t,
    num_selected_it_t,
    select_op_t,
    equality_op_t,
    offset_t,
    selection_option,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchSelectIf<input_it_t,
                          flag_it_t,
                          output_it_t,
                          num_selected_it_t,
                          select_op_t,
                          equality_op_t,
                          offset_t,
                          selection_option>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<T> in = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);
  thrust::device_vector<T> out(elements);
  thrust::device_vector<offset_t> num_unique_out(1);

  input_it_t d_in                = thrust::raw_pointer_cast(in.data());
  output_it_t d_out              = thrust::raw_pointer_cast(out.data());
  flag_it_t d_flags              = nullptr;
  num_selected_it_t d_num_unique = thrust::raw_pointer_cast(num_unique_out.data());

  // Get temporary storage requirements
  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_flags, d_out, d_num_unique, select_op_t{}, equality_op_t{}, elements, 0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  // Get number of unique elements
  dispatch_t::Dispatch(
    temp_storage, temp_size, d_in, d_flags, d_out, d_num_unique, select_op_t{}, equality_op_t{}, elements, 0);

  cudaDeviceSynchronize();
  const OffsetT num_unique = num_unique_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(num_unique);
  state.add_global_memory_writes<offset_t>(1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_flags,
      d_out,
      d_num_unique,
      select_op_t{},
      equality_op_t{},
      elements,
      launch.get_stream());
  });
}

using ::cuda::std::false_type;
using ::cuda::std::true_type;
#ifdef TUNE_InPlace
using is_in_place = nvbench::type_list<TUNE_InPlace>; // expands to "false_type" or "true_type"
#else // !defined(TUNE_InPlace)
using is_in_place = nvbench::type_list<false_type, true_type>;
#endif // TUNE_InPlace

// The implementation of DeviceSelect for 64-bit offset types uses a streaming approach, where it runs multiple passes
// using a 32-bit offset type, so we only need to test one (to save time for tuning and the benchmark CI).
using select_offset_types = nvbench::type_list<int64_t>;

NVBENCH_BENCH_TYPES(unique, NVBENCH_TYPE_AXES(fundamental_types, select_offset_types, is_in_place))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "InPlace{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
