// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <cuda/iterator>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1

#if TUNE_BLOCK_LOAD_ALGORITHM == 0
#  define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#elif TUNE_BLOCK_LOAD_ALGORITHM == 1
#  define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#elif TUNE_BLOCK_LOAD_ALGORITHM == 2
#  define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_VECTORIZE
#endif

#if !TUNE_BASE
template <class KeyInT, class OffsetT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int items_per_thread = TUNE_ITEMS_PER_THREAD;

    static constexpr int bits_per_pass = cub::detail::topk::calc_bits_per_pass<KeyInT>();

    using topk_policy_t =
      cub::detail::topk::AgentTopKPolicy<TUNE_THREADS_PER_BLOCK,
                                         items_per_thread,
                                         bits_per_pass,
                                         TUNE_LOAD_ALGORITHM,
                                         cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename KeyT, int MaxSegmentSize, int MaxNumSelected>
void fixed_seg_size_topk_keys(
  nvbench::state& state,
  nvbench::type_list<KeyT, nvbench::enum_type<MaxSegmentSize>, nvbench::enum_type<MaxNumSelected>>)
{
  // Range of guaranteed total number of items
  constexpr auto min_num_total_items = 1;
  constexpr auto max_num_total_items = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();

  // Iterator types
  using key_input_it_t  = cuda::strided_iterator<cuda::counting_iterator<KeyT*>>;
  using key_output_it_t = cuda::strided_iterator<cuda::counting_iterator<KeyT*>>;

  // Static segment size
  using seg_size_t = cub::detail::segmented_topk::segment_size_static<MaxSegmentSize>;

  // Static k (number of selected output elements per segment)
  using k_value_t = cub::detail::segmented_topk::k_static<MaxNumSelected>;

  // Static selection direction (max)
  using select_direction_value_t = cub::detail::segmented_topk::select_direction_static<cub::detail::topk::select::max>;

  // Number of segments is a host-accessible value
  using num_segments_uniform_t = cub::detail::segmented_topk::num_segments_uniform<>;

  // Total number of items guarantee type
  using total_num_items_guarantee_t =
    cub::detail::segmented_topk::total_num_items_guarantee<min_num_total_items, max_num_total_items>;

  using dispatch_t = cub::detail::segmented_topk::DispatchBatchedTopK<
    key_input_it_t,
    key_output_it_t,
    cub::NullType**,
    cub::NullType**,
    seg_size_t,
    k_value_t,
    select_direction_value_t,
    num_segments_uniform_t,
    total_num_items_guarantee_t
#if !TUNE_BASE
    ,
    policy_hub_t<KeyT, cub::NullType, ::cuda::std::int32_t, MaxNumSelected>
#endif // !TUNE_BASE
    >;

  // Retrieve axis parameters
  const auto max_elements      = static_cast<size_t>(state.get_int64("Elements{io}"));
  const auto segment_size      = static_cast<::cuda::std::ptrdiff_t>(MaxSegmentSize);
  const auto selected_elements = static_cast<::cuda::std::ptrdiff_t>(MaxNumSelected);
  const auto num_segments      = ::cuda::std::max<std::size_t>(1, (max_elements / segment_size));
  const auto elements          = num_segments * segment_size;
  const auto total_num_items   = total_num_items_guarantee_t{static_cast<::cuda::std::int64_t>(elements)};
  const bit_entropy entropy    = str_to_entropy(state.get_string("Entropy"));

  // Skip workloads where k exceeds the segment size
  if (selected_elements >= segment_size)
  {
    state.skip("Skipping workload where K >= SegmentSize.");
    return;
  }

  thrust::device_vector<KeyT> in_keys_buffer = generate(elements, entropy);
  thrust::device_vector<KeyT> out_keys_buffer(selected_elements * num_segments, thrust::no_init);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(in_keys_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(out_keys_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), selected_elements);

  auto segment_sizes     = seg_size_t{};
  auto k                 = k_value_t{};
  auto select_directions = select_direction_value_t{};

  state.add_element_count(elements, "NumElements");
  state.add_element_count(segment_size, "SegmentSize");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements * num_segments, "OutputKeys");

  // allocate temporary storage
  size_t temp_size;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    static_cast<cub::NullType**>(nullptr),
    static_cast<cub::NullType**>(nullptr),
    segment_sizes,
    k,
    select_directions,
    num_segments,
    total_num_items,
    0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  // run the algorithm
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k,
      select_directions,
      num_segments,
      total_num_items,
      launch.get_stream());
  });
}

using key_type_list          = nvbench::type_list<float>;
using segment_size_type_list = nvbench::type_list<uint32_t>;
using out_offset_type_list   = nvbench::type_list<uint32_t>;

using segment_size_        = nvbench::type_list<uint32_t>;
using out_offset_type_list = nvbench::type_list<uint32_t>;

using small_segment_size_list = nvbench::enum_type_list<64, 128, 256, 512, 1024>;
using small_k_list            = nvbench::enum_type_list<8, 16, 32, 128, 512, 1024>;

NVBENCH_BENCH_TYPES(fixed_seg_size_topk_keys, NVBENCH_TYPE_AXES(key_type_list, small_segment_size_list, small_k_list))
  .set_name("small")
  .set_type_axes_names({"KeyT{ct}", "MaxSegmentSize{ct}", "MaxNumSelected{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(28, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"});
