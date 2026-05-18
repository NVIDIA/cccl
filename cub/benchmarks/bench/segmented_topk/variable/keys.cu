// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>

#include <cuda/iterator>
#include <cuda/random>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/random>

#include <stdexcept>
#include <string>
#include <vector>

#include <nvbench_helper.cuh>

namespace
{
enum class pattern_kind : int
{
  random = 0,
  quantized_random,
  relu_quantized,
  tie_heavy,
  pivot_tie
};

[[nodiscard]] pattern_kind string_to_pattern(const std::string& pattern)
{
  if (pattern == "random")
  {
    return pattern_kind::random;
  }
  if (pattern == "quantized_random")
  {
    return pattern_kind::quantized_random;
  }
  if (pattern == "relu_quantized")
  {
    return pattern_kind::relu_quantized;
  }
  if (pattern == "tie_heavy")
  {
    return pattern_kind::tie_heavy;
  }
  if (pattern == "pivot_tie")
  {
    return pattern_kind::pivot_tie;
  }
  throw std::runtime_error("Invalid Pattern axis value: " + pattern);
}

template <int MaxSegmentSize, int K>
[[nodiscard]] thrust::device_vector<float>
gen_data(int num_segments, pattern_kind pattern, const cuda::std::int64_t* d_seg_sizes)
{
  const auto num_keys = static_cast<std::size_t>(num_segments) * static_cast<std::size_t>(MaxSegmentSize);
  auto d_keys         = thrust::device_vector<float>{num_keys, thrust::no_init};

  // gt_count == "greater-than count": number of 2.0 values placed at the tail of each segment's live region.
  constexpr int gt_count = cuda::std::max(1, cuda::std::min(K / 4, MaxSegmentSize / 8));

  thrust::tabulate(d_keys.begin(), d_keys.end(), [pattern, d_seg_sizes] __device__(std::size_t idx) -> float {
    auto quantize = [](float base) -> float {
      const auto r         = cuda::std::rint(base);
      const auto scaled_fr = cuda::std::rint((base - r) * 32.0f);
      return r + (scaled_fr / 32.0f);
    };

    auto random_value = [](unsigned long long idx) -> float {
      cuda::pcg64 rng(42);
      rng.discard(idx);
      cuda::std::normal_distribution<float> normal(0.f, 1.f);
      return normal(rng);
    };

    const auto j = static_cast<int>(idx % MaxSegmentSize);
    switch (pattern)
    {
      //               ##
      //              ####
      //            ########
      //          ############
      //        ################
      //     ######################
      //  ##############################
      //  ------------------------------
      //  -3             0             3
      case pattern_kind::random:
        return random_value(idx);

      //                |
      //                |
      //            |   |   |
      //        |   |   |   |   |
      //    |   |   |   |   |   |   |
      //  ----------------------------
      //  -3            0            3
      case pattern_kind::quantized_random:
        return quantize(random_value(idx));

      //  |
      //  |
      //  |
      //  |
      //  |
      //  |   |
      //  |   |   |
      //  |   |   |   |   |
      //  |   |   |   |   |   |   |
      //  ----------------------------
      //  0                          3
      case pattern_kind::relu_quantized:
        return quantize(cuda::std::max(random_value(idx), 0.f));

      //  |   |   |   |   |   |   |   |
      //  |   |   |   |   |   |   |   |
      //  |   |   |   |   |   |   |   |
      //  --------------------------------
      //  0/64                      63/64
      case pattern_kind::tie_heavy:
        return static_cast<float>(j % 64) / 64.f;

      //  |
      //  |
      //  |
      //  |
      //  |
      //  |
      //  |
      //  |                         |
      //  ----------------------------
      //  1.0                       2.0
      case pattern_kind::pivot_tie: {
        const auto seg_size = static_cast<int>(d_seg_sizes[idx / MaxSegmentSize]);
        return (j >= seg_size - gt_count) ? 2.f : 1.f;
      }
      default:
        _CCCL_UNREACHABLE();
    }
  });

  return d_keys;
}
} // namespace

const std::vector<std::string> valid_patterns = {
  "random", "quantized_random", "relu_quantized", "tie_heavy", "pivot_tie"};

template <typename KeyT, int MaxSegmentSize, int K>
void variable_seg_size_topk_keys(nvbench::state& state,
                                 nvbench::type_list<KeyT, nvbench::enum_type<MaxSegmentSize>, nvbench::enum_type<K>>)
{
  if constexpr (K > MaxSegmentSize)
  {
    state.skip("K > MaxSegmentSize.");
    return;
  }

  const auto num_segments                                         = static_cast<int>(state.get_int64("NumSegments"));
  const thrust::device_vector<cuda::std::int64_t> d_segment_sizes = generate(
    static_cast<std::size_t>(num_segments),
    bit_entropy::_1_000,
    static_cast<cuda::std::int64_t>(K),
    static_cast<cuda::std::int64_t>(MaxSegmentSize));
  const auto input_elements  = thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end());
  const auto output_elements = static_cast<std::size_t>(num_segments) * K;
  const auto total_num_items =
    cub::detail::batched_topk::total_num_items_guarantee<1, cuda::std::numeric_limits<cuda::std::int64_t>::max()>{
      static_cast<cuda::std::int64_t>(input_elements)};

  auto in_keys_buffer = gen_data<MaxSegmentSize, K>(
    num_segments, string_to_pattern(state.get_string("Pattern")), thrust::raw_pointer_cast(d_segment_sizes.data()));
  auto out_keys_buffer = thrust::device_vector<KeyT>(output_elements, thrust::no_init);

  cub::detail::batched_topk::segment_size_per_segment<const cuda::std::int64_t*, 1, MaxSegmentSize> segment_sizes_param{
    thrust::raw_pointer_cast(d_segment_sizes.data())};
  cub::detail::batched_topk::k_static<K> k_param{};
  cub::detail::batched_topk::select_direction_static<cub::detail::topk::select::max> select_directions{};
  cub::detail::batched_topk::num_segments_uniform<> num_segments_uniform_param{
    static_cast<cuda::std::int64_t>(num_segments)};

  auto d_keys_in = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(in_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(MaxSegmentSize));
  auto d_keys_out = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(out_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(K));

  state.add_element_count(input_elements, "NumElements");
  state.add_global_memory_reads<KeyT>(input_elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(output_elements, "OutputKeys");

  size_t temp_size{};
  cub::detail::batched_topk::dispatch(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    static_cast<cub::NullType**>(nullptr),
    static_cast<cub::NullType**>(nullptr),
    segment_sizes_param,
    k_param,
    select_directions,
    num_segments_uniform_param,
    total_num_items,
    nullptr);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::batched_topk::dispatch(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes_param,
      k_param,
      select_directions,
      num_segments_uniform_param,
      total_num_items,
      launch.get_stream());
  });
}

using key_type_list = nvbench::type_list<float>;

using max_segment_size_list = nvbench::enum_type_list< //
  512,
  1024,
  2048,
  4096,
  8192
#if 0 // need these, waiting for implementation to catch up
  ,
  16384,
  32768,
  65536,
  131072,
  262144,
  524288,
  1048576
#endif
  >;

using k_list = nvbench::enum_type_list<512, 1024, 2048>;

NVBENCH_BENCH_TYPES(variable_seg_size_topk_keys, NVBENCH_TYPE_AXES(key_type_list, max_segment_size_list, k_list))
  .set_name("decode_style_variable_topk")
  .set_type_axes_names({"KeyT{ct}", "MaxSegmentSize{ct}", "K{ct}"})
  .add_int64_axis("NumSegments", {1, 2, 4, 8, 16, 32})
  .add_string_axis("Pattern", valid_patterns);
