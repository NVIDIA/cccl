//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/bloom_filter.cuh>
#include <cuda/experimental/__cuco/bloom_filter_policy.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>

#include "../common/defaults.cuh"
#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;
namespace bench = cudax::cuco::benchmark;

/**
 * @brief A benchmark evaluating `cudax::cuco::bloom_filter::add_async` performance.
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void bloom_filter_add(
  nvbench::state& state,
  nvbench::type_list<Key,
                     Word,
                     nvbench::enum_type<BlockBits>,
                     nvbench::enum_type<PatternBits>,
                     nvbench::enum_type<HorizontalLayout>,
                     nvbench::enum_type<VerticalLayout>>)
{
  constexpr auto words_per_block       = BlockBits / cuda::std::numeric_limits<Word>::digits;
  constexpr auto pattern_bits_per_word = (words_per_block == 0) ? 0 : PatternBits / words_per_block;

  if constexpr (!cuda::std::has_single_bit(static_cast<cuda::std::uint32_t>(BlockBits)) || words_per_block == 0)
  {
    state.skip("Invalid filter block size");
  }
  else if constexpr (HorizontalLayout * VerticalLayout != words_per_block)
  {
    state.skip("Invalid vectorization layout");
  }
  else if constexpr (pattern_bits_per_word <= 0 || pattern_bits_per_word > cuda::std::numeric_limits<Word>::digits
                     || pattern_bits_per_word * words_per_block > 64)
  {
    state.skip("Invalid pattern bits per word");
  }
  else
  {
    // `contains` is fully vertical so only the `add` layout is swept here
    constexpr auto contains_horizontal_layout = 1;
    constexpr auto contains_vertical_layout   = words_per_block;

    using policy_type = cudax::cuco::bloom_filter_policy<
      Key,
      cudax::cuco::hash<Key, cudax::cuco::hash_algorithm::xxhash_64>,
      Word,
      words_per_block,
      PatternBits,
      HorizontalLayout,
      VerticalLayout,
      contains_horizontal_layout,
      contains_vertical_layout>;
    using filter_type =
      cudax::cuco::bloom_filter<Key, cuda::std::dynamic_extent, cuda::thread_scope_device, policy_type>;

    const auto num_keys       = state.get_int64("NumInputs");
    const auto filter_size_mb = state.get_int64("FilterSizeMB");

    const auto num_blocks = static_cast<cuda::std::size_t>(filter_size_mb) * 1024 * 1024
                          / (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

    if (num_blocks > policy_type::max_filter_blocks)
    {
      state.skip("num_blocks exceeds max_filter_blocks");
      return;
    }

    const auto device = cuda::device_ref{0};
    cuda::stream stream{device};
    const cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(device);

    auto keys = cuda::make_device_buffer<Key>(stream, device, num_keys, cuda::no_init);
    thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), keys.begin(), keys.end(), Key{0});

    filter_type filter{stream, mr, num_blocks};
    stream.sync();

    state.add_element_count(num_keys);
    state.add_global_memory_reads<Key>(num_keys, "InputSize");

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      filter.add_async({launch.get_stream()}, keys.begin(), keys.end());
      timer.stop();
      filter.clear_async({launch.get_stream()});
    });
  }
}

// Default benchmark: the single layout matching the default `cudax::cuco::bloom_filter_policy`
NVBENCH_BENCH_TYPES(
  bloom_filter_add,
  NVBENCH_TYPE_AXES(
    nvbench::type_list<bench::defaults::bloom_filter_key_type>,
    nvbench::type_list<nvbench::uint32_t>, ///< Word
    nvbench::enum_type_list<256>, ///< BlockBits
    nvbench::enum_type_list<8>, ///< PatternBits
    nvbench::enum_type_list<8>, ///< HorizontalLayout
    nvbench::enum_type_list<1> ///< VerticalLayout
    ))
  .set_name("bloom_filter_add_unique_size")
  .set_type_axes_names({"Key", "Word", "BlockBits", "PatternBits", "HorizontalLayout", "VerticalLayout"})
  .add_int64_axis("NumInputs", {bench::defaults::bloom_filter_n})
  .add_int64_axis("FilterSizeMB", bench::defaults::bloom_filter_size_mb_range_cache);
