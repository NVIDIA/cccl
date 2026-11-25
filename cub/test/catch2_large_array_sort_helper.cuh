// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/util_type.cuh>

#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

#include <c2h/buffer_utils.cuh>
#include <c2h/catch2_test_helper.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/cpu_timer.h>
#include <c2h/device_policy.h>
#include <c2h/generator_common.h> // seed_t

// #define DEBUG_TIMING

#ifdef DEBUG_TIMING
#  define TIME(expr) expr
#else
#  define TIME(expr) /* no op */ []() {}()
#endif

namespace detail
{
template <typename KeyType>
class key_sort_ref_key_transform
{
  static constexpr double max_key = static_cast<double>(::cuda::std::numeric_limits<KeyType>::max());
  double m_conversion;
  std::size_t m_num_items;
  bool m_is_descending;

public:
  key_sort_ref_key_transform(std::size_t num_items, bool is_descending)
      : m_conversion(max_key / static_cast<double>(num_items))
      , m_num_items(num_items)
      , m_is_descending(is_descending)
  {}

  __host__ __device__ KeyType operator()(std::size_t idx) const
  {
    return m_is_descending ? static_cast<KeyType>(static_cast<double>(m_num_items - 1 - idx) * m_conversion)
                           : static_cast<KeyType>(static_cast<double>(idx) * m_conversion);
  }
};

template <typename KeyType>
struct summary
{
  std::size_t index;
  std::size_t count;
  KeyType key;
};

template <typename KeyType>
struct index_to_summary
{
  using summary_t = summary<KeyType>;

  std::size_t num_items;
  std::size_t num_summaries;
  bool is_descending;

  template <typename IndexType>
  __host__ __device__ summary_t operator()(IndexType idx) const
  {
    constexpr KeyType max_key = ::cuda::std::numeric_limits<KeyType>::max();

    const double key_conversion = static_cast<double>(max_key) / static_cast<double>(num_summaries);
    const KeyType key           = is_descending ? static_cast<KeyType>((num_summaries - 1 - idx) * key_conversion)
                                                : static_cast<KeyType>(idx * key_conversion);

    const std::size_t elements_per_summary = num_items / num_summaries;
    const std::size_t run_index            = idx * elements_per_summary;
    const std::size_t run_size = idx == (num_summaries - 1) ? (num_items - run_index) : elements_per_summary;

    return summary_t{run_index, run_size, key};
  }
};

template <typename KeyType>
class pair_sort_ref_key_transform
{
  static constexpr KeyType max_key = ::cuda::std::numeric_limits<KeyType>::max();

  double m_key_conversion; // Converts summary index to key
  std::size_t m_num_summaries;
  std::size_t m_unpadded_run_size; // typical run size
  bool m_is_descending;

public:
  pair_sort_ref_key_transform(std::size_t num_items, std::size_t num_summaries, bool is_descending)
      : m_key_conversion(static_cast<double>(max_key) / static_cast<double>(num_summaries))
      , m_num_summaries(num_summaries)
      , m_unpadded_run_size(num_items / num_summaries)
      , m_is_descending(is_descending)
  {}

  __host__ __device__ KeyType operator()(std::size_t idx) const
  {
    // The final summary may be padded, so truncate the summary_idx at the last valid idx:
    const std::size_t summary_idx = cuda::std::min(m_num_summaries - 1, idx / m_unpadded_run_size);
    const KeyType key = m_is_descending ? static_cast<KeyType>((m_num_summaries - 1 - summary_idx) * m_key_conversion)
                                        : static_cast<KeyType>(summary_idx * m_key_conversion);

    return key;
  }
};

template <typename ValueType>
struct index_to_value
{
  template <typename IndexType>
  __host__ __device__ ValueType operator()(IndexType index)
  {
    return static_cast<ValueType>(index);
  }
};
} // namespace detail

template <typename KeyType, typename ValueType = cub::NullType>
struct large_array_sort_helper
{
private:
  cuda::stream_ref stream;

public:
  explicit large_array_sort_helper(cuda::stream_ref stream)
      : stream(stream)
      , values_ref(c2h::make_host_buffer<ValueType>(stream, 0, cuda::no_init))
      , keys_in(c2h::make_device_buffer<KeyType>(stream, 0, cuda::no_init))
      , values_in(c2h::make_device_buffer<ValueType>(stream, 0, cuda::no_init))
      , keys_out(c2h::make_device_buffer<KeyType>(stream, 0, cuda::no_init))
      , values_out(c2h::make_device_buffer<ValueType>(stream, 0, cuda::no_init))
  {}

  cuda::host_buffer<ValueType> values_ref;
  cuda::device_buffer<KeyType> keys_in;
  cuda::device_buffer<ValueType> values_in;
  cuda::device_buffer<KeyType> keys_out;
  cuda::device_buffer<ValueType> values_out;

  // Double buffers alias the input and output allocations.
  cub::DoubleBuffer<KeyType> keys_buffer;
  cub::DoubleBuffer<ValueType> values_buffer;

  // Release the output allocations when the sort only needs input buffers.
  void deallocate_outputs()
  {
    keys_buffer   = {};
    values_buffer = {};
    keys_out      = c2h::make_device_buffer<KeyType>(stream, 0, cuda::no_init);
    values_out    = c2h::make_device_buffer<ValueType>(stream, 0, cuda::no_init);
  }

  // Populates keys_in with random KeyTypes. Allocates keys_out and configures keys_buffer appropriately.
  // Allocates a total of 2 * num_items * sizeof(KeyType) device memory and no host memory.
  // Shuffle will allocate some additional device memory overhead for scan temp storage.
  // Pass the sorted output to verify_unstable_key_sort to validate.
  void initialize_for_unstable_key_sort(c2h::seed_t seed, std::size_t num_items, bool is_descending)
  {
    TIME(c2h::cpu_timer timer);

    // Preallocate device memory ASAP so we fail quickly on bad_alloc
    keys_in     = c2h::make_device_buffer<KeyType>(stream, num_items, cuda::no_init);
    keys_out    = c2h::make_device_buffer<KeyType>(stream, num_items, cuda::no_init);
    keys_buffer = cub::DoubleBuffer<KeyType>(keys_in.data(), keys_out.data());

    TIME(timer.print_elapsed_seconds_and_reset("Device Alloc"));

    { // Place the sorted keys into keys_out
      auto key_iter = cuda::make_transform_iterator(
        cuda::counting_iterator(std::size_t{0}), detail::key_sort_ref_key_transform<KeyType>(num_items, is_descending));
      thrust::copy(c2h::device_policy.on(stream.get()), key_iter, key_iter + num_items, keys_out.data());
    }

    TIME(timer.print_elapsed_seconds_and_reset("Generate sorted keys"));

    // shuffle random keys into keys_in
    thrust::shuffle_copy(
      c2h::device_policy.on(stream.get()),
      keys_out.data(),
      keys_out.data() + keys_out.size(),
      keys_in.data(),
      thrust::default_random_engine(static_cast<std::uint32_t>(seed.get())));

    TIME(timer.print_elapsed_seconds_and_reset("Shuffle"));

    // Reset keys_out to remove the valid sorted keys:
    thrust::fill(c2h::device_policy.on(stream.get()), keys_out.data(), keys_out.data() + keys_out.size(), KeyType{});

    TIME(timer.print_elapsed_seconds_and_reset("Reset Output"));
  }

  // Verify the results of sorting the keys_in produced by initialize_for_unstable_key_sort.
  void verify_unstable_key_sort(std::size_t num_items, bool is_descending, const cuda::device_buffer<KeyType>& keys)
  {
    TIME(c2h::cpu_timer timer);
    auto key_iter = cuda::make_transform_iterator(
      cuda::counting_iterator(std::size_t{0}), detail::key_sort_ref_key_transform<KeyType>{num_items, is_descending});
    REQUIRE(thrust::equal(c2h::device_policy.on(stream.get()), keys.data(), keys.data() + keys.size(), key_iter));
    TIME(timer.print_elapsed_seconds_and_reset("Validate keys"));
  }

  // Populates keys_in with random KeyTypes and values_in with sequential ValueTypes.
  // Allocates keys_out and values_out and configures keys_buffer and values_buffer appropriately.
  // values_ref will contain the expected stable sorted values.
  // Allocates 2 * num_items * (sizeof(KeyType) + sizeof(ValueType)) device memory.
  // May allocate up to 2 * num_items * (sizeof(KeyType) + sizeof(ValueType)) on the host.
  // Pass the sorted outputs to verify_stable_pair_sort to validate.
  void initialize_for_stable_pair_sort(c2h::seed_t seed, std::size_t num_items, bool is_descending)
  {
    static_assert(!::cuda::std::is_same_v<ValueType, cub::NullType>, "ValueType must be valid.");
    using summary_t = detail::summary<KeyType>;

    const std::size_t max_summaries = this->compute_max_summaries(num_items);
    const std::size_t num_summaries = this->compute_num_summaries(num_items, max_summaries);

    TIME(c2h::cpu_timer timer);

    {
      auto h_unsorted_keys = c2h::make_host_buffer<KeyType>(stream, num_items, cuda::no_init);
      values_ref           = c2h::make_host_buffer<ValueType>(stream, num_items, cuda::no_init);
      auto h_summaries     = c2h::make_host_buffer<summary_t>(stream, num_summaries, cuda::no_init);

      {
        // Check early that there is room for the key/value arrays.
        auto d_summaries = c2h::make_device_buffer<summary_t>(stream, 2 * max_summaries, cuda::no_init);
        thrust::tabulate(c2h::device_policy.on(stream.get()),
                         d_summaries.data(),
                         d_summaries.data() + num_summaries,
                         detail::index_to_summary<KeyType>{num_items, num_summaries, is_descending});
        c2h::copy_to_host(stream, d_summaries.first(num_summaries), h_summaries);
      }
      TIME(timer.print_elapsed_seconds_and_reset("Generate summaries"));

      using range_t = thrust::random::uniform_int_distribution<std::size_t>::param_type;
      constexpr range_t run_range{1, 256};

      thrust::default_random_engine rng(static_cast<std::uint32_t>(seed.get()));
      thrust::random::uniform_int_distribution<std::size_t> dist;
      std::size_t remaining_summaries = num_summaries;
      for (std::size_t i = 0; i < num_items; /*inc in loop*/)
      {
        const std::size_t summ_idx = dist(rng, range_t{0, remaining_summaries - 1});
        summary_t& summary         = h_summaries[summ_idx];
        const std::size_t run_size = std::min(summary.count, dist(rng, run_range));

        std::fill_n(h_unsorted_keys.data() + i, run_size, summary.key);
        std::iota(
          values_ref.data() + summary.index, values_ref.data() + summary.index + run_size, static_cast<ValueType>(i));

        i += run_size;
        summary.index += run_size;
        summary.count -= run_size;
        if (summary.count == 0)
        {
          --remaining_summaries;
          std::swap(summary, h_summaries[remaining_summaries]);
        }
      }
      REQUIRE(remaining_summaries == 0);
      TIME(timer.print_elapsed_seconds_and_reset("Host-side summary processing"));

      keys_in = c2h::make_device_buffer<KeyType>(stream, h_unsorted_keys);
      stream.sync();
      TIME(timer.print_elapsed_seconds_and_reset("Unsorted keys H->D"));
    }

    // Unsorted values are just a sequence.
    values_in = c2h::make_device_buffer<ValueType>(stream, num_items, cuda::no_init);
    thrust::tabulate(c2h::device_policy.on(stream.get()),
                     values_in.data(),
                     values_in.data() + values_in.size(),
                     detail::index_to_value<ValueType>{});

    TIME(timer.print_elapsed_seconds_and_reset("Unsorted value gen"));

    keys_out   = c2h::make_device_buffer<KeyType>(stream, num_items, cuda::no_init);
    values_out = c2h::make_device_buffer<ValueType>(stream, num_items, cuda::no_init);

    TIME(timer.print_elapsed_seconds_and_reset("Prep device outputs"));

    keys_buffer   = cub::DoubleBuffer<KeyType>(keys_in.data(), keys_out.data());
    values_buffer = cub::DoubleBuffer<ValueType>(values_in.data(), values_out.data());
  }

  // Verify the results of sorting the keys_in produced by initialize_for_stable_pair_sort.
  void verify_stable_pair_sort(
    std::size_t num_items,
    bool is_descending,
    const cuda::device_buffer<KeyType>& keys,
    const cuda::device_buffer<ValueType>& values)
  {
    static_assert(!::cuda::std::is_same_v<ValueType, cub::NullType>, "ValueType must be valid.");

    const std::size_t max_summaries = this->compute_max_summaries(num_items);
    const std::size_t num_summaries = this->compute_num_summaries(num_items, max_summaries);

    TIME(c2h::cpu_timer timer);

    auto ref_key_begin = cuda::make_transform_iterator(
      cuda::counting_iterator(std::size_t{0}),
      detail::pair_sort_ref_key_transform<KeyType>(num_items, num_summaries, is_descending));

    REQUIRE(thrust::equal(c2h::device_policy.on(stream.get()), keys.data(), keys.data() + keys.size(), ref_key_begin));

    TIME(timer.print_elapsed_seconds_and_reset("Validate keys"));

    REQUIRE(values.size() == values_ref.size());
    constexpr std::size_t chunk_size = 1 << 20;
    auto h_values = c2h::make_host_buffer<ValueType>(stream, std::min(num_items, chunk_size), cuda::no_init);
    for (std::size_t offset = 0; offset < num_items; offset += chunk_size)
    {
      const auto count = std::min(chunk_size, num_items - offset);
      auto chunk       = h_values.first(count);
      c2h::copy_to_host(stream, values.subspan(offset, count), chunk);
      CAPTURE(offset);
      REQUIRE(std::equal(chunk.begin(), chunk.end(), values_ref.data() + offset));
    }

    TIME(timer.print_elapsed_seconds_and_reset("Validate values"));
  }

private:
  // The maximum number of summaries that will fill the target memory footprint of one full set of key/value pairs.
  static std::size_t compute_max_summaries(std::size_t num_items)
  {
    using summary_t = detail::summary<KeyType>;

    const std::size_t max_summary_mem = num_items * (sizeof(KeyType) + sizeof(ValueType));
    const std::size_t max_summaries   = ::cuda::ceil_div(max_summary_mem, sizeof(summary_t));
    return max_summaries;
  }

  // The actual number of summaries to use, considering memory, key type, and number of items.
  static std::size_t compute_num_summaries(std::size_t num_items, std::size_t max_summaries)
  {
    constexpr KeyType max_key       = ::cuda::std::numeric_limits<KeyType>::max();
    const std::size_t num_summaries = std::min({max_summaries, num_items, static_cast<std::size_t>(max_key)});
    return num_summaries;
  }
};
