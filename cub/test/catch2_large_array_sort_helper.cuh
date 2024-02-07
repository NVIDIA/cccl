/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/util_type.cuh>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/memory.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <numeric>

#include <c2h/cpu_timer.cuh>
#include <c2h/device_policy.cuh>
#include <c2h/vector.cuh>
#include <catch2_test_helper.h>

// #define DEBUG_TIMING

#ifdef DEBUG_TIMING
#  define TIME(expr) expr
#else
#  define TIME(expr) /* no op */ []() {}()
#endif

namespace detail
{

template <typename key_type>
class key_sort_ref_key_transform : public thrust::unary_function<std::size_t, key_type>
{
  static constexpr double max_key = static_cast<double>(::cuda::std::numeric_limits<key_type>::max());
  const double m_conversion;
  std::size_t m_num_items;
  bool m_is_descending;

public:
  key_sort_ref_key_transform(std::size_t num_items, bool is_descending)
      : m_conversion(max_key / num_items)
      , m_num_items(num_items)
      , m_is_descending(is_descending)
  {}

  _CCCL_HOST_DEVICE key_type operator()(std::size_t idx) const
  {
    return m_is_descending ? static_cast<key_type>((m_num_items - 1 - idx) * m_conversion)
                           : static_cast<key_type>(idx * m_conversion);
  }
};

template <typename key_type>
struct summary
{
  std::size_t index;
  std::size_t count;
  key_type key;
};

template <typename key_type>
struct index_to_summary
{
  using summary_t = summary<key_type>;

  std::size_t num_items;
  std::size_t num_summaries;
  bool is_descending;

  template <typename index_type>
  _CCCL_HOST_DEVICE summary_t operator()(index_type idx) const
  {
    constexpr key_type max_key = ::cuda::std::numeric_limits<key_type>::max();

    const double key_conversion = static_cast<double>(max_key) / static_cast<double>(num_summaries);
    const key_type key          = is_descending ? static_cast<key_type>((num_summaries - 1 - idx) * key_conversion)
                                                : static_cast<key_type>(idx * key_conversion);

    const std::size_t elements_per_summary = num_items / num_summaries;
    const std::size_t run_index            = idx * elements_per_summary;
    const std::size_t run_size = idx == (num_summaries - 1) ? (num_items - run_index) : elements_per_summary;

    return summary_t{run_index, run_size, key};
  }
};

template <typename key_type>
class key_value_sort_ref_key_transform : public thrust::unary_function<std::size_t, key_type>
{
  static constexpr key_type max_key = ::cuda::std::numeric_limits<key_type>::max();

  double m_key_conversion; // Converts summary index to key
  std::size_t m_num_summaries;
  std::size_t m_unpadded_run_size; // typical run size
  bool m_is_descending;

public:
  key_value_sort_ref_key_transform(std::size_t num_items, std::size_t num_summaries, bool is_descending)
      : m_key_conversion(static_cast<double>(max_key) / static_cast<double>(num_summaries))
      , m_num_summaries(num_summaries)
      , m_unpadded_run_size(num_items / num_summaries)
      , m_is_descending(is_descending)
  {}

  _CCCL_HOST_DEVICE key_type operator()(std::size_t idx) const
  {
    // The final summary may be padded, so truncate the summary_idx at the last valid idx:
    const std::size_t summary_idx = thrust::min(m_num_summaries - 1, idx / m_unpadded_run_size);
    const key_type key = m_is_descending ? static_cast<key_type>((m_num_summaries - 1 - summary_idx) * m_key_conversion)
                                         : static_cast<key_type>(summary_idx * m_key_conversion);

    return key;
  }
};

template <typename value_type>
struct index_to_value
{
  template <typename index_type>
  _CCCL_HOST_DEVICE value_type operator()(index_type index)
  {
    return static_cast<value_type>(index);
  }
};

} // namespace detail

template <typename key_type, typename value_type = cub::NullType>
struct large_array_sort_helper
{
  // Sorted keys/values in host memory
  // (May be unused if results can be verified with fancy iterators)
  c2h::host_vector<key_type> keys_ref;
  c2h::host_vector<value_type> values_ref;

  // Unsorted keys/values in device memory
  c2h::device_vector<key_type> keys_in;
  c2h::device_vector<value_type> values_in;

  // Allocated device memory for output keys/values
  c2h::device_vector<key_type> keys_out;
  c2h::device_vector<value_type> values_out;

  // Double buffer for keys/values. Aliases the in/out arrays.
  cub::DoubleBuffer<key_type> keys_buffer;
  cub::DoubleBuffer<value_type> values_buffer;

  // By default, both input and output arrays are allocated to ensure that 2 * num_items * (sizeof(key_type) +
  // sizeof(value_type)) device memory is available at the start of the initialize_* methods. This ensures that we'll
  // fail quickly if the problem size exceeds the necessary storage required for sorting. If the output arrays are not
  // being used (e.g. in-place merge sort API with temporary storage allocation), these may be freed easily by calling
  // this method:
  void deallocate_outputs()
  {
    keys_out.clear();
    keys_out.shrink_to_fit();
    values_out.clear();
    values_out.shrink_to_fit();
  }

  // Populates keys_in with random key_types. Allocates keys_out and configures keys_buffer appropriately.
  // Allocates a total of 2 * num_items * sizeof(key_type) device memory and no host memory.
  // Shuffle will allocate some additional device memory overhead for scan temp storage.
  // Pass the sorted output to verify_unstable_key_sort to validate.
  void initialize_for_unstable_key_sort(std::size_t num_items, bool is_descending)
  {
    TIME(c2h::cpu_timer timer);

    // Preallocate device memory ASAP so we fail quickly on bad_alloc
    keys_in.resize(num_items);
    keys_out.resize(num_items);
    keys_buffer =
      cub::DoubleBuffer<key_type>(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()));

    TIME(timer.print_elapsed_seconds_and_reset("Device Alloc"));

    { // Place the sorted keys into keys_out
      auto key_iter = thrust::make_transform_iterator(
        thrust::make_counting_iterator(std::size_t{0}),
        detail::key_sort_ref_key_transform<key_type>(num_items, is_descending));
      thrust::copy(c2h::device_policy, key_iter, key_iter + num_items, keys_out.begin());
    }

    TIME(timer.print_elapsed_seconds_and_reset("Generate sorted keys"));

    // shuffle random keys into keys_in
    thrust::shuffle_copy(
      c2h::device_policy, keys_out.cbegin(), keys_out.cend(), keys_in.begin(), thrust::default_random_engine{});

    TIME(timer.print_elapsed_seconds_and_reset("Shuffle"));

    // Reset keys_out to remove the valid sorted keys:
    thrust::fill(c2h::device_policy, keys_out.begin(), keys_out.end(), key_type{});

    TIME(timer.print_elapsed_seconds_and_reset("Reset Output"));
  }

  // Verify the results of sorting the keys_in produced by initialize_for_unstable_key_sort.
  void verify_unstable_key_sort(std::size_t num_items, bool is_descending, const c2h::device_vector<key_type>& keys)
  {
    TIME(c2h::cpu_timer timer);
    auto key_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(std::size_t{0}),
      detail::key_sort_ref_key_transform<key_type>{num_items, is_descending});
    REQUIRE(thrust::equal(c2h::device_policy, keys.cbegin(), keys.cend(), key_iter));
    TIME(timer.print_elapsed_seconds_and_reset("Validate keys"));
  }

  // Populates keys_in with random key_types and values_in with sequential value_types.
  // Allocates keys_out and values_out and configures keys_buffer and values_buffer appropriately.
  // values_ref will contain the expected stable sorted values.
  // Allocates 2 * num_items * (sizeof(key_type) + sizeof(value_type)) device memory.
  // May allocate up to 2 * num_items * (sizeof(key_type) + sizeof(value_type)) on the host.
  // Pass the sorted outputs to verify_stable_key_value_sort to validate.
  void initialize_for_stable_key_value_sort(std::size_t num_items, bool is_descending)
  {
    static_assert(!::cuda::std::is_same<value_type, cub::NullType>::value, "value_type must be valid.");
    using summary_t            = detail::summary<key_type>;
    constexpr key_type max_key = ::cuda::std::numeric_limits<key_type>::max();

    const std::size_t max_summary_mem = num_items * (sizeof(key_type) + sizeof(value_type));
    const std::size_t max_summaries   = cub::DivideAndRoundUp(max_summary_mem, sizeof(summary_t));
    const std::size_t num_summaries   = std::min(std::min(max_summaries, num_items), static_cast<std::size_t>(max_key));

    TIME(c2h::cpu_timer timer);

    c2h::device_vector<summary_t> d_summaries;
    // Overallocate -- if this fails, there won't be be enough free device memory for the input/output arrays.
    // Better to fail now before spending time computing the inputs/outputs.
    d_summaries.reserve(2 * max_summaries);
    d_summaries.resize(num_summaries);

    TIME(timer.print_elapsed_seconds_and_reset("Device allocate"));

    // Populate the summaries using evenly spaced keys and constant sized runs, padding the last run to fill.
    thrust::tabulate(c2h::device_policy,
                     d_summaries.begin(),
                     d_summaries.end(),
                     detail::index_to_summary<key_type>{num_items, num_summaries, is_descending});

    TIME(timer.print_elapsed_seconds_and_reset("idx -> summary"));

    // Copy the summaries to host memory and release device summary memory.
    c2h::host_vector<summary_t> h_summaries = d_summaries;

    TIME(timer.print_elapsed_seconds_and_reset("D->H Summaries"));

    d_summaries.clear();
    d_summaries.shrink_to_fit();

    TIME(timer.print_elapsed_seconds_and_reset("Free device summaries"));

    // Build the unsorted key and reference value arrays on host:
    c2h::host_vector<key_type> h_unsorted_keys(num_items);
    c2h::host_vector<value_type> h_sorted_values(num_items);

    TIME(timer.print_elapsed_seconds_and_reset("Host allocate"));

    {
      using range_t = typename thrust::random::uniform_int_distribution<std::size_t>::param_type;
      constexpr range_t run_range{1, 256};

      thrust::default_random_engine rng{};
      thrust::random::uniform_int_distribution<std::size_t> dist;
      range_t summary_range{0, num_summaries - 1};
      for (std::size_t i = 0; i < num_items; /*inc in loop*/)
      {
        const std::size_t summ_idx = dist(rng, summary_range);
        summary_t& summary         = h_summaries[summ_idx];
        const std::size_t run_size = std::min(summary.count, dist(rng, run_range));

        std::fill(h_unsorted_keys.begin() + i, // formatting
                  h_unsorted_keys.begin() + i + run_size,
                  summary.key);
        std::iota(h_sorted_values.begin() + summary.index, // formatting
                  h_sorted_values.begin() + summary.index + run_size,
                  static_cast<value_type>(i));

        i += run_size;
        summary.index += run_size;
        summary.count -= run_size;
        if (summary.count == 0)
        {
          using std::swap;
          swap(summary, h_summaries.back());
          h_summaries.pop_back();
          summary_range.second -= 1;
        }
      }
    }

    TIME(timer.print_elapsed_seconds_and_reset("Host-side summary processing"));

    // Release the host summary memory.
    REQUIRE(h_summaries.empty());
    h_summaries.shrink_to_fit();

    TIME(timer.print_elapsed_seconds_and_reset("Host summaries free"));

    // Copy the unsorted keys to device
    keys_in = h_unsorted_keys;
    h_unsorted_keys.clear();
    h_unsorted_keys.shrink_to_fit();

    TIME(timer.print_elapsed_seconds_and_reset("Unsorted keys H->D"));

    // Unsorted values are just a sequence
    values_in.resize(num_items);
    thrust::tabulate(c2h::device_policy, values_in.begin(), values_in.end(), detail::index_to_value<value_type>{});

    TIME(timer.print_elapsed_seconds_and_reset("Unsorted value gen"));

    // Copy the sorted values to the member array.
    // Sorted keys are verified using a fancy iterator.
    values_ref = std::move(h_sorted_values); // Same memory space, just move.

    TIME(timer.print_elapsed_seconds_and_reset("Copy/move refs"));

    keys_out.resize(num_items);
    values_out.resize(num_items);

    TIME(timer.print_elapsed_seconds_and_reset("Prep device outputs"));

    keys_buffer =
      cub::DoubleBuffer<key_type>(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()));
    values_buffer = cub::DoubleBuffer<value_type>(
      thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()));
  }

  // Verify the results of sorting the keys_in produced by initialize_for_stable_key_value_sort.
  void verify_stable_key_value_sort(
    std::size_t num_items,
    bool is_descending,
    const c2h::device_vector<key_type>& keys,
    const c2h::device_vector<value_type>& values)
  {
    static_assert(!::cuda::std::is_same<value_type, cub::NullType>::value, "value_type must be valid.");
    using summary_t            = detail::summary<key_type>;
    constexpr key_type max_key = ::cuda::std::numeric_limits<key_type>::max();

    const std::size_t max_summary_mem = num_items * (sizeof(key_type) + sizeof(value_type));
    const std::size_t max_summaries   = cub::DivideAndRoundUp(max_summary_mem, sizeof(summary_t));
    const std::size_t num_summaries   = std::min(std::min(max_summaries, num_items), static_cast<std::size_t>(max_key));

    TIME(c2h::cpu_timer timer);

    auto ref_key_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(std::size_t{0}),
      detail::key_value_sort_ref_key_transform<key_type>(num_items, num_summaries, is_descending));

    REQUIRE(thrust::equal(c2h::device_policy, keys.cbegin(), keys.cend(), ref_key_begin));

    TIME(timer.print_elapsed_seconds_and_reset("Validate keys"));

    REQUIRE((values == this->values_ref) == true);

    TIME(timer.print_elapsed_seconds_and_reset("Validate values"));
  }
};
