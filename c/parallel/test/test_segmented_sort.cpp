//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional> // std::optional
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/segmented_sort.h>
#include <cccl/c/types.h>

using key_types = c2h::type_list<uint8_t, int16_t, uint32_t, double>;
using item_t    = float;

using BuildResultT = cccl_device_segmented_sort_build_result_t;

struct segmented_sort_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_segmented_sort_cleanup(build_data);
  }
};

using segmented_sort_deleter       = BuildResultDeleter<BuildResultT, segmented_sort_cleanup>;
using segmented_sort_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, segmented_sort_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<segmented_sort_build_cache_t, Tag>::get_or_create().get_value();
}

struct segmented_sort_build
{
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_iterator_t keys_in,
    cccl_iterator_t keys_out,
    cccl_iterator_t values_in,
    cccl_iterator_t values_out,
    int64_t /*num_items*/,
    int64_t /*num_segments*/,
    cccl_iterator_t start_offsets,
    cccl_iterator_t end_offsets,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_segmented_sort_build(
      build_ptr,
      keys_in,
      keys_out,
      values_in,
      values_out,
      start_offsets,
      end_offsets,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path);
  }
};

struct segmented_sort_run
{
  template <typename... Ts>
  CUresult operator()(Ts... args) const noexcept
  {
    return cccl_device_segmented_sort(args...);
  }
};

template <typename BuildCache = segmented_sort_build_cache_t, typename KeyT = std::string>
void segmented_sort(
  cccl_iterator_t keys_in,
  cccl_iterator_t keys_out,
  cccl_iterator_t values_in,
  cccl_iterator_t values_out,
  int64_t num_items,
  int64_t num_segments,
  cccl_iterator_t start_offsets,
  cccl_iterator_t end_offsets,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, segmented_sort_build, segmented_sort_cleanup, segmented_sort_run, BuildCache, KeyT>(
    cache, lookup_key, keys_in, keys_out, values_in, values_out, num_items, num_segments, start_offsets, end_offsets);
}

// ==============
//   Test section
// ==============

static std::tuple<std::string, std::string, std::string> make_step_counting_iterator_sources(
  std::string_view index_ty_name,
  std::string_view state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name)
{
  static constexpr std::string_view it_state_src_tmpl = R"XXX(
struct {0} {{
  {1} linear_id;
  {1} row_size;
}};
)XXX";

  const std::string it_state_def_src = std::format(it_state_src_tmpl, state_name, index_ty_name);

  static constexpr std::string_view it_def_src_tmpl = R"XXX(
extern "C" __device__ void {0}({1}* state, {2} offset)
{{
  state->linear_id += offset;
}}
)XXX";

  const std::string it_advance_fn_def_src =
    std::format(it_def_src_tmpl, /*0*/ advance_fn_name, state_name, index_ty_name);

  static constexpr std::string_view it_deref_src_tmpl = R"XXX(
extern "C" __device__ {2} {0}({1}* state)
{{
  return (state->linear_id) * (state->row_size);
}}
)XXX";

  const std::string it_deref_fn_def_src =
    std::format(it_deref_src_tmpl, dereference_fn_name, state_name, index_ty_name);

  return std::make_tuple(it_state_def_src, it_advance_fn_def_src, it_deref_fn_def_src);
}

struct SegmentedSort_KeysOnly_Fixture_Tag;
C2H_TEST("segmented_sort can sort keys-only with integral types", "[segmented_sort][keys_only]", key_types)
{
  using KeyT = c2h::get<0, TestType>;
  // generate choices for n_segments: 0, 13 and 2 random samples from [50, 200)
  const std::size_t n_segments = GENERATE(0, 13, take(2, random(50, 200)));
  // generate choices for segment size: 1, 20 and random samples
  const std::size_t segment_size = GENERATE(1, 20, take(2, random(10, 100)));

  const std::size_t n_elems = n_segments * segment_size;

  std::vector<int> host_keys_int = generate<int>(n_elems);
  std::vector<KeyT> host_keys(host_keys_int.begin(), host_keys_int.end());
  std::vector<KeyT> host_keys_out(n_elems);

  REQUIRE(host_keys.size() == n_elems);
  REQUIRE(host_keys_out.size() == n_elems);

  pointer_t<KeyT> keys_in_ptr(host_keys); // copy from host to device
  pointer_t<KeyT> keys_out_ptr(host_keys_out); // copy from host to device

  pointer_t<item_t> values_in;
  pointer_t<item_t> values_out;

  using SizeT                                     = unsigned long long;
  static constexpr std::string_view index_ty_name = "unsigned long long";

  struct segment_offset_iterator_state_t
  {
    SizeT linear_id;
    SizeT segment_size;
  };

  static constexpr std::string_view offset_iterator_state_name = "segment_offset_iterator_state_t";
  static constexpr std::string_view advance_offset_method_name = "advance_offset_it";
  static constexpr std::string_view deref_offset_method_name   = "dereference_offset_it";

  const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
    make_step_counting_iterator_sources(
      index_ty_name, offset_iterator_state_name, advance_offset_method_name, deref_offset_method_name);

  iterator_t<SizeT, segment_offset_iterator_state_t> start_offset_it =
    make_iterator<SizeT, segment_offset_iterator_state_t>(
      {offset_iterator_state_name, offset_iterator_state_src},
      {advance_offset_method_name, offset_iterator_advance_src},
      {deref_offset_method_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id    = 0;
  start_offset_it.state.segment_size = segment_size;

  // Create end offset iterator (points to one past start)
  iterator_t<SizeT, segment_offset_iterator_state_t> end_offset_it =
    make_iterator<SizeT, segment_offset_iterator_state_t>(
      {offset_iterator_state_name, ""}, {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

  end_offset_it.state.linear_id    = 1;
  end_offset_it.state.segment_size = segment_size;

  auto& build_cache    = get_cache<SegmentedSort_KeysOnly_Fixture_Tag>();
  const auto& test_key = make_key<KeyT>();

  segmented_sort(
    keys_in_ptr,
    keys_out_ptr,
    values_in,
    values_out,
    n_elems,
    n_segments,
    start_offset_it,
    end_offset_it,
    build_cache,
    test_key);

  // Create expected result by sorting each segment
  std::vector<KeyT> expected_keys = host_keys;
  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = i * segment_size;
    std::size_t segment_end   = segment_start + segment_size;
    std::sort(expected_keys.begin() + segment_start, expected_keys.begin() + segment_end);
  }

  REQUIRE(expected_keys == std::vector<KeyT>(keys_out_ptr));
}

struct SegmentedSort_KeyValuePairs_Fixture_Tag;
C2H_TEST_LIST("segmented_sort can sort key-value pairs with integral types",
              "[segmented_sort][key_value]",
              std::int32_t,
              std::int64_t,
              std::uint32_t,
              std::uint64_t)
{
  // generate choices for n_segments: 0, 10 and random samples
  const std::size_t n_segments = GENERATE(0, 10, take(2, random(30, 100)));
  // generate choices for segment size
  const std::size_t segment_size = GENERATE(1, 15, take(2, random(5, 50)));

  const std::size_t n_elems = n_segments * segment_size;

  std::vector<TestType> host_keys   = generate<TestType>(n_elems);
  std::vector<TestType> host_values = generate<TestType>(n_elems);
  std::vector<TestType> host_keys_out(n_elems);
  std::vector<TestType> host_values_out(n_elems);

  REQUIRE(host_keys.size() == n_elems);
  REQUIRE(host_values.size() == n_elems);

  pointer_t<TestType> keys_in_ptr(host_keys);
  pointer_t<TestType> keys_out_ptr(host_keys_out);
  pointer_t<TestType> values_in_ptr(host_values);
  pointer_t<TestType> values_out_ptr(host_values_out);

  using SizeT                                     = unsigned long long;
  static constexpr std::string_view index_ty_name = "unsigned long long";

  struct segment_offset_iterator_state_t
  {
    SizeT linear_id;
    SizeT segment_size;
  };

  static constexpr std::string_view offset_iterator_state_name = "segment_offset_iterator_state_t";
  static constexpr std::string_view advance_offset_method_name = "advance_offset_it";
  static constexpr std::string_view deref_offset_method_name   = "dereference_offset_it";

  const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
    make_step_counting_iterator_sources(
      index_ty_name, offset_iterator_state_name, advance_offset_method_name, deref_offset_method_name);

  iterator_t<SizeT, segment_offset_iterator_state_t> start_offset_it =
    make_iterator<SizeT, segment_offset_iterator_state_t>(
      {offset_iterator_state_name, offset_iterator_state_src},
      {advance_offset_method_name, offset_iterator_advance_src},
      {deref_offset_method_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id    = 0;
  start_offset_it.state.segment_size = segment_size;

  iterator_t<SizeT, segment_offset_iterator_state_t> end_offset_it =
    make_iterator<SizeT, segment_offset_iterator_state_t>(
      {offset_iterator_state_name, ""}, {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

  end_offset_it.state.linear_id    = 1;
  end_offset_it.state.segment_size = segment_size;

  auto& build_cache    = get_cache<SegmentedSort_KeyValuePairs_Fixture_Tag>();
  const auto& test_key = make_key<TestType>();

  segmented_sort(
    keys_in_ptr,
    keys_out_ptr,
    values_in_ptr,
    values_out_ptr,
    n_elems,
    n_segments,
    start_offset_it,
    end_offset_it,
    build_cache,
    test_key);

  // Create expected result by sorting each segment with key-value pairs
  std::vector<std::pair<TestType, TestType>> key_value_pairs;
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    key_value_pairs.emplace_back(host_keys[i], host_values[i]);
  }

  std::vector<TestType> expected_keys(n_elems);
  std::vector<TestType> expected_values(n_elems);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = i * segment_size;
    std::size_t segment_end   = segment_start + segment_size;

    // Sort this segment by key
    std::sort(
      key_value_pairs.begin() + segment_start, key_value_pairs.begin() + segment_end, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

    // Extract sorted keys and values
    for (std::size_t j = segment_start; j < segment_end; ++j)
    {
      expected_keys[j]   = key_value_pairs[j].first;
      expected_values[j] = key_value_pairs[j].second;
    }
  }

  REQUIRE(expected_keys == std::vector<TestType>(keys_out_ptr));
  REQUIRE(expected_values == std::vector<TestType>(values_out_ptr));
}

struct custom_pair
{
  int key;
  size_t value;

  bool operator==(const custom_pair& other) const
  {
    return key == other.key && value == other.value;
  }

  bool operator<(const custom_pair& other) const
  {
    return key < other.key;
  }
};

struct SegmentedSort_CustomTypes_Fixture_Tag;
C2H_TEST("SegmentedSort works with custom types as keys", "[segmented_sort][custom_types]")
{
  using KeyT   = custom_pair;
  using ValueT = float;

  const std::size_t n_segments   = 25;
  const std::size_t segment_size = 20;
  const std::size_t n_elems      = n_segments * segment_size;

  // Generate custom key data
  std::vector<KeyT> host_keys(n_elems);
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    host_keys[i] = custom_pair{static_cast<int>(i % 1000), static_cast<std::size_t>(i % 100)};
  }

  // Generate float values by first generating ints and then transforming
  std::vector<int> host_values_int = generate<int>(n_elems);
  std::vector<ValueT> host_values(n_elems);
  std::transform(host_values_int.begin(), host_values_int.end(), host_values.begin(), [](int x) {
    return static_cast<ValueT>(x);
  });
  std::vector<KeyT> host_keys_out(n_elems);
  std::vector<ValueT> host_values_out(n_elems);

  pointer_t<KeyT> keys_in_ptr(host_keys);
  pointer_t<KeyT> keys_out_ptr(host_keys_out);
  pointer_t<ValueT> values_in_ptr(host_values);
  pointer_t<ValueT> values_out_ptr(host_values_out);

  using SizeT = ::cuda::std::size_t;
  std::vector<SizeT> segments(n_segments + 1);
  for (std::size_t i = 0; i <= n_segments; ++i)
  {
    segments[i] = i * segment_size;
  }

  pointer_t<SizeT> offset_ptr(segments);

  auto start_offset_it = static_cast<cccl_iterator_t>(offset_ptr);
  auto end_offset_it   = start_offset_it;
  end_offset_it.state  = offset_ptr.ptr + 1;

  auto& build_cache    = get_cache<SegmentedSort_CustomTypes_Fixture_Tag>();
  const auto& test_key = make_key<KeyT, ValueT>();

  segmented_sort(
    keys_in_ptr,
    keys_out_ptr,
    values_in_ptr,
    values_out_ptr,
    n_elems,
    n_segments,
    start_offset_it,
    end_offset_it,
    build_cache,
    test_key);

  // Create expected result
  std::vector<std::pair<KeyT, ValueT>> key_value_pairs;
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    key_value_pairs.emplace_back(host_keys[i], host_values[i]);
  }

  std::vector<KeyT> expected_keys(n_elems);
  std::vector<ValueT> expected_values(n_elems);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = segments[i];
    std::size_t segment_end   = segments[i + 1];

    // Sort this segment by key
    std::sort(
      key_value_pairs.begin() + segment_start, key_value_pairs.begin() + segment_end, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

    // Extract sorted keys and values
    for (std::size_t j = segment_start; j < segment_end; ++j)
    {
      expected_keys[j]   = key_value_pairs[j].first;
      expected_values[j] = key_value_pairs[j].second;
    }
  }

  auto result_keys   = std::vector<KeyT>(keys_out_ptr);
  auto result_values = std::vector<ValueT>(values_out_ptr);

  REQUIRE(expected_keys == result_keys);
  REQUIRE(expected_values == result_values);
}

using SizeT = unsigned long long;

struct variable_segment_offset_iterator_state_t
{
  SizeT linear_id;
  const SizeT* offsets;
};

static std::tuple<std::string, std::string, std::string> make_variable_segment_iterator_sources()
{
  static constexpr std::string_view it_state_src = R"XXX(
struct variable_segment_offset_iterator_state_t {
  unsigned long long linear_id;
  const unsigned long long* offsets;
};
)XXX";

  static constexpr std::string_view it_advance_src = R"XXX(
extern "C" __device__ void advance_variable_offset_it(variable_segment_offset_iterator_state_t* state, unsigned long long offset)
{
  state->linear_id += offset;
}
)XXX";

  static constexpr std::string_view it_deref_src = R"XXX(
extern "C" __device__ unsigned long long dereference_variable_offset_it(variable_segment_offset_iterator_state_t* state)
{
  return state->offsets[state->linear_id];
}
)XXX";

  return std::make_tuple(std::string(it_state_src), std::string(it_advance_src), std::string(it_deref_src));
}

struct SegmentedSort_VariableSegments_Fixture_Tag;
C2H_TEST("SegmentedSort works with variable segment sizes", "[segmented_sort][variable_segments]")
{
  using KeyT   = std::int32_t;
  using ValueT = float;

  const std::size_t n_segments = 20;

  // Create variable segment sizes
  std::vector<std::size_t> segment_sizes = {1, 5, 10, 20, 30, 15, 8, 3, 25, 12, 7, 18, 22, 4, 35, 9, 14, 6, 28, 11};
  REQUIRE(segment_sizes.size() == n_segments);

  std::size_t n_elems = std::accumulate(segment_sizes.begin(), segment_sizes.end(), 0ULL);

  std::vector<KeyT> host_keys = generate<KeyT>(n_elems);
  // Generate float values by first generating ints and then transforming
  std::vector<int> host_values_int = generate<int>(n_elems);
  std::vector<ValueT> host_values(n_elems);
  std::transform(host_values_int.begin(), host_values_int.end(), host_values.begin(), [](int x) {
    return static_cast<ValueT>(x);
  });
  std::vector<KeyT> host_keys_out(n_elems);
  std::vector<ValueT> host_values_out(n_elems);

  pointer_t<KeyT> keys_in_ptr(host_keys);
  pointer_t<KeyT> keys_out_ptr(host_keys_out);
  pointer_t<ValueT> values_in_ptr(host_values);
  pointer_t<ValueT> values_out_ptr(host_values_out);

  // Create segment offset arrays
  std::vector<SizeT> start_offsets(n_segments);
  std::vector<SizeT> end_offsets(n_segments);

  SizeT current_offset = 0;
  for (std::size_t i = 0; i < n_segments; ++i)
  {
    start_offsets[i] = current_offset;
    current_offset += segment_sizes[i];
    end_offsets[i] = current_offset;
  }

  pointer_t<SizeT> start_offsets_ptr(start_offsets);
  pointer_t<SizeT> end_offsets_ptr(end_offsets);

  const auto& [offset_state_src, offset_advance_src, offset_deref_src] = make_variable_segment_iterator_sources();

  iterator_t<SizeT, variable_segment_offset_iterator_state_t> start_offset_it =
    make_iterator<SizeT, variable_segment_offset_iterator_state_t>(
      {"variable_segment_offset_iterator_state_t", offset_state_src},
      {"advance_variable_offset_it", offset_advance_src},
      {"dereference_variable_offset_it", offset_deref_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.offsets   = start_offsets_ptr.ptr;

  iterator_t<SizeT, variable_segment_offset_iterator_state_t> end_offset_it =
    make_iterator<SizeT, variable_segment_offset_iterator_state_t>(
      {"variable_segment_offset_iterator_state_t", ""},
      {"advance_variable_offset_it", ""},
      {"dereference_variable_offset_it", ""});

  end_offset_it.state.linear_id = 0;
  end_offset_it.state.offsets   = end_offsets_ptr.ptr;

  auto& build_cache    = get_cache<SegmentedSort_VariableSegments_Fixture_Tag>();
  const auto& test_key = make_key<KeyT, ValueT>();

  segmented_sort(
    keys_in_ptr,
    keys_out_ptr,
    values_in_ptr,
    values_out_ptr,
    n_elems,
    n_segments,
    start_offset_it,
    end_offset_it,
    build_cache,
    test_key);

  // Create expected result
  std::vector<std::pair<KeyT, ValueT>> key_value_pairs;
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    key_value_pairs.emplace_back(host_keys[i], host_values[i]);
  }

  std::vector<KeyT> expected_keys(n_elems);
  std::vector<ValueT> expected_values(n_elems);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = start_offsets[i];
    std::size_t segment_end   = end_offsets[i];

    // Sort this segment by key
    std::sort(
      key_value_pairs.begin() + segment_start, key_value_pairs.begin() + segment_end, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

    // Extract sorted keys and values
    for (std::size_t j = segment_start; j < segment_end; ++j)
    {
      expected_keys[j]   = key_value_pairs[j].first;
      expected_values[j] = key_value_pairs[j].second;
    }
  }

  auto result_keys   = std::vector<KeyT>(keys_out_ptr);
  auto result_values = std::vector<ValueT>(values_out_ptr);

  REQUIRE(expected_keys == result_keys);
  REQUIRE(expected_values == result_values);
}
