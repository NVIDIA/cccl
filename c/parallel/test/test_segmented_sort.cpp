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
#include <optional> // std::optional
#include <string>
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

using SizeT = ptrdiff_t;

struct segmented_sort_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_segmented_sort_cleanup(build_data);
  }
};

using segmented_sort_deleter       = BuildResultDeleter<BuildResultT, segmented_sort_cleanup>;
using segmented_sort_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, segmented_sort_deleter>>;

template <typename KeyTy, bool descending = false, bool overwrite_okay = false>
struct TestParameters
{
  using KeyT                             = KeyTy;
  static constexpr bool m_descending     = descending;
  static constexpr bool m_overwrite_okay = overwrite_okay;

  constexpr TestParameters() {}

  constexpr bool is_descending() const
  {
    return m_descending;
  }
  constexpr bool is_overwrite_okay() const
  {
    return m_overwrite_okay;
  }
};

using test_params_tuple =
  c2h::type_list<TestParameters<c2h::get<0, key_types>, false, false>,
                 TestParameters<c2h::get<1, key_types>, true, false>,
                 TestParameters<c2h::get<2, key_types>, false, true>,
                 TestParameters<c2h::get<3, key_types>, true, true>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<segmented_sort_build_cache_t, Tag>::get_or_create().get_value();
}

struct segmented_sort_build
{
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_sort_order_t sort_order,
    cccl_iterator_t keys_in,
    cccl_iterator_t /*keys_out*/,
    cccl_iterator_t values_in,
    cccl_iterator_t /*values_out*/,
    int64_t /*num_items*/,
    int64_t /*num_segments*/,
    cccl_iterator_t start_offsets,
    cccl_iterator_t end_offsets,
    bool /*is_overwrite_okay*/,
    int* /*selector*/,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_segmented_sort_build(
      build_ptr,
      sort_order,
      keys_in,
      values_in,
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
  template <typename... Rest>
  CUresult operator()(
    BuildResultT build,
    void* temp_storage,
    size_t* temp_storage_bytes,
    cccl_sort_order_t,
    cccl_iterator_t d_keys_in,
    cccl_iterator_t d_keys_out,
    cccl_iterator_t d_values_in,
    cccl_iterator_t d_values_out,
    int64_t num_items,
    int64_t num_segments,
    cccl_iterator_t start_offsets,
    cccl_iterator_t end_offsets,
    Rest... rest) const noexcept
  {
    return cccl_device_segmented_sort(
      build,
      temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      num_segments,
      start_offsets,
      end_offsets,
      rest...);
  }
};

template <typename BuildCache = segmented_sort_build_cache_t, typename KeyT = std::string>
void segmented_sort(
  cccl_sort_order_t sort_order,
  cccl_iterator_t keys_in,
  cccl_iterator_t keys_out,
  cccl_iterator_t values_in,
  cccl_iterator_t values_out,
  int64_t num_items,
  int64_t num_segments,
  cccl_iterator_t start_offsets,
  cccl_iterator_t end_offsets,
  bool is_overwrite_okay,
  int* selector,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, segmented_sort_build, segmented_sort_cleanup, segmented_sort_run, BuildCache, KeyT>(
    cache,
    lookup_key,
    sort_order,
    keys_in,
    keys_out,
    values_in,
    values_out,
    num_items,
    num_segments,
    start_offsets,
    end_offsets,
    is_overwrite_okay,
    selector);
}

// ==============
//   Test section
// ==============

struct SegmentedSort_KeysOnly_Fixture_Tag;
C2H_TEST("segmented_sort can sort keys-only", "[segmented_sort][keys_only]", test_params_tuple)
{
  using T     = c2h::get<0, TestType>;
  using key_t = typename T::KeyT;

  constexpr auto this_test_params  = T();
  constexpr bool is_descending     = this_test_params.is_descending();
  constexpr auto order             = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;
  constexpr bool is_overwrite_okay = this_test_params.is_overwrite_okay();

  const std::size_t n_segments   = GENERATE(0, 13, take(2, random(1 << 10, 1 << 12)));
  const std::size_t segment_size = GENERATE(1, 12, take(2, random(1 << 10, 1 << 12)));

  const std::size_t n_elems = n_segments * segment_size;

  std::vector<int> host_keys_int = generate<int>(n_elems);
  std::vector<key_t> host_keys(n_elems);
  std::transform(host_keys_int.begin(), host_keys_int.end(), host_keys.begin(), [](int val) {
    return static_cast<key_t>(val);
  });
  std::vector<key_t> host_keys_out(n_elems);

  REQUIRE(host_keys.size() == n_elems);
  REQUIRE(host_keys_out.size() == n_elems);

  pointer_t<key_t> keys_in_ptr(host_keys);
  pointer_t<key_t> keys_out_ptr(host_keys_out);

  pointer_t<item_t> values_in;
  pointer_t<item_t> values_out;

  // TODO: Using a step counting iterator does not work right now.
  // static constexpr std::string_view index_ty_name = "signed long long";

  // struct segment_offset_iterator_state_t
  // {
  //   SizeT linear_id;
  //   SizeT segment_size;
  // };

  // static constexpr std::string_view offset_iterator_state_name = "segment_offset_iterator_state_t";
  // static constexpr std::string_view advance_offset_method_name = "advance_offset_it";
  // static constexpr std::string_view deref_offset_method_name   = "dereference_offset_it";

  // const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
  //   make_step_counting_iterator_sources(
  //     index_ty_name, offset_iterator_state_name, advance_offset_method_name, deref_offset_method_name);

  // iterator_t<SizeT, segment_offset_iterator_state_t> start_offset_it =
  //   make_iterator<SizeT, segment_offset_iterator_state_t>(
  //     {offset_iterator_state_name, offset_iterator_state_src},
  //     {advance_offset_method_name, offset_iterator_advance_src},
  //     {deref_offset_method_name, offset_iterator_deref_src});

  // start_offset_it.state.linear_id    = 0;
  // start_offset_it.state.segment_size = segment_size;

  // // Create end offset iterator (points to one past start)
  // iterator_t<SizeT, segment_offset_iterator_state_t> end_offset_it =
  //   make_iterator<SizeT, segment_offset_iterator_state_t>(
  //     {offset_iterator_state_name, ""}, {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

  // end_offset_it.state.linear_id    = 1;
  // end_offset_it.state.segment_size = segment_size;

  // // Provide host-advance callbacks for offset iterators
  // auto start_offsets_cccl         = static_cast<cccl_iterator_t>(start_offset_it);
  // auto end_offsets_cccl           = static_cast<cccl_iterator_t>(end_offset_it);
  // start_offsets_cccl.host_advance = &host_advance_linear_id<segment_offset_iterator_state_t>;
  // end_offsets_cccl.host_advance   = &host_advance_linear_id<segment_offset_iterator_state_t>;

  std::vector<SizeT> start_offsets(n_segments);
  std::vector<SizeT> end_offsets(n_segments);
  for (std::size_t i = 0; i < n_segments; ++i)
  {
    start_offsets[i] = static_cast<SizeT>(i * segment_size);
    end_offsets[i]   = static_cast<SizeT>((i + 1) * segment_size);
  }

  pointer_t<SizeT> start_offsets_ptr(start_offsets);
  pointer_t<SizeT> end_offsets_ptr(end_offsets);

  auto& build_cache             = get_cache<SegmentedSort_KeysOnly_Fixture_Tag>();
  const std::string& key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<key_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay)});
  const auto& test_key = std::make_optional(key_string);

  int selector = -1;

  segmented_sort(
    order,
    keys_in_ptr,
    keys_out_ptr,
    values_in,
    values_out,
    n_elems,
    n_segments,
    // start_offsets_cccl,
    // end_offsets_cccl,
    start_offsets_ptr,
    end_offsets_ptr,
    is_overwrite_okay,
    &selector,
    build_cache,
    test_key);

  // Create expected result by sorting each segment
  std::vector<key_t> expected_keys = host_keys;
  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = i * segment_size;
    std::size_t segment_end   = segment_start + segment_size;
    if (is_descending)
    {
      std::sort(expected_keys.begin() + segment_start, expected_keys.begin() + segment_end, std::greater<key_t>());
    }
    else
    {
      std::sort(expected_keys.begin() + segment_start, expected_keys.begin() + segment_end);
    }
  }

  auto& output_keys = (is_overwrite_okay && selector == 0) ? keys_in_ptr : keys_out_ptr;
  REQUIRE(expected_keys == std::vector<key_t>(output_keys));
}

struct SegmentedSort_KeyValuePairs_Fixture_Tag;
C2H_TEST("segmented_sort can sort key-value pairs", "[segmented_sort][key_value]", test_params_tuple)
{
  using T     = c2h::get<0, TestType>;
  using key_t = typename T::KeyT;

  constexpr auto this_test_params  = T();
  constexpr bool is_descending     = this_test_params.is_descending();
  constexpr auto order             = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;
  constexpr bool is_overwrite_okay = this_test_params.is_overwrite_okay();

  const std::size_t n_segments   = GENERATE(0, 13, take(2, random(1 << 10, 1 << 12)));
  const std::size_t segment_size = GENERATE(1, 12, take(2, random(1 << 10, 1 << 12)));

  const std::size_t n_elems = n_segments * segment_size;

  std::vector<int> host_keys_int = generate<int>(n_elems);
  std::vector<key_t> host_keys(n_elems);
  std::transform(host_keys_int.begin(), host_keys_int.end(), host_keys.begin(), [](int val) {
    return static_cast<key_t>(val);
  });
  std::vector<int> host_values_int = generate<int>(n_elems);
  std::vector<item_t> host_values(n_elems);
  std::transform(host_values_int.begin(), host_values_int.end(), host_values.begin(), [](int val) {
    return static_cast<item_t>(val);
  });

  std::vector<key_t> host_keys_out(n_elems);
  std::vector<item_t> host_values_out(n_elems);

  REQUIRE(host_keys.size() == n_elems);
  REQUIRE(host_values.size() == n_elems);

  pointer_t<key_t> keys_in_ptr(host_keys);
  pointer_t<key_t> keys_out_ptr(host_keys_out);

  pointer_t<item_t> values_in_ptr(host_values);
  pointer_t<item_t> values_out_ptr(host_values_out);

  std::vector<SizeT> start_offsets(n_segments);
  std::vector<SizeT> end_offsets(n_segments);
  for (std::size_t i = 0; i < n_segments; ++i)
  {
    start_offsets[i] = static_cast<SizeT>(i * segment_size);
    end_offsets[i]   = static_cast<SizeT>((i + 1) * segment_size);
  }

  pointer_t<SizeT> start_offsets_ptr(start_offsets);
  pointer_t<SizeT> end_offsets_ptr(end_offsets);

  auto& build_cache             = get_cache<SegmentedSort_KeyValuePairs_Fixture_Tag>();
  const std::string& key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<key_t>(),
     KeyBuilder::type_as_key<item_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay),
     KeyBuilder::bool_as_key(n_elems == 0)}); // this results in the values pointer being null which results in a keys
                                              // only build
  const auto& test_key = std::make_optional(key_string);

  int selector = -1;

  segmented_sort(
    order,
    keys_in_ptr,
    keys_out_ptr,
    values_in_ptr,
    values_out_ptr,
    n_elems,
    n_segments,
    // start_offsets_cccl,
    // end_offsets_cccl,
    start_offsets_ptr,
    end_offsets_ptr,
    is_overwrite_okay,
    &selector,
    build_cache,
    test_key);

  // Create expected result by sorting each segment with key-value pairs
  std::vector<std::pair<key_t, item_t>> key_value_pairs;
  key_value_pairs.reserve(n_elems);
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    key_value_pairs.emplace_back(host_keys[i], host_values[i]);
  }

  std::vector<key_t> expected_keys(n_elems);
  std::vector<item_t> expected_values(n_elems);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = i * segment_size;
    std::size_t segment_end   = segment_start + segment_size;

    if (is_descending)
    {
      std::stable_sort(key_value_pairs.begin() + segment_start,
                       key_value_pairs.begin() + segment_end,
                       [](const auto& a, const auto& b) {
                         return b.first < a.first;
                       });
    }
    else
    {
      std::stable_sort(key_value_pairs.begin() + segment_start,
                       key_value_pairs.begin() + segment_end,
                       [](const auto& a, const auto& b) {
                         return a.first < b.first;
                       });
    }

    // Extract sorted keys and values
    for (std::size_t j = segment_start; j < segment_end; ++j)
    {
      expected_keys[j]   = key_value_pairs[j].first;
      expected_values[j] = key_value_pairs[j].second;
    }
  }

  auto& output_keys = (is_overwrite_okay && selector == 0) ? keys_in_ptr : keys_out_ptr;
  auto& output_vals = (is_overwrite_okay && selector == 0) ? values_in_ptr : values_out_ptr;
  REQUIRE(expected_keys == std::vector<key_t>(output_keys));
  REQUIRE(expected_values == std::vector<item_t>(output_vals));
}

// These tests with custom types are currently failing TODO: add issue
#ifdef NEVER_DEFINED
struct custom_pair
{
  int key;
  size_t value;

  bool operator==(const custom_pair& other) const
  {
    return key == other.key && value == other.value;
  }
};

struct SegmentedSort_CustomTypes_Fixture_Tag;
C2H_TEST("SegmentedSort works with custom types as values", "[segmented_sort][custom_types]", test_params_tuple)
{
  using T       = c2h::get<0, TestType>;
  using key_t   = typename T::KeyT;
  using value_t = custom_pair;

  constexpr auto this_test_params  = T();
  constexpr bool is_descending     = this_test_params.is_descending();
  constexpr auto order             = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;
  constexpr bool is_overwrite_okay = this_test_params.is_overwrite_okay();

  const std::size_t n_segments   = GENERATE(0, 13, take(2, random(1 << 10, 1 << 12)));
  const std::size_t segment_size = GENERATE(1, 12, take(2, random(1 << 10, 1 << 12)));

  const std::size_t n_elems = n_segments * segment_size;

  // Generate primitive keys
  std::vector<int> host_keys_int = generate<int>(n_elems);
  std::vector<key_t> host_keys(n_elems);
  std::transform(host_keys_int.begin(), host_keys_int.end(), host_keys.begin(), [](int x) {
    return static_cast<key_t>(x);
  });

  // Generate custom values
  std::vector<value_t> host_values(n_elems);
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    host_values[i] = value_t{static_cast<int>(i % 1000), static_cast<std::size_t>(i % 100)};
  }
  std::vector<key_t> host_keys_out(n_elems);
  std::vector<value_t> host_values_out(n_elems);

  pointer_t<key_t> keys_in_ptr(host_keys);
  pointer_t<key_t> keys_out_ptr(host_keys_out);
  pointer_t<value_t> values_in_ptr(host_values);
  pointer_t<value_t> values_out_ptr(host_values_out);

  using SizeT = long;
  std::vector<SizeT> segments(n_segments + 1);
  for (std::size_t i = 0; i <= n_segments; ++i)
  {
    segments[i] = i * segment_size;
  }

  pointer_t<SizeT> offset_ptr(segments);

  auto start_offset_it = static_cast<cccl_iterator_t>(offset_ptr);
  auto end_offset_it   = start_offset_it;
  end_offset_it.state  = offset_ptr.ptr + 1;

  auto& build_cache             = get_cache<SegmentedSort_CustomTypes_Fixture_Tag>();
  const std::string& key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<key_t>(),
     KeyBuilder::type_as_key<value_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay),
     KeyBuilder::bool_as_key(n_elems == 0)});
  const auto& test_key = std::make_optional(key_string);

  int selector = -1;

  segmented_sort(
    order,
    keys_in_ptr,
    keys_out_ptr,
    values_in_ptr,
    values_out_ptr,
    n_elems,
    n_segments,
    start_offset_it,
    end_offset_it,
    is_overwrite_okay,
    &selector,
    build_cache,
    test_key);

  // Create expected result
  std::vector<std::pair<key_t, value_t>> key_value_pairs;
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    key_value_pairs.emplace_back(host_keys[i], host_values[i]);
  }

  std::vector<key_t> expected_keys(n_elems);
  std::vector<value_t> expected_values(n_elems);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = segments[i];
    std::size_t segment_end   = segments[i + 1];

    if (is_descending)
    {
      std::stable_sort(key_value_pairs.begin() + segment_start,
                       key_value_pairs.begin() + segment_end,
                       [](const auto& a, const auto& b) {
                         return b.first < a.first;
                       });
    }
    else
    {
      std::stable_sort(key_value_pairs.begin() + segment_start,
                       key_value_pairs.begin() + segment_end,
                       [](const auto& a, const auto& b) {
                         return a.first < b.first;
                       });
    }

    // Extract sorted keys and values
    for (std::size_t j = segment_start; j < segment_end; ++j)
    {
      expected_keys[j]   = key_value_pairs[j].first;
      expected_values[j] = key_value_pairs[j].second;
    }
  }

  auto& output_keys = (is_overwrite_okay && selector == 0) ? keys_in_ptr : keys_out_ptr;
  auto& output_vals = (is_overwrite_okay && selector == 0) ? values_in_ptr : values_out_ptr;

  REQUIRE(expected_keys == std::vector<key_t>(output_keys));
  REQUIRE(expected_values == std::vector<value_t>(output_vals));
}
#endif

struct SegmentedSort_VariableSegments_Fixture_Tag;
C2H_TEST("SegmentedSort works with variable segment sizes", "[segmented_sort][variable_segments]", test_params_tuple)
{
  using T     = c2h::get<0, TestType>;
  using key_t = typename T::KeyT;

  constexpr auto this_test_params  = T();
  constexpr bool is_descending     = this_test_params.is_descending();
  constexpr auto order             = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;
  constexpr bool is_overwrite_okay = this_test_params.is_overwrite_okay();

  const std::size_t n_segments = GENERATE(20, 600);

  // Create variable segment sizes
  const std::vector<std::size_t> base_pattern = {
    1, 5, 10, 20, 30, 50, 100, 3, 25, 600, 7, 18, 300, 4, 35, 9, 14, 700, 28, 11};
  std::vector<std::size_t> segment_sizes;
  segment_sizes.reserve(n_segments);
  while (segment_sizes.size() < n_segments)
  {
    const std::size_t remaining  = n_segments - segment_sizes.size();
    const std::size_t copy_count = std::min(remaining, base_pattern.size());
    segment_sizes.insert(segment_sizes.end(), base_pattern.begin(), base_pattern.begin() + copy_count);
  }
  REQUIRE(segment_sizes.size() == n_segments);

  std::size_t n_elems = std::accumulate(segment_sizes.begin(), segment_sizes.end(), 0ULL);

  std::vector<int> host_keys_int = generate<int>(n_elems);
  std::vector<key_t> host_keys(n_elems);
  std::transform(host_keys_int.begin(), host_keys_int.end(), host_keys.begin(), [](int val) {
    return static_cast<key_t>(val);
  });

  // Generate float values by first generating ints and then transforming
  std::vector<int> host_values_int = generate<int>(n_elems);
  std::vector<item_t> host_values(n_elems);
  std::transform(host_values_int.begin(), host_values_int.end(), host_values.begin(), [](int val) {
    return static_cast<item_t>(val);
  });
  std::vector<key_t> host_keys_out(n_elems);
  std::vector<item_t> host_values_out(n_elems);

  pointer_t<key_t> keys_in_ptr(host_keys);
  pointer_t<key_t> keys_out_ptr(host_keys_out);
  pointer_t<item_t> values_in_ptr(host_values);
  pointer_t<item_t> values_out_ptr(host_values_out);

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

  auto& build_cache             = get_cache<SegmentedSort_VariableSegments_Fixture_Tag>();
  const std::string& key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<key_t>(),
     KeyBuilder::type_as_key<item_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay)});
  const auto& test_key = std::make_optional(key_string);

  int selector = -1;

  segmented_sort(
    order,
    keys_in_ptr,
    keys_out_ptr,
    values_in_ptr,
    values_out_ptr,
    n_elems,
    n_segments,
    start_offsets_ptr,
    end_offsets_ptr,
    is_overwrite_okay,
    &selector,
    build_cache,
    test_key);

  // Create expected result
  std::vector<std::pair<key_t, item_t>> key_value_pairs;
  for (std::size_t i = 0; i < n_elems; ++i)
  {
    key_value_pairs.emplace_back(host_keys[i], host_values[i]);
  }

  std::vector<key_t> expected_keys(n_elems);
  std::vector<item_t> expected_values(n_elems);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    std::size_t segment_start = start_offsets[i];
    std::size_t segment_end   = end_offsets[i];

    if (is_descending)
    {
      std::stable_sort(key_value_pairs.begin() + segment_start,
                       key_value_pairs.begin() + segment_end,
                       [](const auto& a, const auto& b) {
                         return b.first < a.first;
                       });
    }
    else
    {
      std::stable_sort(key_value_pairs.begin() + segment_start,
                       key_value_pairs.begin() + segment_end,
                       [](const auto& a, const auto& b) {
                         return a.first < b.first;
                       });
    }

    // Extract sorted keys and values
    for (std::size_t j = segment_start; j < segment_end; ++j)
    {
      expected_keys[j]   = key_value_pairs[j].first;
      expected_values[j] = key_value_pairs[j].second;
    }
  }

  auto& output_keys = (is_overwrite_okay && selector == 0) ? keys_in_ptr : keys_out_ptr;
  auto& output_vals = (is_overwrite_okay && selector == 0) ? values_in_ptr : values_out_ptr;
  REQUIRE(expected_keys == std::vector<key_t>(output_keys));
  REQUIRE(expected_values == std::vector<item_t>(output_vals));
}
