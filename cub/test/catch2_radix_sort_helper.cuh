// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

// #define CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <cuda/buffer>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/type_traits>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

#include <c2h/catch2_test_helper.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/generators.h>
#include <c2h/utility.h>
#include <c2h/vector.h>

// Index types used for OffsetsT testing
using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::uint64_t>;
using all_offset_types =
  c2h::type_list<cuda::std::int64_t, cuda::std::uint64_t, cuda::std::int32_t, cuda::std::uint32_t>;

// The launchers defined in catch2_test_launch_helper.h do not support
// passing objects by reference since the device-launch tests cannot
// pass references to a __global__ function. The DoubleBuffer object
// must be passed by reference to the radix sort APIs so that the selector
// can be updated appropriately for the caller. This wrapper allows the
// selector to be updated in a way that's compatible with the launch helpers.
// Call initialize() before using to allocate temporary memory, and finalize()
// when finished to release.
struct double_buffer_sort_t
{
private:
  bool m_is_descending;
  int* m_selector{nullptr};

public:
  explicit double_buffer_sort_t(bool is_descending)
      : m_is_descending(is_descending)
  {}

  void initialize()
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&m_selector, sizeof(int)));
  }

  void finalize()
  {
    REQUIRE(cudaSuccess == cudaFreeHost(m_selector));
    m_selector = nullptr;
  }

  int selector() const
  {
    return *m_selector;
  }

  template <class KeyT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t
  operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, cub::DoubleBuffer<KeyT> keys, As... as)
  {
    const cudaError_t status =
      m_is_descending ? cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, keys, as...)
                      : cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, as...);

    *m_selector = keys.selector;
    return status;
  }

  template <class KeyT, class ValueT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    cub::DoubleBuffer<KeyT> keys,
    cub::DoubleBuffer<ValueT> values,
    As... as)
  {
    const cudaError_t status =
      m_is_descending
        ? cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, keys, values, as...)
        : cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values, as...);

    *m_selector = keys.selector;
    return status;
  }
};

struct double_buffer_segmented_sort_t
{
private:
  bool m_is_descending;
  int* m_selector{nullptr};

public:
  explicit double_buffer_segmented_sort_t(bool is_descending)
      : m_is_descending(is_descending)
  {}

  void initialize()
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&m_selector, sizeof(int)));
  }

  void finalize()
  {
    REQUIRE(cudaSuccess == cudaFreeHost(m_selector));
    m_selector = nullptr;
  }

  int selector() const
  {
    return *m_selector;
  }

  template <class KeyT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t
  operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, cub::DoubleBuffer<KeyT> keys, As... as)
  {
    const cudaError_t status =
      m_is_descending
        ? cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, keys, as...)
        : cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, as...);

    *m_selector = keys.selector;
    return status;
  }

  template <class KeyT, class ValueT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    cub::DoubleBuffer<KeyT> keys,
    cub::DoubleBuffer<ValueT> values,
    As... as)
  {
    const cudaError_t status =
      m_is_descending
        ? cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, keys, values, as...)
        : cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values, as...);

    *m_selector = keys.selector;
    return status;
  }
};

// Helpers to assist with specifying default args to DeviceRadixSort API:
template <typename T>
constexpr int begin_bit()
{
  return 0;
}

template <typename T>
constexpr int end_bit()
{
  return static_cast<int>(sizeof(T) * CHAR_BIT);
}

template <class Keys, class StripedKeys>
void stripe_radix_keys(const Keys& h_keys, StripedKeys& h_striped_keys, int begin_bit, int end_bit)
{
  using KeyT          = typename Keys::value_type;
  using traits_t      = cub::Traits<KeyT>;
  using bit_ordered_t = typename traits_t::UnsignedBits;

  const int num_bits = end_bit - begin_bit;

  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    bit_ordered_t key = ::cuda::std::bit_cast<bit_ordered_t>(h_keys[i]);

    if constexpr (::cuda::is_floating_point_v<KeyT>)
    {
      const bit_ordered_t negative_zero = bit_ordered_t(1) << bit_ordered_t((sizeof(bit_ordered_t) * 8) - 1);

      if (key == negative_zero)
      {
        key = 0;
      }
    }

    key = traits_t::TwiddleIn(key);

    if (num_bits == 0)
    {
      key = 0;
    }
    else if (begin_bit > 0 || end_bit < static_cast<int>(sizeof(KeyT) * CHAR_BIT))
    {
      key &= ((bit_ordered_t{1} << num_bits) - 1) << begin_bit;
    }

    // striped keys are used to compare bit ordered representation of keys,
    // so we do not twiddle-out the key here:
    // key = traits_t::TwiddleOut(key);

    h_striped_keys[i] = key;
  }
}

template <class T>
struct indirect_binary_comparator_t
{
  const T* h_ptr{};
  bool is_descending{};

  indirect_binary_comparator_t(const T* h_ptr, bool is_descending)
      : h_ptr(h_ptr)
      , is_descending(is_descending)
  {}

  bool operator()(std::size_t a, std::size_t b)
  {
    if (is_descending)
    {
      return h_ptr[a] > h_ptr[b];
    }

    return h_ptr[a] < h_ptr[b];
  }
};

template <class StripedKeys, class Permutation, class SegBeginIterT, class SegEndIterT>
void sort_radix_permutation(
  const StripedKeys& h_striped_keys,
  Permutation& h_permutation,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it)
{
  if (h_permutation.empty())
  {
    return;
  }

  auto* permutation = thrust::raw_pointer_cast(h_permutation.data());
  std::iota(permutation, permutation + h_permutation.size(), std::size_t{0});

  using bits_t = typename StripedKeys::value_type;
  const indirect_binary_comparator_t<bits_t> comp{thrust::raw_pointer_cast(h_striped_keys.data()), is_descending};

  for (std::size_t segment = 0; segment < num_segments; ++segment)
  {
    std::stable_sort(permutation + h_seg_begin_it[segment], permutation + h_seg_end_it[segment], comp);
  }
}

template <class KeyT, class SegBeginIterT, class SegEndIterT>
c2h::host_vector<std::size_t> get_permutation(
  const c2h::host_vector<KeyT>& h_keys,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it,
  int begin_bit,
  int end_bit)
{
  using bits_t = typename cub::Traits<KeyT>::UnsignedBits;
  c2h::host_vector<bits_t> h_striped_keys(h_keys.size());
  c2h::host_vector<std::size_t> h_permutation(h_keys.size());
  stripe_radix_keys(h_keys, h_striped_keys, begin_bit, end_bit);
  sort_radix_permutation(h_striped_keys, h_permutation, is_descending, num_segments, h_seg_begin_it, h_seg_end_it);
  return h_permutation;
}

template <class KeyT>
cuda::host_buffer<std::size_t> get_permutation(
  cuda::stream_ref stream, const cuda::host_buffer<KeyT>& h_keys, bool is_descending, int begin_bit, int end_bit)
{
  using bits_t = typename cub::Traits<KeyT>::UnsignedBits;
  const std::array<std::size_t, 2> segments{0, h_keys.size()};
  auto h_striped_keys = c2h::make_host_buffer<bits_t>(stream, h_keys.size(), cuda::no_init);
  auto h_permutation  = c2h::make_host_buffer<std::size_t>(stream, h_keys.size(), cuda::no_init);
  stripe_radix_keys(h_keys, h_striped_keys, begin_bit, end_bit);
  sort_radix_permutation(h_striped_keys, h_permutation, is_descending, 1, segments.data(), segments.data() + 1);
  return h_permutation;
}

template <class KeyT>
cuda::host_buffer<KeyT> radix_sort_reference(
  cuda::stream_ref stream,
  const cuda::host_buffer<KeyT>& h_keys,
  bool is_descending,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const auto permutation = get_permutation(stream, h_keys, is_descending, begin_bit, end_bit);
  auto result            = c2h::make_host_buffer<KeyT>(stream, h_keys.size(), cuda::no_init);
  for (std::size_t i = 0; i < h_keys.size(); ++i)
  {
    result[i] = h_keys[permutation[i]];
  }
  return result;
}

template <class KeyT, class ValueT>
std::pair<cuda::host_buffer<KeyT>, cuda::host_buffer<ValueT>> radix_sort_reference(
  cuda::stream_ref stream,
  const cuda::host_buffer<KeyT>& h_keys,
  const cuda::host_buffer<ValueT>& h_values,
  bool is_descending)
{
  const auto permutation = get_permutation(stream, h_keys, is_descending, begin_bit<KeyT>(), end_bit<KeyT>());
  auto keys              = c2h::make_host_buffer<KeyT>(stream, h_keys.size(), cuda::no_init);
  auto values            = c2h::make_host_buffer<ValueT>(stream, h_values.size(), cuda::no_init);
  for (std::size_t i = 0; i < h_keys.size(); ++i)
  {
    keys[i]   = h_keys[permutation[i]];
    values[i] = h_values[permutation[i]];
  }
  return {cuda::std::move(keys), cuda::std::move(values)};
}

template <class KeyT>
c2h::host_vector<KeyT> radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<KeyT> h_keys(d_keys);
  const std::array<std::size_t, 2> segments{0, d_keys.size()};
  const c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, 1, segments.cbegin(), segments.cbegin() + 1, begin_bit, end_bit);
  c2h::host_vector<KeyT> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());

  return result;
}

template <class KeyT, class ValueT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> result;
  result.first.resize(d_keys.size());
  result.second.resize(d_keys.size());

  const std::array<std::size_t, 2> segments{0, d_keys.size()};

  const c2h::host_vector<KeyT> h_keys(d_keys);
  const c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, 1, segments.cbegin(), segments.cbegin() + 1, begin_bit, end_bit);

  const c2h::host_vector<ValueT> h_values(d_values);
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result.first.begin(), result.second.begin()));

  return result;
}

template <class KeyT, class SegBeginIterT, class SegEndIterT>
c2h::host_vector<KeyT> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<KeyT> h_keys(d_keys);
  const c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
  c2h::host_vector<KeyT> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());

  return result;
}

template <class KeyT, class ValueT, class SegBeginIterT, class SegEndIterT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> result;
  result.first.resize(d_keys.size());
  result.second.resize(d_keys.size());

  const c2h::host_vector<KeyT> h_keys(d_keys);
  const c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);

  const c2h::host_vector<ValueT> h_values(d_values);
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result.first.begin(), result.second.begin()));

  return result;
}

template <class KeyT, class OffsetT>
c2h::host_vector<KeyT> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const std::size_t num_segments = h_offsets.size() - 1;
  auto h_seg_begin_it            = h_offsets.cbegin();
  auto h_seg_end_it              = h_offsets.cbegin() + 1;
  return segmented_radix_sort_reference(
    d_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <class KeyT, class OffsetT>
c2h::host_vector<KeyT> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets_begin,
  const c2h::device_vector<OffsetT>& d_offsets_end,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets_begin(d_offsets_begin);
  const c2h::host_vector<OffsetT> h_offsets_end(d_offsets_end);
  const std::size_t num_segments = h_offsets_begin.size();
  auto h_seg_begin_it            = h_offsets_begin.cbegin();
  auto h_seg_end_it              = h_offsets_end.cbegin();
  return segmented_radix_sort_reference(
    d_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <class KeyT, class ValueT, class OffsetT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const std::size_t num_segments = h_offsets.size() - 1;
  auto h_seg_begin_it            = h_offsets.cbegin();
  auto h_seg_end_it              = h_offsets.cbegin() + 1;
  return segmented_radix_sort_reference(
    d_keys, d_values, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <class KeyT, class ValueT, class OffsetT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets_begin,
  const c2h::device_vector<OffsetT>& d_offsets_end,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets_begin(d_offsets_begin);
  const c2h::host_vector<OffsetT> h_offsets_end(d_offsets_end);
  const std::size_t num_segments = h_offsets_begin.size();
  auto h_seg_begin_it            = h_offsets_begin.cbegin();
  auto h_seg_end_it              = h_offsets_end.cbegin();
  return segmented_radix_sort_reference(
    d_keys, d_values, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <typename OffsetT>
struct radix_offset_scan_op_t
{
  OffsetT num_items;

  __host__ __device__ OffsetT operator()(OffsetT a, OffsetT b) const
  {
    const OffsetT sum = a + b;
    return ::cuda::std::min(sum, num_items);
  }
};

template <class OffsetT>
void generate_segment_offsets(c2h::seed_t seed, c2h::device_vector<OffsetT>& offsets, std::size_t num_items)
{
  const std::size_t num_segments        = offsets.size() - 1;
  const OffsetT expected_segment_length = static_cast<OffsetT>(::cuda::ceil_div(num_items, num_segments));
  const OffsetT max_segment_length      = (expected_segment_length * 2) + 1;
  c2h::gen(seed, offsets, OffsetT{0}, max_segment_length);
  thrust::exclusive_scan(
    c2h::device_policy,
    offsets.begin(),
    offsets.end(),
    offsets.begin(),
    OffsetT{0},
    radix_offset_scan_op_t<OffsetT>{static_cast<OffsetT>(num_items)});
}
