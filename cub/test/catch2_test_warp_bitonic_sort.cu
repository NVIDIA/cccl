// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_sort.cuh>

#include <thrust/iterator/zip_iterator.h>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <algorithm>

#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

struct CustomLess
{
  template <typename T>
  __device__ __host__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }

  template <typename T>
  static __device__ __host__ T get_oob_default()
  {
    return cuda::std::numeric_limits<T>::max();
  };
};

inline constexpr int warp_threads = cub::detail::warp_threads;

/**
 * @brief Kernel to dispatch to the appropriate WarpBitonicSort member function, sorting keys-only.
 */
template <int ItemsPerThread, int TotalWarps, typename KeyT, typename ActionT>
__global__ void warp_bitonic_sort_kernel(KeyT* in, KeyT* out, int valid_items, ActionT action)
{
  using warp_bitonic_sort_t = cub::detail::WarpBitonicSort<ItemsPerThread, KeyT>;

  // Get linear thread and warp index
  const auto tid    = static_cast<int>(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
  const int warp_id = tid / warp_threads;
  const int lane    = tid % warp_threads;

  // Test case of partially finished CTA
  if (warp_id >= TotalWarps)
  {
    return;
  }

  // Thread-local storage
  KeyT thread_data[ItemsPerThread];

  // Instantiate warp-scope algorithm
  warp_bitonic_sort_t warp_sort;

  const int warp_offset = valid_items * warp_id;

  // Load data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * warp_threads + lane;
    if (idx < valid_items)
    {
      thread_data[i] = in[warp_offset + idx];
    }
  }

  // Run bitonic sort test
  action(warp_sort, thread_data, valid_items);

  // Store data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * warp_threads + lane;
    if (idx < valid_items)
    {
      out[warp_offset + idx] = thread_data[i];
    }
  }
}

/**
 * @brief Kernel to dispatch to the appropriate WarpBitonicSort member function, sorting key-value
 * pairs.
 */
template <int ItemsPerThread, int TotalWarps, typename KeyT, typename ValueT, typename ActionT>
__global__ void warp_bitonic_sort_kernel(
  KeyT* keys_in, KeyT* keys_out, ValueT* values_in, ValueT* values_out, int valid_items, ActionT action)
{
  using warp_bitonic_sort_t = cub::detail::WarpBitonicSort<ItemsPerThread, KeyT, ValueT>;

  // Get linear thread and warp index
  const auto tid    = static_cast<int>(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
  const int warp_id = tid / warp_threads;
  const int lane    = tid % warp_threads;

  // Test case of partially finished CTA
  if (warp_id >= TotalWarps)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];

  // Instantiate warp-scope algorithm
  warp_bitonic_sort_t warp_sort;

  const int warp_offset = valid_items * warp_id;

  // Load data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * warp_threads + lane;
    if (idx < valid_items)
    {
      keys[i]   = keys_in[warp_offset + idx];
      values[i] = values_in[warp_offset + idx];
    }
  }

  // Run bitonic sort test
  action(warp_sort, keys, values, valid_items);

  // Store data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * warp_threads + lane;
    if (idx < valid_items)
    {
      keys_out[warp_offset + idx]   = keys[i];
      values_out[warp_offset + idx] = values[i];
    }
  }
}

// -----------------------------------------------------------
// Dimensions being instantiated:
// {full,partial_oob,partial} x {keys, kv-pairs}
// -----------------------------------------------------------

/**
 * @brief Delegate wrapper for WarpBitonicSort::Sort on keys-only
 */
struct sort_keys_full_t
{
  template <int ItemsPerThread, typename KeyT, typename WarpSortT>
  __device__ void operator()(const WarpSortT& warp_sort, KeyT (&thread_data)[ItemsPerThread], int /*valid_items*/) const
  {
    warp_sort.Sort(thread_data, CustomLess{});
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicSort::Sort on keys-only with oob_default
 */
struct sort_keys_partial_oob_t
{
  template <int ItemsPerThread, typename KeyT, typename WarpSortT>
  __device__ void operator()(const WarpSortT& warp_sort, KeyT (&thread_data)[ItemsPerThread], int valid_items) const
  {
    warp_sort.Sort(thread_data, CustomLess{}, valid_items, CustomLess::get_oob_default<KeyT>());
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicSort::Sort on keys-only
 */
struct sort_keys_partial_t
{
  template <int ItemsPerThread, typename KeyT, typename WarpSortT>
  __device__ void operator()(const WarpSortT& warp_sort, KeyT (&thread_data)[ItemsPerThread], int valid_items) const
  {
    warp_sort.Sort(thread_data, CustomLess{}, valid_items);
  }
};

/**
 * @brief Delegate wrapper for WarpBitonicSort::Sort on key-value pairs
 */
struct sort_pairs_full_t
{
  template <int ItemsPerThread, typename KeyT, typename ValueT, typename WarpSortT>
  __device__ void operator()(
    const WarpSortT& warp_sort, KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int /*valid_items*/
  ) const
  {
    warp_sort.Sort(keys, values, CustomLess{});
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicSort::Sort on key-value pairs with oob_default
 */
struct sort_pairs_partial_oob_t
{
  template <int ItemsPerThread, typename KeyT, typename ValueT, typename WarpSortT>
  __device__ void operator()(
    const WarpSortT& warp_sort, KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int valid_items) const
  {
    warp_sort.Sort(keys, values, CustomLess{}, valid_items, CustomLess::get_oob_default<KeyT>());
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicSort::Sort on key-value pairs
 */
struct sort_pairs_partial_t
{
  template <int ItemsPerThread, typename KeyT, typename ValueT, typename WarpSortT>
  __device__ void operator()(
    const WarpSortT& warp_sort, KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int valid_items) const
  {
    warp_sort.Sort(keys, values, CustomLess{}, valid_items);
  }
};

/**
 * @brief Dispatch helper function for sorting keys
 */
template <int ItemsPerThread, int TotalWarps, typename KeyT, typename ActionT>
void warp_bitonic_sort(
  c2h::device_vector<KeyT>& in, c2h::device_vector<KeyT>& out, int valid_items, ActionT action, int num_block_dims)
{
  // only support num_block_dims is 1 or 2
  REQUIRE((num_block_dims == 1 || num_block_dims == 2));
  dim3 block_dims{warp_threads * TotalWarps};
  if (num_block_dims == 2)
  {
    // test the case when blockDim.x < warp_threads
    block_dims = dim3{warp_threads / 2, 2 * TotalWarps};
  }

  warp_bitonic_sort_kernel<ItemsPerThread, TotalWarps>
    <<<1, block_dims>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), valid_items, action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Dispatch helper function for sorting key-value pairs
 */
template <int ItemsPerThread, int TotalWarps, typename KeyT, typename ValueT, typename ActionT>
void warp_bitonic_sort(
  c2h::device_vector<KeyT>& keys_in,
  c2h::device_vector<KeyT>& keys_out,
  c2h::device_vector<ValueT>& values_in,
  c2h::device_vector<ValueT>& values_out,
  int valid_items,
  ActionT action,
  int num_block_dims)
{
  // only support num_block_dims is 1 or 2
  REQUIRE((num_block_dims == 1 || num_block_dims == 2));
  dim3 block_dims{warp_threads * TotalWarps};
  if (num_block_dims == 2)
  {
    // test the case when blockDim.x < warp_threads
    block_dims = dim3{warp_threads / 2, 2 * TotalWarps};
  }

  warp_bitonic_sort_kernel<ItemsPerThread, TotalWarps><<<1, block_dims>>>(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    valid_items,
    action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Performs a sort on per-warp segments of data
 */
template <typename RandomItT>
void compute_host_reference(RandomItT h_data, int valid_items, int total_warps)
{
  for (int i = 0; i < total_warps; i++)
  {
    std::sort(h_data, h_data + valid_items);
    h_data += valid_items;
  }
}

/**
 * @brief Sorts values within each run of equal keys. Useful for comparing unstable sort results
 */
template <typename KeyItT, typename ValueItT>
void sort_values_for_equal_keys(KeyItT keys_begin, KeyItT keys_end, ValueItT values_begin)
{
  auto i = keys_begin;
  while (i < keys_end)
  {
    auto j = i + 1;
    while (j < keys_end && *i == *j)
    {
      ++j;
    }
    std::sort(values_begin + (i - keys_begin), values_begin + (j - keys_begin));
    i = j;
  }
}

template <typename KeyItT, typename ValueItT>
void sort_values_for_equal_keys(KeyItT keys, ValueItT values, int valid_items, int total_warps)
{
  for (int i = 0; i < total_warps; i++)
  {
    sort_values_for_equal_keys(keys, keys + valid_items, values);
    keys += valid_items;
    values += valid_items;
  }
}

// List of key types to test
using custom_t  = c2h::custom_type_t<c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;
using key_types = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t, custom_t>;

// List of value types
using value_types = c2h::type_list<std::int32_t, custom_t>;

// Number of items per thread to test
using items_per_thread_list = c2h::enum_type_list<int, 1, 4, 7>;

// number of block dimensions to launch
using num_block_dims_list = c2h::enum_type_list<int, 1, 2>;

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int num_block_dims   = c2h::get<2, TestType>::value;
  static constexpr int total_warps      = 2;
};

C2H_TEST("Warp sort on keys-only works", "[sort][warp]", key_types, items_per_thread_list, num_block_dims_list)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  // Prepare test data
  const int valid_items     = params::items_per_thread * warp_threads;
  constexpr int total_warps = params::total_warps;
  const int total_items     = total_warps * valid_items;
  c2h::device_vector<type> d_in(total_items);
  c2h::device_vector<type> d_out(total_items);
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_bitonic_sort<params::items_per_thread, total_warps>(
    d_in, d_out, valid_items, sort_keys_full_t{}, params::num_block_dims);

  // Prepare verification data
  c2h::host_vector<type> h_in_out = d_in;
  compute_host_reference(h_in_out.begin(), valid_items, total_warps);

  // Verify results
  const c2h::host_vector<type> h_out(d_out);
  REQUIRE(h_in_out == h_out);
}

C2H_TEST("Warp sort keys-only on partial warp-tile works",
         "[sort][warp]",
         key_types,
         items_per_thread_list,
         num_block_dims_list,
         c2h::type_list<sort_keys_partial_oob_t, sort_keys_partial_t>)
{
  using params   = params_t<TestType>;
  using type     = typename params::type;
  using action_t = typename c2h::get<3, TestType>;

  // Prepare test data
  const int valid_items = GENERATE(
    0,
    1,
    params::items_per_thread * warp_threads - 1,
    params::items_per_thread * warp_threads,
    take(5, random(2, params::items_per_thread * warp_threads - 2)));
  constexpr int total_warps = params::total_warps;
  const int total_items     = total_warps * valid_items;
  c2h::device_vector<type> d_in(total_items);
  c2h::device_vector<type> d_out(total_items);
  c2h::gen(C2H_SEED(5), d_in);

  // Run test
  warp_bitonic_sort<params::items_per_thread, total_warps>(d_in, d_out, valid_items, action_t{}, params::num_block_dims);

  // Prepare verification data
  c2h::host_vector<type> h_in_out(d_in);
  compute_host_reference(h_in_out.begin(), valid_items, total_warps);

  // Verify results
  const c2h::host_vector<type> h_out(d_out);
  REQUIRE(h_in_out == h_out);
}

C2H_TEST("Warp sort on keys-value pairs works",
         "[sort][warp]",
         key_types,
         items_per_thread_list,
         num_block_dims_list,
         value_types)
{
  using params     = params_t<TestType>;
  using key_type   = typename params::type;
  using value_type = typename c2h::get<3, TestType>;

  // Prepare test data
  const int valid_items     = params::items_per_thread * warp_threads;
  constexpr int total_warps = params::total_warps;
  const int total_items     = total_warps * valid_items;
  c2h::device_vector<key_type> d_keys_in(total_items);
  c2h::device_vector<key_type> d_keys_out(total_items);
  c2h::device_vector<value_type> d_values_in(total_items);
  c2h::device_vector<value_type> d_values_out(total_items);
  c2h::gen(C2H_SEED(10), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  // Run test
  warp_bitonic_sort<params::items_per_thread, total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, valid_items, sort_pairs_full_t{}, params::num_block_dims);

  // Prepare verification data
  c2h::host_vector<key_type> h_keys_in_out     = d_keys_in;
  c2h::host_vector<value_type> h_values_in_out = d_values_in;
  auto cpu_kv_pairs = thrust::make_zip_iterator(h_keys_in_out.begin(), h_values_in_out.begin());
  compute_host_reference(cpu_kv_pairs, valid_items, total_warps);

  // Verify results
  const c2h::host_vector<key_type> h_keys_out(d_keys_out);
  REQUIRE(h_keys_in_out == h_keys_out);

  sort_values_for_equal_keys(h_keys_in_out.begin(), h_values_in_out.begin(), valid_items, total_warps);
  sort_values_for_equal_keys(d_keys_out.begin(), d_values_out.begin(), valid_items, total_warps);

  const c2h::host_vector<value_type> h_values_out(d_values_out);
  REQUIRE(h_values_in_out == h_values_out);
}

C2H_TEST("Warp sort on key-value pairs of a partial warp-tile works",
         "[sort][warp]",
         key_types,
         items_per_thread_list,
         num_block_dims_list,
         value_types,
         c2h::type_list<sort_pairs_partial_oob_t, sort_pairs_partial_t>)
{
  using params     = params_t<TestType>;
  using key_type   = typename params::type;
  using value_type = typename c2h::get<3, TestType>;
  using action_t   = typename c2h::get<4, TestType>;

  // Prepare test data
  const int valid_items = GENERATE(
    0,
    1,
    params::items_per_thread * warp_threads - 1,
    params::items_per_thread * warp_threads,
    take(5, random(2, params::items_per_thread * warp_threads - 2)));
  constexpr int total_warps = params::total_warps;
  const int total_items     = total_warps * valid_items;
  c2h::device_vector<key_type> d_keys_in(total_items);
  c2h::device_vector<key_type> d_keys_out(total_items);
  c2h::device_vector<value_type> d_values_in(total_items);
  c2h::device_vector<value_type> d_values_out(total_items);
  c2h::gen(C2H_SEED(5), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  // Run test
  warp_bitonic_sort<params::items_per_thread, total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, valid_items, action_t{}, params::num_block_dims);

  // Prepare verification data
  c2h::host_vector<key_type> h_keys_in_out     = d_keys_in;
  c2h::host_vector<value_type> h_values_in_out = d_values_in;
  auto cpu_kv_pairs = thrust::make_zip_iterator(h_keys_in_out.begin(), h_values_in_out.begin());
  compute_host_reference(cpu_kv_pairs, valid_items, total_warps);

  // Verify results
  const c2h::host_vector<key_type> h_keys_out(d_keys_out);
  REQUIRE(h_keys_in_out == h_keys_out);

  sort_values_for_equal_keys(h_keys_in_out.begin(), h_values_in_out.begin(), valid_items, total_warps);
  sort_values_for_equal_keys(d_keys_out.begin(), d_values_out.begin(), valid_items, total_warps);

  const c2h::host_vector<value_type> h_values_out(d_values_out);
  REQUIRE(h_values_in_out == h_values_out);
}
