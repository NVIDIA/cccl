// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_topk.cuh>

#include <thrust/iterator/zip_iterator.h>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <algorithm>

#include <c2h/catch2_test_helper.h>

struct CustomLess
{
  template <typename T>
  __device__ __host__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }
};

constexpr int WARP_THREADS = cub::detail::warp_threads;

/**
 * @brief Kernel to dispatch to the appropriate WarpBitonicTopK member function, for keys-only.
 */
template <int MAX_K, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename KeyT, typename ActionT>
__global__ void warp_bitonic_topk_kernel(KeyT* keys_in, KeyT* keys_out, int k, int num_items, ActionT action)
{
  using warp_bitonic_topk_t = cub::detail::WarpBitonicTopK<MAX_K, KeyT>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / WARP_THREADS;
  const int lane    = tid % WARP_THREADS;

  // Test case of partially finished CTA
  if (warp_id >= TOTAL_WARPS)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[ITEMS_PER_THREAD];

  // Instantiate warp-scope algorithm
  warp_bitonic_topk_t warp_topk;

  // Load data
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    const int idx = i * WARP_THREADS + lane;
    if (idx < num_items)
    {
      keys[i] = keys_in[num_items * warp_id + idx];
    }
  }

  // Run bitonic topk
  action(warp_topk, keys, k, num_items);

  // Store data
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    const int idx = i * WARP_THREADS + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx] = keys[i];
    }
  }
}

template <int MAX_K, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename KeyT, typename ActionT>
__global__ void warp_bitonic_topk_iterator_kernel(KeyT* keys_in, KeyT* keys_out, int k, int num_items, ActionT action)
{
  using warp_bitonic_topk_t = cub::detail::WarpBitonicTopK<MAX_K, KeyT>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / WARP_THREADS;
  const int lane    = tid % WARP_THREADS;

  // Test case of partially finished CTA
  if (warp_id >= TOTAL_WARPS)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[MAX_K / WARP_THREADS];

  // Instantiate warp-scope algorithm
  __shared__ typename warp_bitonic_topk_t::TempStorage temp_storage[TOTAL_WARPS];
  warp_bitonic_topk_t warp_topk(temp_storage[warp_id]);

  // Run bitonic topk
  const int in_offset = num_items * warp_id;
  action(warp_topk, keys_in + in_offset, k, num_items, keys);

  // Store data
  for (int i = 0; i < MAX_K / WARP_THREADS; ++i)
  {
    const int idx = i * WARP_THREADS + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx] = keys[i];
    }
  }
}

/**
 * @brief Kernel to dispatch to the appropriate WarpBitonicTopK member function, for key-value
 * pairs.
 */
template <int MAX_K, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename KeyT, typename ValueT, typename ActionT>
__global__ void warp_bitonic_topk_kernel(
  KeyT* keys_in, KeyT* keys_out, ValueT* values_in, ValueT* values_out, int k, int num_items, ActionT action)
{
  using warp_bitonic_topk_t = cub::detail::WarpBitonicTopK<MAX_K, KeyT, ValueT>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / WARP_THREADS;
  const int lane    = tid % WARP_THREADS;

  // Test case of partially finished CTA
  if (warp_id >= TOTAL_WARPS)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[ITEMS_PER_THREAD];
  ValueT values[ITEMS_PER_THREAD];

  // Instantiate warp-scope algorithm
  warp_bitonic_topk_t warp_topk;

  // Load data
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    const int idx = i * WARP_THREADS + lane;
    if (idx < num_items)
    {
      keys[i]   = keys_in[num_items * warp_id + idx];
      values[i] = values_in[num_items * warp_id + idx];
    }
  }

  // Run bitonic topk
  action(warp_topk, keys, values, k, num_items);

  // Store data
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    const int idx = i * WARP_THREADS + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx]   = keys[i];
      values_out[k * warp_id + idx] = values[i];
    }
  }
}

template <int MAX_K, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename KeyT, typename ValueT, typename ActionT>
__global__ void warp_bitonic_topk_iterator_kernel(
  KeyT* keys_in, KeyT* keys_out, ValueT* values_in, ValueT* values_out, int k, int num_items, ActionT action)
{
  using warp_bitonic_topk_t = cub::detail::WarpBitonicTopK<MAX_K, KeyT, ValueT>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / WARP_THREADS;
  const int lane    = tid % WARP_THREADS;

  // Test case of partially finished CTA
  if (warp_id >= TOTAL_WARPS)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[MAX_K / WARP_THREADS];
  ValueT values[MAX_K / WARP_THREADS];

  // Instantiate warp-scope algorithm
  __shared__ typename warp_bitonic_topk_t::TempStorage temp_storage[TOTAL_WARPS];
  warp_bitonic_topk_t warp_topk(temp_storage[warp_id]);

  // Run bitonic topk
  const int in_offset = num_items * warp_id;
  action(warp_topk, keys_in + in_offset, values_in + in_offset, k, num_items, keys, values);

  // Store data
  for (int i = 0; i < MAX_K / WARP_THREADS; ++i)
  {
    const int idx = i * WARP_THREADS + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx]   = keys[i];
      values_out[k * warp_id + idx] = values[i];
    }
  }
}

// -----------------------------------------------------------
// Dimensions being instantiated:
// {full,partial,partial-iterator} x {keys, kv-pairs}
// -----------------------------------------------------------

/**
 * @brief Delegate wrapper for WarpBitonicTopK::TopK on keys-only
 */
struct topk_keys_t
{
  template <int ITEMS_PER_THREAD, typename KeyT, typename WarpTopKT>
  __device__ void operator()(WarpTopKT& warp_topk, KeyT (&thread_data)[ITEMS_PER_THREAD], int k, int /*num_items*/) const
  {
    warp_topk.TopK(thread_data, CustomLess{}, k);
  }

  static constexpr bool use_iterator{false};
};

/**
 * @brief Delegate wrapper for partial WarpBitonicTopK::TopK on keys-only
 */
struct partial_topk_keys_t
{
  template <int ITEMS_PER_THREAD, typename KeyT, typename WarpTopKT>
  __device__ void operator()(WarpTopKT& warp_topk, KeyT (&thread_data)[ITEMS_PER_THREAD], int k, int num_items) const
  {
    warp_topk.TopK(thread_data, CustomLess{}, k, num_items);
  }

  static constexpr bool use_iterator{false};
};

struct partial_topk_keys_iterator_t
{
  template <int MAX_K_PER_THREAD, typename KeyInputIteratorT, typename KeyT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyInputIteratorT keys_in, int k, int num_items, KeyT (&keys_out)[MAX_K_PER_THREAD]) const
  {
    warp_topk.TopK(keys_in, CustomLess{}, k, num_items, keys_out);
  }

  static constexpr bool use_iterator{true};
};

/**
 * @brief Delegate wrapper for WarpBitonicTopK::TopK on key-value pairs
 */
struct topk_pairs_t
{
  template <int ITEMS_PER_THREAD, typename KeyT, typename ValueT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int k, int /*num_items*/
  ) const
  {
    warp_topk.TopK(keys, values, CustomLess{}, k);
  }

  static constexpr bool use_iterator{false};
};

/**
 * @brief Delegate wrapper for partial WarpBitonicTopK::TopK on key-value pairs
 */
struct partial_topk_pairs_t
{
  template <int ITEMS_PER_THREAD, typename KeyT, typename ValueT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int k, int num_items) const
  {
    warp_topk.TopK(keys, values, CustomLess{}, k, num_items);
  }

  static constexpr bool use_iterator{false};
};

struct partial_topk_pairs_iterator_t
{
  template <int MAX_K_PER_THREAD,
            typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyT,
            typename ValueT,
            typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk,
    KeyInputIteratorT keys_in,
    ValueInputIteratorT values_in,
    int k,
    int num_items,
    KeyT (&keys_out)[MAX_K_PER_THREAD],
    ValueT (&values_out)[MAX_K_PER_THREAD]) const
  {
    warp_topk.TopK(keys_in, values_in, CustomLess{}, k, num_items, keys_out, values_out);
  }

  static constexpr bool use_iterator{true};
};

/**
 * @brief Dispatch helper function for keys
 */
template <int MAX_K, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename KeyT, typename ActionT>
void warp_bitonic_topk(c2h::device_vector<KeyT>& in, c2h::device_vector<KeyT>& out, int k, int num_items, ActionT action)
{
  using Kernel = void (*)(KeyT*, KeyT*, int, int, ActionT);
  Kernel kernel;
  if constexpr (ActionT::use_iterator)
  {
    kernel = warp_bitonic_topk_iterator_kernel<MAX_K, ITEMS_PER_THREAD, TOTAL_WARPS, KeyT, ActionT>;
  }
  else
  {
    kernel = warp_bitonic_topk_kernel<MAX_K, ITEMS_PER_THREAD, TOTAL_WARPS, KeyT, ActionT>;
  }

  kernel<<<1, WARP_THREADS * TOTAL_WARPS>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), k, num_items, action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Dispatch helper function for key-value pairs
 */
template <int MAX_K, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename KeyT, typename ValueT, typename ActionT>
void warp_bitonic_topk(
  c2h::device_vector<KeyT>& keys_in,
  c2h::device_vector<KeyT>& keys_out,
  c2h::device_vector<ValueT>& values_in,
  c2h::device_vector<ValueT>& values_out,
  int k,
  int num_items,
  ActionT action)
{
  using Kernel = void (*)(KeyT*, KeyT*, ValueT*, ValueT*, int, int, ActionT);
  Kernel kernel;
  if constexpr (ActionT::use_iterator)
  {
    kernel = warp_bitonic_topk_iterator_kernel<MAX_K, ITEMS_PER_THREAD, TOTAL_WARPS, KeyT, ValueT, ActionT>;
  }
  else
  {
    kernel = warp_bitonic_topk_kernel<MAX_K, ITEMS_PER_THREAD, TOTAL_WARPS, KeyT, ValueT, ActionT>;
  }

  kernel<<<1, WARP_THREADS * TOTAL_WARPS>>>(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    k,
    num_items,
    action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename KeyT>
void verify_topk_result(
  const c2h::device_vector<KeyT>& d_keys_in, const c2h::device_vector<KeyT>& d_keys_out, int total_warps)
{
  c2h::host_vector<KeyT> keys_in  = d_keys_in;
  c2h::host_vector<KeyT> keys_out = d_keys_out;

  const int num_items = keys_in.size() / total_warps;
  const int k         = keys_out.size() / total_warps;

  for (int warp_id = 0; warp_id < total_warps; ++warp_id)
  {
    const int in_offset  = warp_id * num_items;
    const int out_offset = warp_id * k;

    // the keys of WarpBitonicTopK result are sorted, so we check it using std::sort
    std::sort(keys_in.begin() + in_offset, keys_in.begin() + in_offset + num_items);
    REQUIRE(c2h::host_vector<KeyT>(keys_in.begin() + in_offset, keys_in.begin() + in_offset + k)
            == c2h::host_vector<KeyT>(keys_out.begin() + out_offset, keys_out.begin() + out_offset + k));
  }
}

template <typename KeyT, typename ValueT>
void verify_topk_result(
  const c2h::device_vector<KeyT>& d_keys_in,
  const c2h::device_vector<ValueT>& d_values_in,
  const c2h::device_vector<KeyT>& d_keys_out,
  const c2h::device_vector<ValueT>& d_values_out,
  int total_warps)
{
  c2h::host_vector<KeyT> keys_in      = d_keys_in;
  c2h::host_vector<ValueT> values_in  = d_values_in;
  c2h::host_vector<KeyT> keys_out     = d_keys_out;
  c2h::host_vector<ValueT> values_out = d_values_out;

  const int num_items = keys_in.size() / total_warps;
  const int k         = keys_out.size() / total_warps;

  for (int warp_id = 0; warp_id < total_warps; ++warp_id)
  {
    const int in_offset  = warp_id * num_items;
    const int out_offset = warp_id * k;

    auto pair_in_begin  = thrust::make_zip_iterator(keys_in.begin() + in_offset, values_in.begin() + in_offset);
    auto pair_in_end    = pair_in_begin + num_items;
    auto pair_begin_end = thrust::make_zip_iterator(keys_out.begin() + out_offset, values_out.begin() + out_offset);
    auto pair_out_end   = pair_begin_end + k;

    std::sort(pair_in_begin, pair_in_end);
    // the keys of WarpBitonicTopK result are sorted, so we check it before std::sort(pair_begin_end, pair_out_end)
    REQUIRE(c2h::host_vector<KeyT>(keys_in.begin() + in_offset, keys_in.begin() + in_offset + k)
            == c2h::host_vector<KeyT>(keys_out.begin() + out_offset, keys_out.begin() + out_offset + k));

    // Top-k results are not unique when ties exist at the k-th key, so further check that each output (key, value) pair
    // appears in the input
    std::sort(pair_begin_end, pair_out_end);
    REQUIRE(std::includes(pair_in_begin, pair_in_end, pair_begin_end, pair_out_end));
  }
}

// List of key types to test
using key_types = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t>;

// List of value types
using value_types = c2h::type_list<std::int32_t>;

using max_k_list                  = c2h::enum_type_list<int, WARP_THREADS, WARP_THREADS * 2, WARP_THREADS * 3>;
using extra_items_per_thread_list = c2h::enum_type_list<int, 0, 1, 2, 3, 4, 5, 6, 7>;

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int max_k                  = c2h::get<1, TestType>::value;
  static constexpr int extra_items_per_thread = c2h::get<2, TestType>::value;
  static constexpr int items_per_thread       = max_k / WARP_THREADS + extra_items_per_thread;
  static constexpr int total_warps            = 2;
};

C2H_TEST("Warp topk on keys works", "[topk][warp]", key_types, max_k_list, extra_items_per_thread_list)
{
  using params                   = params_t<TestType>;
  using key_type                 = typename params::type;
  constexpr int max_k            = params::max_k;
  constexpr int items_per_thread = params::items_per_thread;
  constexpr int total_warps      = params::total_warps;

  // Prepare test data
  const int k         = GENERATE_COPY(1, max_k - 1, max_k, take(5, random(2, max_k - 2)));
  const int num_items = items_per_thread * WARP_THREADS;
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::gen(C2H_SEED(10), d_keys_in);

  // Run test
  warp_bitonic_topk<max_k, items_per_thread, total_warps>(d_keys_in, d_keys_out, k, num_items, topk_keys_t{});

  // Verify results
  verify_topk_result(d_keys_in, d_keys_out, total_warps);
}

C2H_TEST(
  "Warp topk on keys of a partial warp-tile works", "[topk][warp]", key_types, max_k_list, extra_items_per_thread_list)
{
  using params                   = params_t<TestType>;
  using key_type                 = typename params::type;
  constexpr int max_k            = params::max_k;
  constexpr int items_per_thread = params::items_per_thread;
  constexpr int total_warps      = params::total_warps;

  // Prepare test data
  const int k         = GENERATE_COPY(1, max_k - 1, max_k, take(5, random(2, max_k - 2)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, items_per_thread * WARP_THREADS)));
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::gen(C2H_SEED(5), d_keys_in);

  // Run test
  warp_bitonic_topk<max_k, items_per_thread, total_warps>(d_keys_in, d_keys_out, k, num_items, partial_topk_keys_t{});

  // Verify results
  verify_topk_result(d_keys_in, d_keys_out, total_warps);
}

C2H_TEST("Warp topk iterator on keys of a partial warp-tile works",
         "[topk][warp]",
         key_types,
         max_k_list,
         c2h::enum_type_list<int, 0>)
{
  using params                   = params_t<TestType>;
  using key_type                 = typename params::type;
  constexpr int max_k            = params::max_k;
  constexpr int items_per_thread = params::items_per_thread;
  constexpr int total_warps      = params::total_warps;

  // Prepare test data
  const int k         = GENERATE_COPY(1, max_k - 1, max_k, take(5, random(2, max_k - 2)));
  const int num_items = GENERATE_COPY(k, k + 1, take(5, random(k + 2, k + 100)), take(5, random(k + 100, k + 1000)));
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::gen(C2H_SEED(5), d_keys_in);

  // Run test
  warp_bitonic_topk<max_k, items_per_thread, total_warps>(
    d_keys_in, d_keys_out, k, num_items, partial_topk_keys_iterator_t{});

  // Verify results
  verify_topk_result(d_keys_in, d_keys_out, total_warps);
}

C2H_TEST(
  "Warp topk on keys-value pairs works", "[topk][warp]", key_types, max_k_list, extra_items_per_thread_list, value_types)
{
  using params                   = params_t<TestType>;
  using key_type                 = typename params::type;
  using value_type               = typename c2h::get<3, TestType>;
  constexpr int max_k            = params::max_k;
  constexpr int items_per_thread = params::items_per_thread;
  constexpr int total_warps      = params::total_warps;

  // Prepare test data
  const int k         = GENERATE_COPY(1, max_k - 1, max_k, take(5, random(2, max_k - 2)));
  const int num_items = items_per_thread * WARP_THREADS;
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<value_type> d_values_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::device_vector<value_type> d_values_out(total_warps * k);
  c2h::gen(C2H_SEED(10), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  // Run test
  warp_bitonic_topk<max_k, items_per_thread, total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, k, num_items, topk_pairs_t{});

  // Verify results
  verify_topk_result(d_keys_in, d_values_in, d_keys_out, d_values_out, total_warps);
}

C2H_TEST("Warp topk on key-value pairs of a partial warp-tile works",
         "[topk][warp]",
         key_types,
         max_k_list,
         extra_items_per_thread_list,
         value_types)
{
  using params                   = params_t<TestType>;
  using key_type                 = typename params::type;
  using value_type               = typename c2h::get<3, TestType>;
  constexpr int max_k            = params::max_k;
  constexpr int items_per_thread = params::items_per_thread;
  constexpr int total_warps      = params::total_warps;

  // Prepare test data
  const int k         = GENERATE_COPY(1, max_k - 1, max_k, take(5, random(2, max_k - 2)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, items_per_thread * WARP_THREADS)));
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<value_type> d_values_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::device_vector<value_type> d_values_out(total_warps * k);
  c2h::gen(C2H_SEED(5), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  // Run test
  warp_bitonic_topk<max_k, items_per_thread, total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, k, num_items, partial_topk_pairs_t{});

  // Verify results
  verify_topk_result(d_keys_in, d_values_in, d_keys_out, d_values_out, total_warps);
}

C2H_TEST("Warp topk iterator on key-value pairs of a partial warp-tile works",
         "[topk][warp]",
         key_types,
         max_k_list,
         c2h::enum_type_list<int, 0>, // unused
         value_types)
{
  using params                   = params_t<TestType>;
  using key_type                 = typename params::type;
  using value_type               = typename c2h::get<3, TestType>;
  constexpr int max_k            = params::max_k;
  constexpr int items_per_thread = params::items_per_thread;
  constexpr int total_warps      = params::total_warps;

  // Prepare test data
  const int k         = GENERATE_COPY(1, max_k - 1, max_k, take(5, random(2, max_k - 2)));
  const int num_items = GENERATE_COPY(k, k + 1, take(5, random(k + 2, k + 100)), take(5, random(k + 100, k + 1000)));
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<value_type> d_values_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::device_vector<value_type> d_values_out(total_warps * k);
  c2h::gen(C2H_SEED(5), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  // Run test
  warp_bitonic_topk<max_k, items_per_thread, total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, k, num_items, partial_topk_pairs_iterator_t{});

  // Verify results
  verify_topk_result(d_keys_in, d_values_in, d_keys_out, d_values_out, total_warps);
}
