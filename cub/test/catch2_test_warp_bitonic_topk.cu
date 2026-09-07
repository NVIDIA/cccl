// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/util_arch.cuh>
#include <cub/warp/warp_bitonic_topk.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>

#include <cuda/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

struct CustomLess
{
  template <typename T>
  [[nodiscard]] __device__ __host__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }

  template <typename T>
  [[nodiscard]] static __device__ __host__ T get_oob_default()
  {
    if constexpr (cuda::std::is_same_v<T, float>)
    {
      return cuda::std::numeric_limits<T>::infinity();
    }
    else
    {
      return cuda::std::numeric_limits<T>::max();
    }
  };
};

using cub::detail::warp_threads;
using cub::detail::WarpBitonicTopKAlgorithm;

/**
 * @brief Kernel to dispatch to the appropriate WarpBitonicTopK member function, for keys-only.
 */
template <WarpBitonicTopKAlgorithm Algo,
          int LogicalWarpThreads,
          int MaxK,
          int ItemsPerThread,
          int TotalWarps,
          typename KeyT,
          typename ActionT>
__global__ void array_kernel(KeyT* keys_in, KeyT* keys_out, int k, int num_items, ActionT action)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, LogicalWarpThreads, cub::NullType, Algo>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / LogicalWarpThreads;
  const int lane    = tid % LogicalWarpThreads;

  // Test case of partially finished CTA
  if (warp_id >= TotalWarps)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[ItemsPerThread];

  // Instantiate warp-scope algorithm
  __shared__ typename warp_topk_t::TempStorage temp_storage[TotalWarps];
  warp_topk_t warp_topk{temp_storage[warp_id]};

  // Load data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * LogicalWarpThreads + lane;
    if (idx < num_items)
    {
      keys[i] = keys_in[num_items * warp_id + idx];
    }
  }

  // Run bitonic top-k
  action(warp_topk, keys, k, num_items);

  // Store data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * LogicalWarpThreads + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx] = keys[i];
    }
  }
}

template <WarpBitonicTopKAlgorithm Algo,
          int LogicalWarpThreads,
          int MaxK,
          int ItemsPerThread,
          int TotalWarps,
          typename KeyT,
          typename ActionT>
__global__ void iterator_kernel(KeyT* keys_in, KeyT* keys_out, int k, int num_items, ActionT action)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, LogicalWarpThreads, cub::NullType, Algo>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / LogicalWarpThreads;
  const int lane    = tid % LogicalWarpThreads;

  // Test case of partially finished CTA
  if (warp_id >= TotalWarps)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[MaxK / LogicalWarpThreads];

  // Instantiate warp-scope algorithm
  __shared__ typename warp_topk_t::TempStorage temp_storage[TotalWarps];
  warp_topk_t warp_topk{temp_storage[warp_id]};

  // Run bitonic top-k
  const int in_offset = num_items * warp_id;
  action(warp_topk, keys_in + in_offset, k, num_items, keys);

  // Store data
  for (int i = 0; i < MaxK / LogicalWarpThreads; ++i)
  {
    const int idx = i * LogicalWarpThreads + lane;
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
template <WarpBitonicTopKAlgorithm Algo,
          int LogicalWarpThreads,
          int MaxK,
          int ItemsPerThread,
          int TotalWarps,
          typename KeyT,
          typename ValueT,
          typename ActionT>
__global__ void
array_kernel(KeyT* keys_in, KeyT* keys_out, ValueT* values_in, ValueT* values_out, int k, int num_items, ActionT action)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, LogicalWarpThreads, ValueT, Algo>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / LogicalWarpThreads;
  const int lane    = tid % LogicalWarpThreads;

  // Test case of partially finished CTA
  if (warp_id >= TotalWarps)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];

  // Instantiate warp-scope algorithm
  __shared__ typename warp_topk_t::TempStorage temp_storage[TotalWarps];
  warp_topk_t warp_topk{temp_storage[warp_id]};

  // Load data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * LogicalWarpThreads + lane;
    if (idx < num_items)
    {
      keys[i]   = keys_in[num_items * warp_id + idx];
      values[i] = values_in[num_items * warp_id + idx];
    }
  }

  // Run bitonic top-k
  action(warp_topk, keys, values, k, num_items);

  // Store data
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = i * LogicalWarpThreads + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx]   = keys[i];
      values_out[k * warp_id + idx] = values[i];
    }
  }
}

template <WarpBitonicTopKAlgorithm Algo,
          int LogicalWarpThreads,
          int MaxK,
          int ItemsPerThread,
          int TotalWarps,
          typename KeyT,
          typename ValueT,
          typename ActionT>
__global__ void iterator_kernel(
  KeyT* keys_in, KeyT* keys_out, ValueT* values_in, ValueT* values_out, int k, int num_items, ActionT action)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MaxK, KeyT, LogicalWarpThreads, ValueT, Algo>;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / LogicalWarpThreads;
  const int lane    = tid % LogicalWarpThreads;

  // Test case of partially finished CTA
  if (warp_id >= TotalWarps)
  {
    return;
  }

  // Thread-local storage
  KeyT keys[MaxK / LogicalWarpThreads];
  ValueT values[MaxK / LogicalWarpThreads];

  // Instantiate warp-scope algorithm
  __shared__ typename warp_topk_t::TempStorage temp_storage[TotalWarps];
  warp_topk_t warp_topk{temp_storage[warp_id]};

  // Run bitonic top-k
  const int in_offset = num_items * warp_id;
  action(warp_topk, keys_in + in_offset, values_in + in_offset, k, num_items, keys, values);

  // Store data
  for (int i = 0; i < MaxK / LogicalWarpThreads; ++i)
  {
    const int idx = i * LogicalWarpThreads + lane;
    if (idx < k)
    {
      keys_out[k * warp_id + idx]   = keys[i];
      values_out[k * warp_id + idx] = values[i];
    }
  }
}

// -----------------------------------------------------------
// Dimensions being instantiated:
// {full,partial-oob,partial,iterator} x {keys, kv-pairs}
// -----------------------------------------------------------

/**
 * @brief Delegate wrapper for WarpBitonicTopK::TopK on keys-only
 */
struct topk_keys_full_t
{
  template <int ItemsPerThread, typename KeyT, typename WarpTopKT>
  __device__ void operator()(WarpTopKT& warp_topk, KeyT (&thread_data)[ItemsPerThread], int k, int /*num_items*/) const
  {
    warp_topk.TopK(thread_data, CustomLess{}, k);
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicTopK::TopK on keys-only with oob_default
 */
struct topk_keys_partial_oob_t
{
  template <int ItemsPerThread, typename KeyT, typename WarpTopKT>
  __device__ void operator()(WarpTopKT& warp_topk, KeyT (&thread_data)[ItemsPerThread], int k, int num_items) const
  {
    warp_topk.TopK(thread_data, CustomLess{}, k, num_items, CustomLess::get_oob_default<KeyT>());
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicTopK::TopK on keys-only
 */
struct topk_keys_partial_t
{
  template <int ItemsPerThread, typename KeyT, typename WarpTopKT>
  __device__ void operator()(WarpTopKT& warp_topk, KeyT (&thread_data)[ItemsPerThread], int k, int num_items) const
  {
    warp_topk.TopK(thread_data, CustomLess{}, k, num_items);
  }
};

struct topk_keys_iterator_t
{
  template <int MaxKPerThread, typename KeyInputIteratorT, typename KeyT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyInputIteratorT keys_in, int k, int num_items, KeyT (&keys_out)[MaxKPerThread]) const
  {
    warp_topk.TopK(keys_in, CustomLess{}, k, num_items, keys_out);
  }
};

/**
 * @brief Delegate wrapper for WarpBitonicTopK::TopK on key-value pairs
 */
struct topk_pairs_full_t
{
  template <int ItemsPerThread, typename KeyT, typename ValueT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int k, int /*num_items*/) const
  {
    warp_topk.TopK(keys, values, CustomLess{}, k);
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicTopK::TopK on key-value pairs with oob_default
 */
struct topk_pairs_partial_oob_t
{
  template <int ItemsPerThread, typename KeyT, typename ValueT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int k, int num_items) const
  {
    warp_topk.TopK(keys, values, CustomLess{}, k, num_items, CustomLess::get_oob_default<KeyT>());
  }
};

/**
 * @brief Delegate wrapper for partial WarpBitonicTopK::TopK on key-value pairs
 */
struct topk_pairs_partial_t
{
  template <int ItemsPerThread, typename KeyT, typename ValueT, typename WarpTopKT>
  __device__ void operator()(
    WarpTopKT& warp_topk, KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], int k, int num_items) const
  {
    warp_topk.TopK(keys, values, CustomLess{}, k, num_items);
  }
};

struct topk_pairs_iterator_t
{
  template <int MaxKPerThread,
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
    KeyT (&keys_out)[MaxKPerThread],
    ValueT (&values_out)[MaxKPerThread]) const
  {
    warp_topk.TopK(keys_in, values_in, CustomLess{}, k, num_items, keys_out, values_out);
  }
};

/**
 * @brief Dispatch helper function for keys
 */
template <WarpBitonicTopKAlgorithm Algo,
          int LogicalWarpThreads,
          int MaxK,
          int ItemsPerThread,
          int TotalWarps,
          typename KeyT,
          typename ActionT>
void warp_bitonic_topk(c2h::device_vector<KeyT>& in, c2h::device_vector<KeyT>& out, int k, int num_items, ActionT action)
{
  const auto kernel = [] {
    if constexpr (cuda::std::is_same_v<ActionT, topk_keys_iterator_t>)
    {
      return &iterator_kernel<Algo, LogicalWarpThreads, MaxK, ItemsPerThread, TotalWarps, KeyT, ActionT>;
    }
    else
    {
      return &array_kernel<Algo, LogicalWarpThreads, MaxK, ItemsPerThread, TotalWarps, KeyT, ActionT>;
    }
  }();

  // deliberately round up block dim to architectural warp size
  kernel<<<1, ::cuda::round_up(LogicalWarpThreads * TotalWarps, warp_threads)>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), k, num_items, action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Dispatch helper function for key-value pairs
 */
template <WarpBitonicTopKAlgorithm Algo,
          int LogicalWarpThreads,
          int MaxK,
          int ItemsPerThread,
          int TotalWarps,
          typename KeyT,
          typename ValueT,
          typename ActionT>
void warp_bitonic_topk(
  c2h::device_vector<KeyT>& keys_in,
  c2h::device_vector<KeyT>& keys_out,
  c2h::device_vector<ValueT>& values_in,
  c2h::device_vector<ValueT>& values_out,
  int k,
  int num_items,
  ActionT action)
{
  const auto kernel = [] {
    if constexpr (cuda::std::is_same_v<ActionT, topk_pairs_iterator_t>)
    {
      return &iterator_kernel<Algo, LogicalWarpThreads, MaxK, ItemsPerThread, TotalWarps, KeyT, ValueT, ActionT>;
    }
    else
    {
      return &array_kernel<Algo, LogicalWarpThreads, MaxK, ItemsPerThread, TotalWarps, KeyT, ValueT, ActionT>;
    }
  }();

  // deliberately round up block dim to architectural warp size
  kernel<<<1, ::cuda::round_up(LogicalWarpThreads * TotalWarps, warp_threads)>>>(
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

    // The keys of the WarpBitonicTopK result are sorted, so check them using std::sort.
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
    auto pair_out_begin = thrust::make_zip_iterator(keys_out.begin() + out_offset, values_out.begin() + out_offset);
    auto pair_out_end   = pair_out_begin + k;

    std::sort(pair_in_begin, pair_in_end);
    // The keys of the WarpBitonicTopK result are sorted, so check them are sorted before sorting the output pairs
    REQUIRE(c2h::host_vector<KeyT>(keys_in.begin() + in_offset, keys_in.begin() + in_offset + k)
            == c2h::host_vector<KeyT>(keys_out.begin() + out_offset, keys_out.begin() + out_offset + k));

    // Top-k results are not unique when ties exist at the k-th key, so further check that each output (key, value) pair
    // appears in the input
    std::sort(pair_out_begin, pair_out_end);
    REQUIRE(std::includes(pair_in_begin, pair_in_end, pair_out_begin, pair_out_end));
  }
}

// List of key types to test
using key_types = c2h::type_list<std::uint8_t, float>;

// List of value types
using value_types = c2h::type_list<std::int32_t>;

using algo_list =
  c2h::enum_type_list<WarpBitonicTopKAlgorithm, WarpBitonicTopKAlgorithm::eager, WarpBitonicTopKAlgorithm::buffered>;
using logical_warp_threads_list   = c2h::enum_type_list<int, 1, 4, warp_threads>;
using max_k_per_thread_list       = c2h::enum_type_list<int, 1, 2, 3>;
using extra_items_per_thread_list = c2h::enum_type_list<int, 0, 1, 2, 3, 4>;

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr WarpBitonicTopKAlgorithm algo = c2h::get<1, TestType>::value;
  static constexpr int logical_warp_threads      = c2h::get<2, TestType>::value;
  static constexpr int max_k                     = logical_warp_threads * c2h::get<3, TestType>::value;
  static constexpr int extra_items_per_thread    = c2h::get<4, TestType>::value;
  static constexpr int items_per_thread          = max_k / logical_warp_threads + extra_items_per_thread;
  // When the logical warp is smaller than the architectural warp, deliberately leaves the last architectural warp
  // partially occupied
  static constexpr int total_warps =
    (logical_warp_threads == warp_threads) ? 2 : (2 * warp_threads / logical_warp_threads - 1);
};

template <typename TestType, typename ActionT>
void test_topk_keys(int k, int num_items)
{
  using params                       = params_t<TestType>;
  using key_type                     = typename params::type;
  constexpr auto algo                = params::algo;
  constexpr int logical_warp_threads = params::logical_warp_threads;
  constexpr int max_k                = params::max_k;
  constexpr int items_per_thread     = params::items_per_thread;
  constexpr int total_warps          = params::total_warps;

  // Prepare test data
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::gen(C2H_SEED(5), d_keys_in);

  // Run test
  warp_bitonic_topk<algo, logical_warp_threads, max_k, items_per_thread, total_warps>(
    d_keys_in, d_keys_out, k, num_items, ActionT{});

  // Verify results
  verify_topk_result(d_keys_in, d_keys_out, total_warps);
}

template <typename TestType, typename ActionT>
void test_topk_pairs(int k, int num_items)
{
  using params                       = params_t<TestType>;
  using key_type                     = typename params::type;
  using value_type                   = typename c2h::get<5, TestType>;
  constexpr auto algo                = params::algo;
  constexpr int logical_warp_threads = params::logical_warp_threads;
  constexpr int max_k                = params::max_k;
  constexpr int items_per_thread     = params::items_per_thread;
  constexpr int total_warps          = params::total_warps;

  // Prepare test data
  c2h::device_vector<key_type> d_keys_in(total_warps * num_items);
  c2h::device_vector<value_type> d_values_in(total_warps * num_items);
  c2h::device_vector<key_type> d_keys_out(total_warps * k);
  c2h::device_vector<value_type> d_values_out(total_warps * k);
  c2h::gen(C2H_SEED(5), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  // Run test
  warp_bitonic_topk<algo, logical_warp_threads, max_k, items_per_thread, total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, k, num_items, ActionT{});

  // Verify results
  verify_topk_result(d_keys_in, d_values_in, d_keys_out, d_values_out, total_warps);
}

CUB_TEST("Warp top-k on keys works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         algo_list,
         logical_warp_threads_list,
         max_k_per_thread_list,
         extra_items_per_thread_list)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = params::items_per_thread * params::logical_warp_threads;

  test_topk_keys<TestType, topk_keys_full_t>(k, num_items);
}

CUB_TEST("Warp top-k on keys of a partial warp-tile with an OOB default works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         c2h::enum_type_list<WarpBitonicTopKAlgorithm, WarpBitonicTopKAlgorithm::eager>,
         logical_warp_threads_list,
         max_k_per_thread_list,
         extra_items_per_thread_list)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, params::items_per_thread * params::logical_warp_threads)));

  test_topk_keys<TestType, topk_keys_partial_oob_t>(k, num_items);
}

CUB_TEST("Warp top-k on keys of a partial warp-tile works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         algo_list,
         logical_warp_threads_list,
         max_k_per_thread_list,
         extra_items_per_thread_list)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, params::items_per_thread * params::logical_warp_threads)));

  test_topk_keys<TestType, topk_keys_partial_t>(k, num_items);
}

CUB_TEST("Warp top-k iterator on keys of a partial warp-tile works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         c2h::enum_type_list<WarpBitonicTopKAlgorithm, WarpBitonicTopKAlgorithm::buffered>,
         logical_warp_threads_list,
         max_k_per_thread_list,
         c2h::enum_type_list<int, 0>)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, k + 1, take(5, random(k + 2, k + 100)), take(5, random(k + 100, k + 1000)));

  test_topk_keys<TestType, topk_keys_iterator_t>(k, num_items);
}

CUB_TEST("Warp top-k on key-value pairs works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         algo_list,
         logical_warp_threads_list,
         max_k_per_thread_list,
         extra_items_per_thread_list,
         value_types)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = params::items_per_thread * params::logical_warp_threads;

  test_topk_pairs<TestType, topk_pairs_full_t>(k, num_items);
}

CUB_TEST("Warp top-k on key-value pairs of a partial warp-tile with an OOB default works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         c2h::enum_type_list<WarpBitonicTopKAlgorithm, WarpBitonicTopKAlgorithm::eager>,
         logical_warp_threads_list,
         max_k_per_thread_list,
         extra_items_per_thread_list,
         value_types)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, params::items_per_thread * params::logical_warp_threads)));

  test_topk_pairs<TestType, topk_pairs_partial_oob_t>(k, num_items);
}

CUB_TEST("Warp top-k on key-value pairs of a partial warp-tile works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         algo_list,
         logical_warp_threads_list,
         max_k_per_thread_list,
         extra_items_per_thread_list,
         value_types)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, params::items_per_thread * params::logical_warp_threads)));

  test_topk_pairs<TestType, topk_pairs_partial_t>(k, num_items);
}

CUB_TEST("Warp top-k iterator on key-value pairs of a partial warp-tile works",
         "[topk][warp]",
         CUB_SMALL,
         key_types,
         c2h::enum_type_list<WarpBitonicTopKAlgorithm, WarpBitonicTopKAlgorithm::buffered>,
         logical_warp_threads_list,
         max_k_per_thread_list,
         c2h::enum_type_list<int, 0>, // unused
         value_types)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, k + 1, take(5, random(k + 2, k + 100)), take(5, random(k + 100, k + 1000)));

  test_topk_pairs<TestType, topk_pairs_iterator_t>(k, num_items);
}

// Keep custom_t coverage narrow because it is expensive to instantiate.
using custom_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

CUB_TEST("Warp top-k on custom key-value pairs works",
         "[topk][warp]",
         CUB_SMALL,
         c2h::type_list<custom_t>, // key_types,
         algo_list,
         c2h::enum_type_list<int, 4, warp_threads>, // logical_warp_threads
         c2h::enum_type_list<int, 2>, // max_k
         c2h::enum_type_list<int, 1>, // extra_items_per_thread
         c2h::type_list<custom_t>, // value_types,
         c2h::type_list<topk_pairs_partial_oob_t, topk_pairs_partial_t, topk_pairs_iterator_t>)
{
  using params        = params_t<TestType>;
  constexpr int max_k = params::max_k;

  const int k         = GENERATE_COPY(1, max_k, take(5, random(1, max_k)));
  const int num_items = GENERATE_COPY(k, take(5, random(k, params::items_per_thread * params::logical_warp_threads)));

  test_topk_pairs<TestType, topk_pairs_partial_t>(k, num_items);
}
