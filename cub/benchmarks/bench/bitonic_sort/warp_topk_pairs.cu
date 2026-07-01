// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_topk.cuh>

#include <thrust/device_vector.h>

#include <cuda/std/limits>

#include <vector>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

using key_types    = fundamental_types;
using value_types  = offset_types;
using len_values   = nvbench::enum_type_list<32, 64, 96, 128, 160, 192>;
using max_k_values = nvbench::enum_type_list<32, 64, 96>;
// the perf is mostly determined by max_k values, so just use k=1
const std::vector<nvbench::int64_t> k_values{1};

constexpr int WARP_THREADS                  = 32;
constexpr int NUM_ITERATIONS                = 100;
constexpr int BLOCK_DIM_FOR_THROUGHPUT_MODE = 128;

enum class Mode
{
  // launch single warp
  Latency,
  // launch one full wave of thread blocks. Measure Elem/s.
  Throughput
};
using modes = nvbench::enum_type_list<Mode::Latency, Mode::Throughput>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  Mode,
  // Callable to generate input strings:
  [](Mode value) {
    switch (value)
    {
      case Mode::Latency:
        return "latency";
      case Mode::Throughput:
        return "throughput";
      default:
        return "Unknown";
    }
  },
  // Callable to generate descriptions:
  [](auto) {
    return std::string{};
  })

struct CustomLess
{
  template <typename T>
  __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }

  template <typename T>
  static constexpr T oob = cuda::std::numeric_limits<T>::max();
};

template <Mode mode>
constexpr int calc_block_dim()
{
  if constexpr (mode == Mode::Latency)
  {
    return WARP_THREADS;
  }
  else
  {
    return BLOCK_DIM_FOR_THROUGHPUT_MODE;
  }
}

template <Mode mode, typename Kernel>
int calc_grid_dim(int num_SMs, int block_dim, Kernel kernel)
{
  if constexpr (mode == Mode::Latency)
  {
    return 1;
  }
  else
  {
    int max_blocks_per_SM = 0;
    NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_dim, 0));
    return max_blocks_per_SM * num_SMs;
  }
}

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int LEN, int MAX_K>
void run_topk(nvbench::state& state, int num_items)
{
  if constexpr (MAX_K > LEN)
  {
    state.skip("Skipping workload where max_k > len.");
  }
  else
  {
    const int k = static_cast<int>(state.get_int64("k"));
    if (k > MAX_K || k > num_items)
    {
      state.skip("Skipping workload where k > max_k or k > num_items.");
      return;
    }

    constexpr int items_per_thread = LEN / WARP_THREADS;
    const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int, int>;

    const int num_SMs       = state.get_device().value().get_number_of_sms();
    constexpr int block_dim = calc_block_dim<mode>();
    const int grid_dim      = calc_grid_dim<mode>(num_SMs, block_dim, kernel);
    state.add_element_count(grid_dim * (block_dim / WARP_THREADS) * num_items * NUM_ITERATIONS);

    state.exec([grid_dim, block_dim, kernel, k, num_items](nvbench::launch& launch) {
      kernel<<<grid_dim, block_dim, 0, launch.get_stream()>>>(NUM_ITERATIONS, ActionT{}, k, num_items);
    });
  }
}

template <int MAX_K, int ITEMS_PER_THREAD>
struct full_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int k, int) const
  {
    cub::detail::WarpBitonicTopK<MAX_K, KeyT, ValueT>{}.TopK(keys, values, CustomLess{}, k);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int LEN, int MAX_K>
void full(nvbench::state& state,
          nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<LEN>, nvbench::enum_type<MAX_K>>)
{
  run_topk<full_op_t<MAX_K, LEN / WARP_THREADS>, mode, KeyT, ValueT, LEN, MAX_K>(state, LEN);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values, max_k_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len", "max_k"})
  .add_int64_axis("k", k_values);

template <int MAX_K, int ITEMS_PER_THREAD>
struct partial_oob_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int k, int len) const
  {
    cub::detail::WarpBitonicTopK<MAX_K, KeyT, ValueT>{}.TopK(keys, values, CustomLess{}, k, len, CustomLess::oob<KeyT>);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int LEN, int MAX_K>
void partial_oob(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<LEN>, nvbench::enum_type<MAX_K>>)
{
  constexpr int num_items = LEN;
  run_topk<partial_oob_op_t<MAX_K, LEN / WARP_THREADS>, mode, KeyT, ValueT, LEN, MAX_K>(state, num_items);
}

NVBENCH_BENCH_TYPES(partial_oob, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values, max_k_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len", "max_k"})
  .add_int64_axis("k", k_values);

template <int MAX_K, int ITEMS_PER_THREAD>
struct partial_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int k, int len) const
  {
    cub::detail::WarpBitonicTopK<MAX_K, KeyT, ValueT>{}.TopK(keys, values, CustomLess{}, k, len);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int LEN, int MAX_K>
void partial(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<LEN>, nvbench::enum_type<MAX_K>>)
{
  constexpr int num_items = LEN;
  run_topk<partial_op_t<MAX_K, LEN / WARP_THREADS>, mode, KeyT, ValueT, LEN, MAX_K>(state, num_items);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, key_types, value_types, len_values, max_k_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len", "max_k"})
  .add_int64_axis("k", k_values);

template <int THREADS_PER_BLOCK, int MAX_K, typename KeyT, typename ValueT>
__global__ void iterator_topk_kernel(int num_iterations, KeyT* keys_in, ValueT* values_in, int k, int num_items)
{
  using warp_topk_t = cub::detail::WarpBitonicTopK<MAX_K, KeyT, ValueT>;
  static_assert(THREADS_PER_BLOCK % WARP_THREADS == 0);
  constexpr int warps_per_block = THREADS_PER_BLOCK / WARP_THREADS;

  __shared__ typename warp_topk_t::TempStorage temp_storage[warps_per_block];

  const int warp_id      = blockIdx.x * warps_per_block + threadIdx.x / WARP_THREADS;
  const int input_offset = warp_id * num_items * num_iterations;

  KeyT keys_out[MAX_K / WARP_THREADS];
  ValueT values_out[MAX_K / WARP_THREADS];

  warp_topk_t warp_topk(temp_storage[threadIdx.x / WARP_THREADS]);
  for (int i = 0; i < num_iterations; ++i)
  {
    const int offset = input_offset + i * num_items;
    warp_topk.TopK(keys_in + offset, values_in + offset, CustomLess{}, k, num_items, keys_out, values_out);
    sink(keys_out);
    sink(values_out);
  }
}

template <Mode mode, typename KeyT, typename ValueT, int MAX_K>
void iterator(nvbench::state& state,
              nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<MAX_K>>)
{
  const int num_items = static_cast<int>(state.get_int64("len"));
  const int k         = static_cast<int>(state.get_int64("k"));

  if (MAX_K > num_items || k > MAX_K || k > num_items)
  {
    state.skip("Skipping workload where max_k > len, k > max_k, or k > len.");
    return;
  }

  constexpr int block_dim = calc_block_dim<mode>();
  const auto kernel       = iterator_topk_kernel<block_dim, MAX_K, KeyT, ValueT>;

  const int num_SMs  = state.get_device().value().get_number_of_sms();
  const int grid_dim = calc_grid_dim<mode>(num_SMs, block_dim, kernel);

  const size_t input_items_per_iteration = grid_dim * (block_dim / WARP_THREADS) * num_items;
  const size_t input_items               = input_items_per_iteration * NUM_ITERATIONS;

  thrust::device_vector<KeyT> keys_in     = generate(input_items);
  thrust::device_vector<ValueT> values_in = generate(input_items);

  state.add_element_count(input_items);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_dim, block_dim, 0, launch.get_stream()>>>(
      NUM_ITERATIONS,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(values_in.data()),
      k,
      num_items);
  });
}

NVBENCH_BENCH_TYPES(iterator, NVBENCH_TYPE_AXES(modes, key_types, value_types, max_k_values))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "max_k"})
  .add_int64_axis("len", {32, 64, 96, 128, 256, 512, 1024, 2048})
  .add_int64_axis("k", {32});
