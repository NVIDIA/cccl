// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/std/limits>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

using key_types               = fundamental_types;
using value_types             = offset_types;
using multiple_of_32_sequence = nvbench::enum_type_list<32, 64, 96, 128, 160, 192, 224, 256>;

constexpr int WARP_THREADS   = 32;
constexpr int NUM_ITERATIONS = 100;
constexpr int BLOCK_SIZE     = 128;

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

template <typename Kernel>
void calc_launch_params(Mode mode, int num_SMs, int block_size, Kernel kernel, int& grid_dim, int& block_dim)
{
  if (mode == Mode::Latency)
  {
    grid_dim  = 1;
    block_dim = WARP_THREADS;
  }
  else if (mode == Mode::Throughput)
  {
    int max_blocks_per_SM = 0;
    NVBENCH_CUDA_CALL_NOEXCEPT(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
    grid_dim  = max_blocks_per_SM * num_SMs;
    block_dim = block_size;
  }
}

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int LEN>
void run_bench(nvbench::state& state)
{
  constexpr int items_per_thread = LEN / WARP_THREADS;
  const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int>;

  const int num_SMs = state.get_device().value().get_number_of_sms();
  int grid_dim;
  int block_dim;
  calc_launch_params(mode, num_SMs, BLOCK_SIZE, kernel, grid_dim, block_dim);
  state.add_element_count(grid_dim * (block_dim / WARP_THREADS) * LEN * NUM_ITERATIONS);

  state.exec([grid_dim, block_dim, kernel](nvbench::launch& launch) {
    kernel<<<grid_dim, block_dim, 0, launch.get_stream()>>>(NUM_ITERATIONS, ActionT{}, LEN);
  });
}

template <int ITEMS_PER_THREAD>
struct full_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int) const
  {
    cub::detail::WarpBitonicSort<ITEMS_PER_THREAD, KeyT, ValueT>{}.Sort(keys, values, CustomLess{});
  }
};

template <Mode mode, typename KeyT, typename ValueT, int len>
void full(nvbench::state& state, nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<len>>)
{
  run_bench<full_op_t<len / WARP_THREADS>, mode, KeyT, ValueT, len>(state);
}

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(modes, key_types, value_types, multiple_of_32_sequence))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});

template <int ITEMS_PER_THREAD>
struct partial_oob_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int len) const
  {
    cub::detail::WarpBitonicSort<ITEMS_PER_THREAD, KeyT, ValueT>{}.Sort(
      keys, values, CustomLess{}, len, CustomLess::oob<KeyT>);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int len>
void partial_oob(nvbench::state& state,
                 nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<len>>)
{
  run_bench<partial_oob_op_t<len / WARP_THREADS>, mode, KeyT, ValueT, len>(state);
}

NVBENCH_BENCH_TYPES(partial_oob, NVBENCH_TYPE_AXES(modes, key_types, value_types, multiple_of_32_sequence))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});

template <int ITEMS_PER_THREAD>
struct partial_op_t
{
  template <typename KeyT, typename ValueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  operator()(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], int len) const
  {
    cub::detail::WarpBitonicSort<ITEMS_PER_THREAD, KeyT, ValueT>{}.Sort(keys, values, CustomLess{}, len);
  }
};

template <Mode mode, typename KeyT, typename ValueT, int len>
void partial(nvbench::state& state, nvbench::type_list<nvbench::enum_type<mode>, KeyT, ValueT, nvbench::enum_type<len>>)
{
  run_bench<partial_op_t<len / WARP_THREADS>, mode, KeyT, ValueT, len>(state);
}

NVBENCH_BENCH_TYPES(partial, NVBENCH_TYPE_AXES(modes, key_types, value_types, multiple_of_32_sequence))
  .set_type_axes_names({"mode", "KeyT", "ValueT", "len"});
