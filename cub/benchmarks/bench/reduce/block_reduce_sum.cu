// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce.cuh>

#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

using value_types = nvbench::type_list<
  int8_t,
  int16_t,
  int32_t,
  int64_t,
#if _CCCL_HAS_INT128()
  int128_t,
#endif
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
  __half,
#endif
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
  __nv_bfloat16,
#endif
  float,
  double
#if _CCCL_HAS_FLOAT128()
  ,
  __float128
#endif
  >;

using algorithms =
  nvbench::enum_type_list<cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                          cub::BLOCK_REDUCE_RAKING,
                          cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                          cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cub::BlockReduceAlgorithm,
  [](cub::BlockReduceAlgorithm algorithm) {
    return cub::detail::to_string(algorithm);
  },
  [](auto) {
    return std::string{};
  })

using op_t = ::cuda::std::plus<>;

// BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC combines warp aggregates with a shared-memory atomic fetch_add
template <typename T, cub::BlockReduceAlgorithm Algorithm>
inline constexpr bool is_supported_v =
  Algorithm != cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC
  || (sizeof(T) <= 8 && ::cuda::std::is_arithmetic_v<T>);

template <int BlockSize, cub::BlockReduceAlgorithm Algorithm>
struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using BlockReduce = cub::BlockReduce<T, BlockSize, Algorithm>;
    using TempStorage = typename BlockReduce::TempStorage;
    __shared__ TempStorage temp_storage;
    const T aggregate = BlockReduce{temp_storage}.Reduce(thread_data, op_t{});
    // Reuse one TempStorage across chained calls to mimic realistic workloads. BlockReduce needs a barrier before
    // reuse.
    __syncthreads();
    return aggregate;
  }
};

template <typename T, int BlockSize, cub::BlockReduceAlgorithm Algorithm>
void run_block_reduce(nvbench::state& state)
{
  using action_t              = benchmark_op_t<BlockSize, Algorithm>;
  constexpr int unroll_factor = 128; // compromise between compile time and noise
  const auto& kernel          = benchmark_kernel<BlockSize, unroll_factor, action_t, T, false>;
  const int num_SMs     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_SM = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, BlockSize, 0));
  // NVBENCH_CUDA_CALL_NOEXCEPT swallows a failed occupancy query, which would leave
  // max_blocks_per_SM at 0 and turn the launch into a bare cudaErrorInvalidConfiguration.
  if (max_blocks_per_SM == 0)
  {
    state.skip("Skipping: no resident blocks for this type on this device.");
    return;
  }
  // Latency mode runs a single block so the chained reductions form one dependency chain, as in one DeviceReduce tile.
  const int grid_size = state.get_string("Mode") == "latency" ? 1 : max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, BlockSize>>>(action_t{});
  });
}

template <typename T, cub::BlockReduceAlgorithm Algorithm>
void block_reduce(nvbench::state& state, nvbench::type_list<T, nvbench::enum_type<Algorithm>>)
{
  if constexpr (!is_supported_v<T, Algorithm>)
  {
    state.skip("Skipping: algorithm requires atomic addition for this type.");
  }
  else
  {
    // BlockSize is a compile-time parameter of BlockReduce, so each axis value dispatches to its own instantiation.
    switch (state.get_int64("BlockSize"))
    {
      case 64:
        return run_block_reduce<T, 64, Algorithm>(state);
      case 128:
        return run_block_reduce<T, 128, Algorithm>(state);
      case 256:
        return run_block_reduce<T, 256, Algorithm>(state);
      case 512:
        return run_block_reduce<T, 512, Algorithm>(state);
      case 1024:
        return run_block_reduce<T, 1024, Algorithm>(state);
      default:
        state.skip("Skipping: unsupported block size.");
    }
  }
}

NVBENCH_BENCH_TYPES(block_reduce, NVBENCH_TYPE_AXES(value_types, algorithms))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "Algorithm{ct}"})
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(6, 10, 1))
  .add_string_axis("Mode", {"throughput", "latency"});
