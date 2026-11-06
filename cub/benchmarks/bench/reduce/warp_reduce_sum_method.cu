// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WarpReduce::Sum counterpart of warp_reduce_sum.cu, which benchmarks
// WarpReduce::Reduce(x, cuda::std::plus<>). cuda.coop's warp sum lowers to
// WarpReduce::Sum, so this is the like-for-like C++ reference for it.

#include <cub/config.cuh>

#include <cub/warp/warp_reduce.cuh>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
                     __half,
#endif
                     float,
                     double>;

using unroll_factors = nvbench::enum_type_list<32, 64, 128>;
using unroll_modes   = nvbench::enum_type_list<false, true>;

struct sum_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using WarpReduce  = cub::WarpReduce<T>;
    using TempStorage = typename WarpReduce::TempStorage;
    __shared__ TempStorage temp_storage[32];
    auto warp_id = threadIdx.x / 32;
    return WarpReduce{temp_storage[warp_id]}.Sum(thread_data);
  }
};

template <typename T, int UnrollFactor, bool Unroll>
void warp_reduce_sum_method(nvbench::state& state,
                            nvbench::type_list<T, nvbench::enum_type<UnrollFactor>, nvbench::enum_type<Unroll>>)
{
  constexpr int block_size = 256;
  const auto& kernel       = benchmark_kernel<block_size, UnrollFactor, sum_op_t, T, Unroll>;
  const int num_SMs     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_SM = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  const int grid_size = max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(sum_op_t{});
  });
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  bool,
  [](bool unroll) {
    return unroll ? "full" : "loop";
  },
  [](bool) {
    return std::string{};
  })

NVBENCH_BENCH_TYPES(warp_reduce_sum_method, NVBENCH_TYPE_AXES(value_types, unroll_factors, unroll_modes))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "UnrollFactor", "Unroll"});
