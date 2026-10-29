// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_scan.cuh>

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

using op_t = ::cuda::std::plus<>;

template <int BlockSize>
struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using BlockScan   = cub::BlockScan<T, BlockSize>;
    using TempStorage = typename BlockScan::TempStorage;
    __shared__ TempStorage temp_storage;
    T inclusive_output;
    BlockScan{temp_storage}.InclusiveScan(thread_data, inclusive_output, op_t{});
    // Reuse one TempStorage across chained calls to mimic realistic workloads. BlockScan needs a barrier before reuse.
    __syncthreads();
    return inclusive_output;
  }
};

template <typename T, int BlockSize>
void run_block_scan(nvbench::state& state)
{
  constexpr int unroll_factor = 128; // compromise between compile time and noise
  const auto& kernel          = benchmark_kernel<BlockSize, unroll_factor, benchmark_op_t<BlockSize>, T, false>;
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
  const int grid_size = max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, BlockSize>>>(benchmark_op_t<BlockSize>{});
  });
}

template <typename T>
void block_scan(nvbench::state& state, nvbench::type_list<T>)
{
  // BlockSize is a compile-time parameter of BlockScan, so each axis value dispatches to its own instantiation.
  switch (state.get_int64("BlockSize"))
  {
    case 64:
      return run_block_scan<T, 64>(state);
    case 128:
      return run_block_scan<T, 128>(state);
    case 256:
      return run_block_scan<T, 256>(state);
    case 512:
      return run_block_scan<T, 512>(state);
    case 1024:
      return run_block_scan<T, 1024>(state);
    default:
      state.skip("Skipping: unsupported block size.");
  }
}

NVBENCH_BENCH_TYPES(block_scan, NVBENCH_TYPE_AXES(value_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(6, 10, 1));
