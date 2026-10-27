//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/bit>
#include <cuda/std/cmath>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>

#include <nvbench/nvbench.cuh>

struct benchmark_op_t
{
  __device__ __forceinline__ double operator()(double value) const
  {
    // storage is used to prevent code elimination by the compiler (adapted from isnan benchmark)
    // the rotation moves a mantissa bit into the sign bit, so the next input can be negative again
    // __fp_get_storage and __fp_from_storage are no op
    const auto storage = cuda::std::__fp_get_storage(value);
    const auto result  = cuda::std::__fp_get_storage(cuda::std::fabs(value));
    return cuda::std::__fp_from_storage<double>(result ^ cuda::std::rotr(storage, 7));
  }
};

void fabs_bench(nvbench::state& state)
{
  constexpr int block_size    = 256;
  constexpr int unroll_factor = 1024;
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, benchmark_op_t, double, false>;
  const int num_SMs     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_SM = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  const int grid_size = max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(benchmark_op_t{});
  });
}

NVBENCH_BENCH(fabs_bench).set_name("base");
