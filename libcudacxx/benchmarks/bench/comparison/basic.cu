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
#include <cuda/std/cmath>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>

#include <nvbench/nvbench.cuh>

struct double_pair
{
  double x{};
  double y{};
};

struct isless_op
{
  __device__ __forceinline__ bool operator()(double x, double y) const
  {
    return cuda::std::isless(x, y);
  }
};

struct islessequal_op
{
  __device__ __forceinline__ bool operator()(double x, double y) const
  {
    return cuda::std::islessequal(x, y);
  }
};

struct isgreater_op
{
  __device__ __forceinline__ bool operator()(double x, double y) const
  {
    return cuda::std::isgreater(x, y);
  }
};

struct isgreaterequal_op
{
  __device__ __forceinline__ bool operator()(double x, double y) const
  {
    return cuda::std::isgreaterequal(x, y);
  }
};

struct islessgreater_op
{
  __device__ __forceinline__ bool operator()(double x, double y) const
  {
    return cuda::std::islessgreater(x, y);
  }
};

template <typename CompareOp>
struct benchmark_op_t
{
  __device__ __forceinline__ double_pair operator()(double_pair value) const
  {
    // both operands change every iteration, so the compiler cannot hoist work on a constant operand
    // storage is used to prevent code elimination by the compiler (same approach as in isnan)
    // __fp_get_storage and __fp_from_storage are no op
    const auto storage = cuda::std::__fp_get_storage(value.x);
    return {value.y, cuda::std::__fp_from_storage<double>(storage + CompareOp{}(value.x, value.y))};
  }
};

template <typename CompareOp>
void comparison_bench(nvbench::state& state, nvbench::type_list<CompareOp>)
{
  constexpr int block_size    = 256;
  constexpr int unroll_factor = 1024;
  const auto& kernel    = benchmark_kernel<block_size, unroll_factor, benchmark_op_t<CompareOp>, double_pair, false>;
  const int num_SMs     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_SM = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  const int grid_size = max_blocks_per_SM * num_SMs;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(benchmark_op_t<CompareOp>{});
  });
}

NVBENCH_DECLARE_TYPE_STRINGS(isless_op, "isless", "");
NVBENCH_DECLARE_TYPE_STRINGS(islessequal_op, "islessequal", "");
NVBENCH_DECLARE_TYPE_STRINGS(isgreater_op, "isgreater", "");
NVBENCH_DECLARE_TYPE_STRINGS(isgreaterequal_op, "isgreaterequal", "");
NVBENCH_DECLARE_TYPE_STRINGS(islessgreater_op, "islessgreater", "");

using ops = nvbench::type_list<isless_op, islessequal_op, isgreater_op, isgreaterequal_op, islessgreater_op>;

NVBENCH_BENCH_TYPES(comparison_bench, NVBENCH_TYPE_AXES(ops)).set_name("base").set_type_axes_names({"op"});
