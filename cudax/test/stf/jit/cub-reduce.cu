//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief JIT kernels based on CUB
 */

#include <thrust/device_vector.h>

#include <cuda/experimental/__stf/nvrtc/jit_utils.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

const char* kernel_template = R"(
#include <cuda/experimental/__stf/nvrtc/slice.cuh>
#include <cub/block/block_reduce.cuh>

extern "C"
__global__ void %KERNEL_NAME%(%s dyn_values, %s dyn_partials, size_t nelems)
{
  const int BLOCK_THREADS = %d;
  using T = int;
  %s values{dyn_values};
  %s partials{dyn_partials};

  using namespace cub;
  typedef BlockReduce<T, BLOCK_THREADS> BlockReduceT;

  auto thread_id = BLOCK_THREADS * blockIdx.x + threadIdx.x;

  // Local reduction
  T local_sum = 0;
  for (size_t ind = thread_id; ind < nelems; ind += blockDim.x * gridDim.x)
  {
    local_sum += values(ind);
  }

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Per-thread tile data
  T result = BlockReduceT(temp_storage).Sum(local_sum);

  if (threadIdx.x == 0)
  {
    partials(blockIdx.x) = result;
  }
}
)";

template <typename Ctx>
void run()
{
  Ctx ctx;

  const size_t N          = 1024 * 16;
  const size_t BLOCK_SIZE = 128;
  const size_t num_blocks = 32;

  int *X, ref_tot;

  X       = new int[N];
  ref_tot = 0;

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = rand() % N;
    ref_tot += X[ind];
  }

  auto values   = ctx.logical_data(X, {N});
  auto partials = ctx.logical_data(shape_of<slice<int>>(num_blocks));
  auto result   = ctx.logical_data(shape_of<slice<int>>(1));

  ctx.cuda_kernel_chain(values.read(), partials.write(), result.write())->*[&](auto values, auto partials, auto result) {
    jit_adapter jvalues{values};
    jit_adapter jpartials{partials};
    jit_adapter jresult{result};

    CUfunction reduce_kernel_1 = lazy_jit(
      kernel_template,
      get_nvrtc_flags(),
      "",
      jvalues.kernel_param_t_name(),
      jpartials.kernel_param_t_name(),
      BLOCK_SIZE,
      jvalues.kernel_side_t_name(),
      jpartials.kernel_side_t_name());

    CUfunction reduce_kernel_2 = lazy_jit(
      kernel_template,
      get_nvrtc_flags(),
      "",
      jpartials.kernel_param_t_name(),
      jresult.kernel_param_t_name(),
      BLOCK_SIZE,
      jpartials.kernel_side_t_name(),
      jresult.kernel_side_t_name());

    // clang-format off
      return ::std::vector<cuda_kernel_desc> {
        // reduce values into partials
        { reduce_kernel_1, num_blocks, BLOCK_SIZE, 0, jvalues.to_kernel_arg(), jpartials.to_kernel_arg(), N},
        // reduce partials on a single block into result
        { reduce_kernel_2, num_blocks, BLOCK_SIZE, 0, jpartials.to_kernel_arg(), jresult.to_kernel_arg(), num_blocks}
      };
    // clang-format on
  };

  ctx.host_launch(result.read())->*[&](auto p) {
    if (p(0) != ref_tot)
    {
      fprintf(stderr, "INCORRECT RESULT: p sum = %d, ref tot = %d\n", p(0), ref_tot);
      abort();
    }
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
