//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void stencil_kernel(slice<T> Un, slice<const T> Un1)
{
  size_t N = Un.extent(0);
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x)
  {
    Un(i) = 0.9 * Un1(i) + 0.05 * Un1((i + N - 1) % N) + 0.05 * Un1((i + 1) % N);
  }
}

int main(int argc, char** argv)
{
  stream_ctx ctx;

  int NITER               = 500;
  int NBLOCKS             = 20;
  const size_t BLOCK_SIZE = 2048 * 1024;

  if (argc > 1)
  {
    NITER = atoi(argv[1]);
  }

  if (argc > 2)
  {
    NBLOCKS = atoi(argv[2]);
  }

  const size_t TOTAL_SIZE = NBLOCKS * BLOCK_SIZE;

  double* Un  = new double[TOTAL_SIZE];
  double* Un1 = new double[TOTAL_SIZE];

  for (size_t idx = 0; idx < TOTAL_SIZE; idx++)
  {
    Un[idx]  = (idx == 0) ? 1.0 : 0.0;
    Un1[idx] = Un[idx];
  }

  auto lUn  = ctx.logical_data(make_slice(Un, TOTAL_SIZE));
  auto lUn1 = ctx.logical_data(make_slice(Un1, TOTAL_SIZE));

  //    std::shared_ptr<execution_grid> all_devs = exec_place::all_devices();
  // use grid [ 0 0 0 0 ] for debugging purpose
  auto all_devs = exec_place::repeat(exec_place::device(0), 4);

  data_place cdp = data_place::composite(tiled_partition<BLOCK_SIZE>(), all_devs);

  for (int iter = 0; iter < NITER; iter++)
  {
    // UPDATE Un from Un1
    ctx.task(lUn.rw(cdp), lUn1.read(cdp))->*[&](auto stream, auto sUn, auto sUn1) {
      stencil_kernel<double><<<32, 128, 0, stream>>>(sUn, sUn1);
    };

    // We make sure that the total sum of elements remains constant
    if (iter % 250 == 0)
    {
      double sum = 0.0;

      ctx.task(exec_place::host, lUn.read())->*[&](auto stream, auto sUn) {
        cuda_safe_call(cudaStreamSynchronize(stream));
        for (size_t offset = 0; offset < TOTAL_SIZE; offset++)
        {
          sum += sUn(offset);
        }
      };

      // TODO add an assertion to check whether sum is close enough to 1.0
      // fprintf(stderr, "iter %d : CHECK SUM = %e\n", iter, sum);
    }

    std::swap(lUn, lUn1);
  }

  ctx.finalize();
}
