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
#include <cuda/experimental/__stf/utility/pretty_print.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void stencil2D_kernel(slice<T, 2> sUn, slice<const T, 2> sUn1)
{
  size_t N = sUn.extent(0);
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x)
  {
    for (size_t j = 0; j < N; j++)
    {
      sUn(j, i) = 0.8 * sUn1(j, i) + 0.05 * sUn1(j, (i + 1) % N) + 0.05 * sUn1(j, (i - 1 + N) % N)
                + 0.05 * sUn1((j + 1) % N, i) + 0.05 * sUn1((j - 1 + N) % N, i);
    }
  }
}

int main(int argc, char** argv)
{
  stream_ctx ctx;

  size_t NITER  = 500;
  size_t N      = 1000;
  bool vtk_dump = false;

  if (argc > 1)
  {
    NITER = atoi(argv[1]);
  }

  if (argc > 2)
  {
    N = atoi(argv[2]);
  }

  if (argc > 3)
  {
    int val  = atoi(argv[3]);
    vtk_dump = (val == 1);
  }

  size_t TOTAL_SIZE = N * N;

  double* Un  = new double[TOTAL_SIZE];
  double* Un1 = new double[TOTAL_SIZE];

  for (size_t idx = 0; idx < TOTAL_SIZE; idx++)
  {
    Un[idx]  = (idx == 0) ? 1.0 : 0.0;
    Un1[idx] = Un[idx];
  }

  auto lUn  = ctx.logical_data(make_slice(Un, std::tuple{N, N}, N));
  auto lUn1 = ctx.logical_data(make_slice(Un1, std::tuple{N, N}, N));

  //    std::shared_ptr<execution_grid> all_devs = exec_place::all_devices();
  // use grid [ 0 0 0 0 ] for debugging purpose
  auto all_devs = exec_place::repeat(exec_place::device(0), 4);

  // Partition over the vector of processor along the y-axis of the data domain
  // TODO implement the proper tiled_partitioning along y !
  data_place cdp = data_place::composite(tiled_partition<128>(), all_devs);

  for (size_t iter = 0; iter < NITER; iter++)
  {
    // UPDATE Un from Un1
    ctx.task(lUn.rw(cdp), lUn1.read(cdp))->*[&](auto stream, auto sUn, auto sUn1) {
      stencil2D_kernel<double><<<32, 128, 0, stream>>>(sUn, sUn1);
    };

    // We make sure that the total sum of elements remains constant
    if (iter % 250 == 0)
    {
      double sum = 0.0;

      ctx.task(exec_place::host, lUn.read())->*[&](auto stream, auto sUn) {
        cuda_safe_call(cudaStreamSynchronize(stream));
        for (size_t j = 0; j < N; j++)
        {
          for (size_t i = 0; i < N; i++)
          {
            sum += sUn(j, i);
          }
        }

        if (vtk_dump)
        {
          char str[32];
          snprintf(str, 32, "Un_%05zu.vtk", iter);
          mdspan_to_vtk(sUn, std::string(str));
        }
      };

      // fprintf(stderr, "iter %d : CHECK SUM = %e\n", iter, sum);
    }

    std::swap(lUn, lUn1);
  }

  ctx.finalize();
}
