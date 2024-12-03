//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief A naive parallel histogram algorithm written with launch
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__host__ __device__ double X0(int i)
{
  return sin((double) i);
}

int main(int argc, char** argv)
{
  stream_ctx ctx;

  double lower_level          = -1.0;
  double upper_level          = 1.0;
  constexpr size_t num_levels = 21;

  size_t N = 128 * 1024UL;
  if (argc > 1)
  {
    N = size_t(atoll(argv[1]));
  }

  int check = 1;
  if (argc > 2)
  {
    check = atoi(argv[2]);
  }

  // fprintf(stderr, "SIZE %s\n", pretty_print_bytes(N * sizeof(double)).c_str());

  std::vector<double> X(N);
  std::vector<size_t> histo(num_levels - 1);

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  // If we were to register each part one by one, there could be pages which
  // cross multiple parts, and the pinning operation would fail.
  cuda_safe_call(cudaHostRegister(&X[0], N * sizeof(double), cudaHostRegisterPortable));

  auto lX = ctx.logical_data(&X[0], N);
  lX.set_symbol("X");

  auto lhisto = ctx.logical_data(&histo[0], num_levels - 1);
  lhisto.set_symbol("histogram");

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));
  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  constexpr size_t BLOCK_THREADS = 128;

  // size_t NDEVS = 1;
  // auto where = exec_place::repeat(exec_place::current_device(), NDEVS);
  auto where = exec_place::current_device();

  auto spec = con<8>(con(BLOCK_THREADS, mem((num_levels - 1) * sizeof(size_t))));

  ctx.launch(spec, where, lX.read(), lhisto.write())->*[=] _CCCL_DEVICE(auto th, auto x, auto histo) {
    size_t block_id = th.rank(0);

    slice<size_t> smem_hist = th.template storage<size_t>(1);
    assert(smem_hist.size() == (num_levels - 1));

    /* Thread local histogram */
    size_t local_hist[num_levels - 1];
    for (size_t k = 0; k < num_levels - 1; k++)
    {
      local_hist[k] = 0;
      smem_hist[k]  = 0;
    }

    if (th.rank() == 0)
    {
      for (size_t k = 0; k < num_levels - 1; k++)
      {
        histo[k] = 0;
      }
    }

    for (size_t i = th.rank(); i < x.size(); i += th.size())
    {
      double xi = x(i);
      if (xi >= lower_level && xi < upper_level)
      {
        size_t bin = size_t(((num_levels - 1) * (xi - lower_level)) / (upper_level - lower_level));
        local_hist[bin]++;
      }
    }

    // smem was zero'ed
    th.inner().sync();

    /* Each thread contributes to an histogram in shared memory */
    for (size_t k = 0; k < num_levels - 1; k++)
    {
      atomicAdd((unsigned long long*) &smem_hist[k], local_hist[k]);
    }

    // histo was zero'ed
    th.sync();

    if (th.inner().rank() == 0)
    {
      for (size_t k = 0; k < num_levels - 1; k++)
      {
        atomicAdd((unsigned long long*) &histo[k], smem_hist[k]);
      }
    }
  };

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  float ms = 0;
  cuda_safe_call(cudaEventElapsedTime(&ms, start, stop));

  // fprintf(stdout, "%zu %f ms\n", N / 1024 / 1024, ms);

  if (check)
  {
    // fprintf(stderr, "Checking result...\n");
    size_t refhist[num_levels - 1];
    for (size_t i = 0; i < num_levels - 1; i++)
    {
      refhist[i] = 0;
    }

    for (size_t i = 0; i < N; i++)
    {
      double xi = X[i];
      if (xi >= lower_level && xi < upper_level)
      {
        size_t bin = size_t(((num_levels - 1) * (xi - lower_level)) / (upper_level - lower_level));
        refhist[bin]++;
      }
    }

    // double dlevel = (upper_level - lower_level) / (num_levels - 1);
    for (size_t i = 0; i < num_levels - 1; i++)
    {
      EXPECT(refhist[i] == histo[i]);
      // fprintf(stderr, "[%lf:%lf[ %ld\n", lower_level + i * dlevel, lower_level + (i + 1) * dlevel, histo[i]);
    }
  }
}
