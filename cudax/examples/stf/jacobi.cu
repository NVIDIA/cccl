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
 * @brief Jacobi method with launch
 *
 */

#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

/* Implement atomicMax with a compare and swap */
_CCCL_DEVICE double atomicMax(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old             = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

template <typename thread_hierarchy_t>
_CCCL_DEVICE double reduce_max(thread_hierarchy_t& t, double local_max)
{
  auto ti             = t.inner();
  slice<double> error = t.template storage<double>(0);

  error(0) = 0.0;
  t.sync();

  // Note we do not use t.static_width(1) because t is a runtime variable so it
  // cannot be used directly to statically evaluate the size.
  __shared__ double block_max[thread_hierarchy_t::static_width(1)];
  block_max[ti.rank()] = local_max;
  for (size_t s = ti.size() / 2; s > 0; s /= 2)
  {
    if (ti.rank() < s)
    {
      block_max[ti.rank()] = fmax(block_max[ti.rank() + s], block_max[ti.rank()]);
    }
    ti.sync();
  }

  if (ti.rank() == 0)
  {
    atomicMax(&error(0), block_max[0]);
  }
  t.sync();

  return error(0);
}

int main(int argc, char** argv)
{
  context ctx;

  size_t n        = 4096;
  size_t m        = 4096;
  size_t iter_max = 100;
  double tol      = 0.0000001;

  if (argc > 2)
  {
    n = atol(argv[1]);
    m = atol(argv[2]);
  }

  if (argc > 3)
  {
    iter_max = atoi(argv[3]);
  }

  if (argc > 4)
  {
    tol = atof(argv[4]);
  }

  auto lA    = ctx.logical_data(shape_of<slice<double, 2>>(m, n));
  auto lAnew = ctx.logical_data(lA.shape());

  auto all_devs = exec_place::all_devices();

  ctx.parallel_for(blocked_partition(), all_devs, lA.shape(), lA.write(), lAnew.write()).set_symbol("init")->*
    [=] _CCCL_DEVICE(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = (i == j) ? 10.0 : -1.0;
    };

  cudaEvent_t start, stop;

  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  auto spec = con(con<64>(), mem(sizeof(double)));

  ctx.launch(spec, all_devs, lA.rw(), lAnew.write())->*[iter_max, tol, n, m] _CCCL_DEVICE(auto t, auto A, auto Anew) {
    auto ti = t.inner();
    for (size_t iter = 0; iter < iter_max; iter++)
    {
      // thread-local maximum error
      double local_error = 0.0;

      for (auto [i, j] : t.apply_partition(inner<1>(shape(A))))
      {
        Anew(i, j) = 0.25 * (A(i - 1, j) + A(i + 1, j) + A(i, j - 1) + A(i, j + 1));

        local_error = fmax(local_error, fabs(A(i, j) - Anew(i, j)));
      }

      // compute the overall maximum error
      double error = reduce_max(t, local_error);

      /* Fill A with the new values */
      for (auto [i, j] : t.apply_partition(shape(A)))
      {
        A(i, j) = Anew(i, j);
      }

      if (iter % 25 == 0 && t.rank() == 0)
      {
        printf("iter %zu : error %e (tol %e)\n", iter, error, tol);
      }
    }
  };

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time: %f ms\n", elapsedTime);
}
