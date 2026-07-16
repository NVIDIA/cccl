//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief AXPY over data distributed across the machine's devices with a
 * structured partition specification
 *
 * The partition ("dimension 0, blocked over the grid of devices") is
 * expressed once as a cute_partition. The same description is then used to:
 *
 * 1. EVALUATE the placement before committing any memory
 *    (evaluate_localized_placement: bytes per place, placement accuracy);
 * 2. back a logical data with a composite data place, so STF tasks operate
 *    on memory whose pages physically live on the device that owns them;
 * 3. perform a raw geometry-aware allocation (allocate_nd(data_dims, elemsize))
 *    outside of any STF context.
 *
 * Each place computes its own blocked portion (the idiomatic grid-task
 * pattern), so no cross-device access is required; peer/mempool access setup
 * is handled by the places machinery itself.
 */

#include <cuda/experimental/stf.cuh>

#include <cmath>
#include <cstdio>

using namespace cuda::experimental::stf;

__global__ void axpy(size_t start, size_t cnt, double a, const double* x, double* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t i = tid; i < cnt; i += nthreads)
  {
    y[start + i] += a * x[start + i];
  }
}

double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

int main()
{
  // The places machinery enumerates the devices and sets up peer/mempool
  // access between them; on a single-GPU machine this is one place.
  auto all_devs        = exec_place::all_devices();
  const size_t nplaces = all_devs.get_dims().size();

  const size_t N = 4 * 1024 * 1024;

  // "Dimension 0, blocked over grid axis 0" - the per-dimension specification
  auto part = ::cuda::experimental::places::make_partition(
    dim4(N), {::cuda::experimental::places::dim_spec{dim_policy::blocked, 0, 0}}, all_devs.get_dims());

  // 1. Score the mapping before allocating anything
  auto stats = evaluate_localized_placement(all_devs, part, sizeof(double));
  printf("Placement over %zu place(s): %zu blocks in %zu allocations, accuracy %.1f%%\n",
         nplaces,
         stats.nblocks,
         stats.nallocs,
         100.0 * stats.accuracy());
  for (const auto& entry : stats.bytes_per_place)
  {
    printf("  %s: %.2f MB\n", entry.first.c_str(), entry.second / (1024.0 * 1024.0));
  }

  // 2. Run STF tasks over logical data placed by the same policy
  stream_ctx ctx;

  ::std::vector<double> X(N), Y(N);
  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  auto lX = ctx.logical_data(&X[0], {N});
  auto lY = ctx.logical_data(&Y[0], {N});

  const double alpha = 3.14;

  // The composite data place distributes instances across the grid with the
  // classic blocked partitioner (the callback form of the same policy)
  auto dist = data_place::composite(blocked_partition_custom<0>{}, all_devs);

  // One task over the grid; each place computes its own blocked chunk
  auto t = ctx.task(all_devs, lX.read(dist), lY.rw(dist));
  t->*[&](auto, auto dX, auto dY) {
    const size_t chunk = (N + nplaces - 1) / nplaces;
    for (size_t i = 0; i < nplaces; i++)
    {
      const size_t start = i * chunk;
      if (start >= N)
      {
        // With ceil-division chunks, trailing places may have no work
        continue;
      }
      const size_t cnt = ::std::min(chunk, N - start);
      auto active      = t.activate_place(i);
      axpy<<<128, 128, 0, t.get_stream(i)>>>(start, cnt, alpha, dX.data_handle(), dY.data_handle());
    }
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    if (fabs(Y[i] - (Y0(i) + alpha * X0(i))) > 0.0001)
    {
      fprintf(stderr, "Verification FAILED at %zu\n", i);
      return 1;
    }
  }
  printf("STF task over composite-placed data: verified\n");

  // 3. Raw geometry-aware allocation, no STF context involved
  auto dp     = ::cuda::experimental::places::make_composite_data_place(all_devs, part);
  void* raw   = dp.allocate_nd(dim4(N), sizeof(double));
  auto* d_buf = static_cast<double*>(raw);
  cuda_safe_call(cudaMemset(d_buf, 0, N * sizeof(double)));
  cuda_safe_call(cudaDeviceSynchronize());
  dp.deallocate(raw, N * sizeof(double));
  printf("Raw shaped allocation on the partitioned place: OK\n");

  return 0;
}
