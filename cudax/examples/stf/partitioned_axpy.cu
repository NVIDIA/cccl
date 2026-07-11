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
 * JAX-like partition specification
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
 * Runs on a single GPU (the grid then holds the same device twice); with
 * several GPUs each place is a distinct device.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t i = tid; i < x.size(); i += nthreads)
  {
    y(i) += a * x(i);
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
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  CUdevice dev0;
  cuda_safe_call(cuInit(0));
  cuda_safe_call(cuDeviceGet(&dev0, 0));
  int supports_vmm;
  cuda_safe_call(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev0));
  if (!supports_vmm)
  {
    fprintf(stderr, "VMM not supported on this machine, skipping example.\n");
    return 0;
  }

  // The single task below touches the whole range from each device, which
  // requires peer access between all participating devices; fall back to one
  // device when it is unavailable.
  int usable_devs = ::std::min(ndevs, 4);
  for (int a = 0; a < usable_devs; a++)
  {
    for (int b = 0; b < usable_devs; b++)
    {
      int can_access = 1;
      if (a != b)
      {
        cuda_safe_call(cudaDeviceCanAccessPeer(&can_access, a, b));
      }
      if (!can_access)
      {
        fprintf(stderr, "Peer access unavailable between devices %d and %d: using a single device.\n", a, b);
        usable_devs = 1;
      }
    }
  }

  // A grid of up to 4 places (at least 2 so the partitioning is visible even
  // on a single-GPU machine, where places then share device 0)
  const size_t nplaces = ::std::max(2, usable_devs);
  ::std::vector<exec_place> places;
  for (size_t i = 0; i < nplaces; i++)
  {
    places.push_back(exec_place::device(static_cast<int>(i % usable_devs)));
  }
  auto all_devs = make_grid(mv(places));

  const size_t N = 4 * 1024 * 1024;

  // "Dimension 0, blocked over grid axis 0" - the JAX-like specification
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

  ctx.task(all_devs, lX.read(dist), lY.rw(dist))->*[&](cudaStream_t s, auto dX, auto dY) {
    axpy<<<128, 128, 0, s>>>(alpha, dX, dY);
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
