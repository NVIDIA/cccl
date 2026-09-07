//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief locality_domain_helper driven place construction
 *
 * Mirrors the green_context_helper usage pattern: enumerate the domains of
 * each device with the helper, build places from the views it hands out, and
 * run one task per view to check the pattern end to end.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void add_one(slice<double> x)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  size_t n = x.extent(0);
  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    x(ind) += 1.0;
  }
}

int main()
{
  int ndevs = 0;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  // Helper-driven enumeration over every device, adapting to each count
  std::vector<locality_domain_helper> helpers;
  helpers.reserve(ndevs);
  size_t total_domains = 0;
  for (int d = 0; d < ndevs; d++)
  {
    helpers.emplace_back(d);
    total_domains += helpers.back().get_count();
  }

  // Never 0: every device reports at least one whole-device domain.
  EXPECT(total_domains >= static_cast<size_t>(ndevs));

  // View-based factories agree with the ordinal-based ones
  for (const auto& helper : helpers)
  {
    for (size_t i = 0; i < helper.get_count(); i++)
    {
      const locality_domain_view view = helper.get_view(i);
      EXPECT(view.devid == helper.get_device_id());

      EXPECT(exec_place::locality_domain(view) == exec_place::locality_domain(view.devid, view.domain_id));
      EXPECT(data_place::locality_domain(view) == data_place::locality_domain(view.devid, view.domain_id));
    }
  }

  // One task per view, everywhere, then check the result
  stream_ctx ctx;
  const int n = 1024;

  double X[n];
  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 42.0 + ind;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));

  for (const auto& helper : helpers)
  {
    for (size_t i = 0; i < helper.get_count(); i++)
    {
      ctx.task(exec_place::locality_domain(helper.get_view(i)), handle_X.rw())->*[](cudaStream_t s, auto dX) {
        add_one<<<16, 128, 0, s>>>(dX);
      };
    }
  }

  ctx.host_launch(handle_X.read())->*[&](auto hX) {
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(hX(ind) - (42.0 + ind + static_cast<double>(total_domains))) < 0.00001);
    }
  };

  ctx.finalize();

  return 0;
}
