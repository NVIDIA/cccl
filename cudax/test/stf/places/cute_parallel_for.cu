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
 * @brief parallel_for over a grid driven by a cute_partition instance
 *
 * The same user-facing parallel_for entry point accepts stateful partitioners:
 * the partition decides both the kernel decomposition (per-place sub-shapes)
 * and the data placement (composite data place backed by the partition).
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

using cuda::experimental::places::dim_policy;
using cuda::experimental::places::dim_spec;
using cuda::experimental::places::make_partition;

int main()
{
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  stream_ctx ctx;

  // A grid of two places (same device when only one GPU is present)
  ::std::vector<exec_place> places;
  places.push_back(exec_place::device(0));
  places.push_back(exec_place::device(ndevs > 1 ? 1 : 0));
  auto grid = make_grid(mv(places));

  // 1-D: dimension 0 blocked over the grid
  {
    const size_t n = 1024 * 1024;
    auto lA        = ctx.logical_data(shape_of<slice<size_t>>(n));

    auto part = make_partition(dim4(n), {dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());

    ctx.parallel_for(part, grid, lA.shape(), lA.write())->*[] _CCCL_DEVICE(size_t i, auto a) {
      a(i) = 3 * i + 7;
    };

    ctx.host_launch(lA.read())->*[&](auto a) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(a(i) == 3 * i + 7);
      }
    };
  }

  // 3-D: dimension 1 blocked over the grid (the per-dimension expressiveness
  // the classic blocked_partition cannot provide)
  {
    const size_t nx = 32, ny = 64, nz = 16;
    auto lB = ctx.logical_data(shape_of<slice<size_t, 3>>(nx, ny, nz));

    auto part =
      make_partition(dim4(nx, ny, nz), {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}, dim_spec{}}, grid.get_dims());

    ctx.parallel_for(part, grid, lB.shape(), lB.write())->*[] _CCCL_DEVICE(size_t x, size_t y, size_t z, auto b) {
      b(x, y, z) = x + 100 * y + 10000 * z;
    };

    ctx.host_launch(lB.read())->*[&](auto b) {
      for (size_t x = 0; x < nx; x++)
      {
        for (size_t y = 0; y < ny; y++)
        {
          for (size_t z = 0; z < nz; z++)
          {
            EXPECT(b(x, y, z) == x + 100 * y + 10000 * z);
          }
        }
      }
    };
  }

  // The classic stateless partitioners keep working through the same entry
  {
    const size_t n = 4096;
    auto lC        = ctx.logical_data(shape_of<slice<size_t>>(n));

    ctx.parallel_for(blocked_partition(), grid, lC.shape(), lC.write())->*[] _CCCL_DEVICE(size_t i, auto c) {
      c(i) = i;
    };

    ctx.host_launch(lC.read())->*[&](auto c) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(c(i) == i);
      }
    };
  }

  ctx.finalize();

  // Uneven extents are rejected by the sub-shape derivation: the exact-cover
  // restriction of the parallel_for path must fail loudly rather than compute
  // on phantom coordinates (validated at the apply() contract level - a
  // partitioner throwing mid-task is not recoverable by design)
  {
    const size_t n = 1023; // not divisible by 2 places
    auto part      = make_partition(dim4(n), {dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());

    bool thrown = false;
    try
    {
      auto sub = part.apply(shape_of<slice<size_t>>(n), pos4(0), grid.get_dims());
      (void) sub;
    }
    catch (const ::std::invalid_argument&)
    {
      thrown = true;
    }
    EXPECT(thrown, "uneven extents must be rejected by the sub-shape derivation");
  }

  printf("cute_parallel_for: all checks passed\n");
  return 0;
}
