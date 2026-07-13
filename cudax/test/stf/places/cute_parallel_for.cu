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

  // Uneven extents: the padding phantoms are excluded by the sub-shape's
  // predicate (CuTe predication), so odd sizes work end to end
  {
    const size_t n = 1023; // not divisible by 2 places
    auto lD        = ctx.logical_data(shape_of<slice<size_t>>(n));

    auto part = make_partition(dim4(n), {dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());

    ctx.parallel_for(part, grid, lD.shape(), lD.write())->*[] _CCCL_DEVICE(size_t i, auto d) {
      d(i) = 2 * i + 1;
    };

    ctx.host_launch(lD.read())->*[&](auto d) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(d(i) == 2 * i + 1);
      }
    };
  }

  // Interior region: the box is a region within the tensor the partition was
  // built for; each place computes its owned coordinates restricted to the
  // box, and the boundary stays untouched
  {
    const size_t nx = 64, ny = 32;
    auto lE = ctx.logical_data(shape_of<slice<size_t, 2>>(nx, ny));

    auto part = make_partition(dim4(nx, ny), {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());

    ctx.parallel_for(part, grid, lE.shape(), lE.write())->*[] _CCCL_DEVICE(size_t x, size_t y, auto e) {
      e(x, y) = 7;
    };

    box interior({1ul, nx - 1}, {1ul, ny - 1});
    ctx.parallel_for(part, grid, interior, lE.rw())->*[] _CCCL_DEVICE(size_t x, size_t y, auto e) {
      e(x, y) = 100 + x + y;
    };

    ctx.host_launch(lE.read())->*[&](auto e) {
      for (size_t x = 0; x < nx; x++)
      {
        for (size_t y = 0; y < ny; y++)
        {
          const bool inside = (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1);
          EXPECT(e(x, y) == (inside ? 100 + x + y : 7));
        }
      }
    };
  }

  // Boundary-style thin regions: iterate the face with a classic scale-free
  // partitioner (tight, no discarded lanes) while keeping placement on the
  // cute composite through explicit deps. This relies on separately
  // constructed cute composites comparing equal (same instance identity, no
  // duplication) - guarded here.
  {
    const size_t nx = 64, ny = 32;
    auto lF = ctx.logical_data(shape_of<slice<size_t, 2>>(nx, ny));

    auto part = make_partition(dim4(nx, ny), {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());
    auto dist = cuda::experimental::places::make_composite_data_place(grid, part);

    EXPECT(dist == cuda::experimental::places::make_composite_data_place(grid, part),
           "cute composites from the same partition must compare equal");

    // Volumetric pass placed and decomposed by the partition
    ctx.parallel_for(part, grid, lF.shape(), lF.write())->*[] _CCCL_DEVICE(size_t x, size_t y, auto f) {
      f(x, y) = 1;
    };

    // Face update: classic iteration over the thin box, same placement
    box face({0ul, nx}, {0ul, 1ul});
    ctx.parallel_for(blocked_partition(), grid, face, lF.rw(dist))->*[] _CCCL_DEVICE(size_t x, size_t y, auto f) {
      f(x, y) = 42;
    };

    ctx.host_launch(lF.read())->*[&](auto f) {
      for (size_t x = 0; x < nx; x++)
      {
        for (size_t y = 0; y < ny; y++)
        {
          EXPECT(f(x, y) == (y == 0 ? 42 : 1));
        }
      }
    };
  }

  ctx.finalize();

  printf("cute_parallel_for: all checks passed\n");
  return 0;
}
