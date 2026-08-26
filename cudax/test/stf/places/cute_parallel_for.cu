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
 * The same user-facing parallel_for entry point accepts value-defined
 * partitioners: the partition decides both the kernel decomposition
 * (per-place sub-shapes) and the data placement (composite data place backed
 * by the partition).
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/localization/composite_slice.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

namespace
{
void test_cute_composite_cache(const exec_place& grid)
{
  const size_t n = 4096;
  const dim4 data_dims(n);
  const auto part        = make_partition(data_dims, partition_spec{blocked<0>}, grid.get_dims());
  const auto place       = cuda::experimental::places::make_composite_data_place(grid, part);
  const auto delinearize = [data_dims](size_t ind) {
    return data_dims.index_to_pos(ind);
  };

  reserved::composite_slice_cache cache;

  // Equal element counts are insufficient: a different tensor shape changes
  // delinearization and therefore ownership.
  const dim4 mismatched_dims(n / 2, 2);
  const auto mismatched_delinearize = [mismatched_dims](size_t ind) {
    return mismatched_dims.index_to_pos(ind);
  };
  bool mismatch_thrown = false;
  try
  {
    (void) cache.get(place, mismatched_delinearize, n, sizeof(size_t), mismatched_dims);
  }
  catch (const ::std::invalid_argument&)
  {
    mismatch_thrown = true;
  }
  EXPECT(mismatch_thrown);

  auto [first, first_prereqs] = cache.get(place, delinearize, n, sizeof(size_t), data_dims);
  EXPECT(first_prereqs.empty());
  const auto first_base = first->get_base_ptr();

  cache.put(place, mv(first), first_prereqs, n, sizeof(size_t), data_dims);

  // A separately constructed but equivalent place must find the same cached
  // VMM allocation through the value-keyed CuTe pool.
  const auto equivalent_part    = make_partition(data_dims, partition_spec{blocked<0>}, grid.get_dims());
  const auto equivalent_place   = cuda::experimental::places::make_composite_data_place(grid, equivalent_part);
  auto [second, second_prereqs] = cache.get(equivalent_place, delinearize, n, sizeof(size_t), data_dims);
  EXPECT(second_prereqs.empty());
  EXPECT(second->get_base_ptr() == first_base);

  cache.put(equivalent_place, mv(second), second_prereqs, n, sizeof(size_t), data_dims);
  EXPECT(cache.deinit().empty());
}

void test_static_codegen_parity(stream_ctx& ctx, const exec_place& grid)
{
  const size_t nx   = 64;
  const size_t ny   = 32;
  auto typed_data   = ctx.logical_data(shape_of<slice<size_t, 2>>(nx, ny));
  auto classic_data = ctx.logical_data(shape_of<slice<size_t, 2>>(nx, ny));
  const auto part   = make_partition(dim4(nx, ny), partition_spec{whole, blocked<0>}, grid.get_dims());
  auto write        = [] _CCCL_DEVICE(size_t x, size_t y, auto values) {
    values(x, y) = x + 100 * y;
  };

  ctx.parallel_for(part, grid, typed_data.shape(), typed_data.write())->*decltype(write)(write);
  ctx.parallel_for(blocked_partition(), grid, classic_data.shape(), classic_data.write())->*decltype(write)(write);

  ctx.host_launch(typed_data.read(), classic_data.read())->*[=](auto typed, auto classic) {
    for (size_t y = 0; y < ny; y++)
    {
      for (size_t x = 0; x < nx; x++)
      {
        EXPECT(typed(x, y) == x + 100 * y);
        EXPECT(classic(x, y) == typed(x, y));
      }
    }
  };
}

void test_cute_graph_backend(const exec_place& grid)
{
  const size_t n = 1023;
  graph_ctx ctx;
  auto data       = ctx.logical_data(shape_of<slice<size_t>>(n));
  const auto part = make_partition(dim4(n), partition_spec{blocked<0>}, grid.get_dims());

  ctx.parallel_for(part, grid, data.shape(), data.write())->*[] _CCCL_DEVICE(size_t i, auto values) {
    values(i) = 5 * i + 3;
  };
  ctx.host_launch(data.read())->*[=](auto values) {
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(values(i) == 5 * i + 3);
    }
  };
  ctx.finalize();
}
} // namespace

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

  test_cute_composite_cache(grid);
  test_static_codegen_parity(ctx, grid);

  // 1-D: dimension 0 blocked over the grid
  {
    const size_t n = 1024 * 1024;
    auto lA        = ctx.logical_data(shape_of<slice<size_t>>(n));

    auto part = make_partition(dim4(n), partition_spec{blocked<0>}, grid.get_dims());

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

    auto part = make_partition(dim4(nx, ny, nz), partition_spec{whole, blocked<0>, whole}, grid.get_dims());

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

    auto part = make_partition(dim4(n), partition_spec{blocked<0>}, grid.get_dims());

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

    auto part = make_partition(dim4(nx, ny), partition_spec{whole, blocked<0>}, grid.get_dims());

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
  // constructed equal partitions producing the same composite identity -
  // guarded here.
  {
    const size_t nx = 64, ny = 32;
    auto lF = ctx.logical_data(shape_of<slice<size_t, 2>>(nx, ny));

    auto part                  = make_partition(dim4(nx, ny), partition_spec{whole, blocked<0>}, grid.get_dims());
    auto dist                  = cuda::experimental::places::make_composite_data_place(grid, part);
    const auto equivalent_part = make_partition(dim4(nx, ny), partition_spec{whole, blocked<0>}, grid.get_dims());

    EXPECT(dist == cuda::experimental::places::make_composite_data_place(grid, equivalent_part),
           "cute composites from equal partitions must compare equal");

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
  test_cute_graph_backend(grid);

  printf("cute_parallel_for: all checks passed\n");
  return 0;
}
