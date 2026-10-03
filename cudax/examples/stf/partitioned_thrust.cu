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
 * @brief A grid task over composite-placed logical data that calls library
 *        algorithms (Thrust, sharded verbs) instead of a hand-written kernel.
 *
 * `partitioned_axpy.cu` shows the raw form: the task body recomputes the
 * per-place chunk arithmetic, activates each place and launches a kernel on
 * each grid stream. Here the body asks for the SHARDED VIEW of each argument
 * (`sharded::task_view`) — one shard per place at the cut the composite data
 * place applied, no user-supplied geometry — and for the task's per-place
 * ENVIRONMENTS (`sharded::task_envs`, built lazily from the task's own
 * streams). It then runs:
 *
 *   1. Thrust per shard, on that place's stream        (Y = alpha X + Y)
 *   2. a sharded verb over the views                   (Y = Y + X)
 *   3. a sharded reduction                             (sum of Y)
 *
 * STF owns the bracketing: every grid stream is forked from the task's
 * dependencies and joined at the end, so the body adds no synchronization.
 * The call environment is the task's stream, which makes the results
 * ordered on it. CUB temporaries inside the verbs come from each shard
 * environment's memory resource (the place's), not from the caller.
 */

#include <cuda/experimental/__sharded/adapters/task_view.cuh>
#include <cuda/experimental/sharded.cuh>
#include <cuda/experimental/stf.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdio>

using namespace cuda::experimental::stf;
namespace sh = cuda::experimental::sharded;

double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

struct saxpy_functor
{
  double alpha;
  __host__ __device__ double operator()(double x, double y) const
  {
    return alpha * x + y;
  }
};

struct add_op
{
  __host__ __device__ double operator()(double y, double x) const
  {
    return y + x;
  }
};

int main()
{
  // The grid of places the task runs on: the device's locality domains (one
  // place on a device without them, several on a multi-die part).
  auto grid      = exec_place::all_locality_domains();
  const size_t P = grid.size();

  const size_t N     = 4 * 1024 * 1024;
  const double alpha = 3.14;

  stream_ctx ctx;

  // Logical data from shapes: no host buffers, the instances are born on the
  // composite place below.
  auto lX = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("X");
  auto lY = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("Y");

  // The composite data place: blocked over the grid. This is the only place
  // the geometry is stated; the task body reads it back from the argument.
  auto dist = data_place::composite(blocked_partition(), grid);

  // Initialize on the grid: each place fills its own part of X and Y.
  ctx.parallel_for(blocked_partition(), grid, lX.shape(), lX.write(dist), lY.write(dist))
      ->*[] _CCCL_DEVICE(size_t i, auto x, auto y) {
            x(i) = sin((double) i);
            y(i) = cos((double) i);
          };

  double sum = 0.0;

  auto t = ctx.task(grid, lX.read(dist), lY.rw(dist));
  t.set_symbol("saxpy + add + sum");
  t->*[&](auto, auto dX, auto dY) {
    // The sharded view of argument 0 and 1: one shard per place, at the cut
    // `dist` applied, executed by that place on the task's stream for it.
    auto vX = sh::task_view(t, 0, dX);
    auto vY = sh::task_view(t, 1, dY);

    for (size_t g = 0; g < vY.num_shards(); g++)
    {
      const auto& s = vY.shard(g);
      printf("  shard %zu: [%zu, %zu) on %s\n", g, s.global_offset, s.global_offset + s.size, s.place.to_string().c_str());
    }

    // Per-place environments, built on access from the task's streams.
    auto envs           = sh::task_envs(t, vY);
    const auto call_env = ::cuda::stream_ref{t.get_stream()};

    // 1. Thrust, one call per shard on that shard's stream: Y = alpha X + Y
    sh::for_each_shard(
      vY,
      envs,
      [&](size_t g, const auto& y, cudaStream_t s) {
        const auto& x = vX.shard(g);
        thrust::transform(thrust::cuda::par_nosync.on(s), x.data, x.data + x.size, y.data, y.data, saxpy_functor{alpha});
      },
      call_env);

    // 2. A sharded verb over the same views: Y = Y + X (stream-ordered after 1)
    sh::zip_transform(vY, envs, add_op{}, call_env, vY, vX);

    // 3. A sharded reduction: per-shard CUB reductions (temporaries from each
    //    shard's place) combined across places. Synchronous by contract.
    sum = sh::sum(vY, envs, call_env);
  };

  // Verify in a host callback: STF brings Y to the host after the task.
  bool ok = true;
  ctx.host_launch(lY.read())->*[&](auto hY) {
    double sum_ref = 0.0;
    for (size_t i = 0; i < N; i++)
    {
      const double expected = Y0(i) + (alpha + 1.0) * X0(i);
      if (fabs(hY(i) - expected) > 1e-9)
      {
        fprintf(stderr, "Verification FAILED at %zu: %f vs %f\n", i, hY(i), expected);
        ok = false;
        return;
      }
      sum_ref += expected;
    }
    if (fabs(sum - sum_ref) > 1e-6 * fabs(sum_ref))
    {
      fprintf(stderr, "Sum FAILED: %f vs %f\n", sum, sum_ref);
      ok = false;
    }
  };

  ctx.finalize();

  if (!ok)
  {
    return 1;
  }

  printf("partitioned_thrust: %zu place(s), Y = Y0 + (alpha+1) X0 and sum verified\n", P);
  return 0;
}
