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
 * @brief PROTOTYPE — a grid task over composite-placed logical data whose
 *        body gets the sharded view of each argument (`task_view`) and a
 *        lazy per-place env range from the task (`task_envs`), then calls
 *        (1) Thrust per shard, (2) a sharded verb, (3) a sharded reduction,
 *        with STF keeping the bracketing. Also exercises the getter's
 *        contract: replicated throws, composite over another grid throws,
 *        single place under a grid task throws, single place under a
 *        single-place task gives one shard.
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

namespace
{

double X0(size_t i)
{
  return sin(static_cast<double>(i));
}
double Y0(size_t i)
{
  return cos(static_cast<double>(i));
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

template <class F>
bool expect_throw(const char* label, F&& f)
{
  try
  {
    f();
  }
  catch (const ::std::invalid_argument& e)
  {
    printf("  %s: threw as expected: %s\n", label, e.what());
    return true;
  }
  printf("  %s: DID NOT THROW\n", label);
  return false;
}

} // namespace

int main()
{
  setvbuf(stdout, nullptr, _IONBF, 0);
  cuda_safe_call(cudaSetDevice(0));

  auto grid      = exec_place::all_locality_domains();
  const size_t P = grid.size();
  printf("sharded_task_view: task grid %s (%zu places)\n", grid.to_string().c_str(), P);

  const size_t N     = 4 * 1024 * 1024 + 101; // ragged tail
  const double alpha = 3.14;

  ::std::vector<double> X(N), Y(N);
  for (size_t i = 0; i < N; ++i)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  stream_ctx ctx;
  auto lX = ctx.logical_data(make_slice(X.data(), N)).set_symbol("X");
  auto lY = ctx.logical_data(make_slice(Y.data(), N)).set_symbol("Y");

  auto dist = data_place::composite(blocked_partition(), grid);

  double sum_in_task = 0.0;
  bool contract_ok   = true;

  // ------------------------------------------------------------------
  // [1] The grid task: views from the arguments, envs from the task.
  // ------------------------------------------------------------------
  auto t1 = ctx.task(grid, lX.read(dist), lY.rw(dist));
  t1.set_symbol("saxpy+add+sum");
  t1->*[&](auto, auto dX, auto dY) {
    auto& t = t1;
    auto vX = sh::task_view(t, 0, dX);
    auto vY = sh::task_view(t, 1, dY);
    EXPECT(vX.num_shards() == P);
    EXPECT(vY.num_shards() == P);
    EXPECT(vX.size() == N);

    // The logical cut: report it, and verify against the VMM backing.
    for (size_t g = 0; g < P; ++g)
    {
      const auto& s = vY.shard(g);
      printf("  shard %zu: [%zu, %zu) on %s\n", g, s.global_offset, s.global_offset + s.size, s.place.to_string().c_str());
    }
    if (const auto* backing = sh::composite_backing(dY.data_handle()))
    {
      const auto& st = backing->get_stats();
      printf("  VMM backing: %zu blocks in %zu allocations, placement accuracy %.1f%%\n",
             st.nblocks,
             st.nallocs,
             100.0 * st.accuracy);
    }

    auto envs   = sh::task_envs(t, vY);
    const auto call_env = ::cuda::stream_ref{t.get_stream()};

    // (1) Thrust arm: per shard, on the task's stream for that place. Y = alpha X + Y
    sh::for_each_shard(
      vY,
      envs,
      [&](size_t g, const auto& d, cudaStream_t s) {
        const auto& x = vX.shard(g);
        thrust::transform(thrust::cuda::par_nosync.on(s), x.data, x.data + x.size, d.data, d.data, saxpy_functor{alpha});
      },
      call_env);

    // (2) Verb arm: Y = Y + X (lane-ordered after (1) on the same streams)
    sh::zip_transform(vY, envs, add_op{}, call_env, vY, vX);

    // (3) Reduction arm: synchronous by contract (blocks the host inside the
    //     body; the asynchronous form is reduce_into a device scalar).
    sum_in_task = sh::sum(vY, envs, call_env);
  };

  // ------------------------------------------------------------------
  // [2] Contract: throws
  // ------------------------------------------------------------------
  auto t2 = ctx.task(grid, lX.read(data_place::replicated(grid)));
  t2.set_symbol("replicated");
  t2->*[&](auto, auto dX) {
    auto& t = t2;
    contract_ok &= expect_throw("replicated argument", [&] {
      (void) sh::task_view(t, 0, dX);
    });
  };

  // "single-place instance under a grid task" never reaches the body: STF
  // itself asserts (`dep_allocate requires a resolved data_place`) when a
  // grid task's dependency has no explicit data place. The getter's throw
  // for that row is a backstop, enforced upstream.

  auto one = grid.get_place(0);
  if (P > 1)
  {
    auto t4 = ctx.task(one, lX.read(dist));
    t4.set_symbol("other-grid");
    t4->*[&](auto, auto dX) {
      auto& t = t4;
      contract_ok &= expect_throw("composite over another grid", [&] {
        (void) sh::task_view(t, 0, dX);
      });
    };
  }

  // ------------------------------------------------------------------
  // [3] Single place, single-place task: one shard, verbs still work.
  // ------------------------------------------------------------------
  double sum_single = 0.0;
  auto t5 = ctx.task(one, lX.read());
  t5.set_symbol("single");
  t5->*[&](auto, auto dX) {
    auto& t = t5;
    auto vX = sh::task_view(t, 0, dX);
    EXPECT(vX.num_shards() == 1);
    EXPECT(vX.shard(0).size == N);
    auto envs  = sh::task_envs(t, vX);
    sum_single = sh::sum(vX, envs, ::cuda::stream_ref{t.get_stream()});
  };

  ctx.finalize();

  // ------------------------------------------------------------------
  // Verification
  // ------------------------------------------------------------------
  double sum_ref = 0.0, sum_x = 0.0;
  for (size_t i = 0; i < N; ++i)
  {
    const double expected = Y0(i) + (alpha + 1.0) * X0(i);
    if (fabs(Y[i] - expected) > 1e-9)
    {
      fprintf(stderr, "Verification FAILED at %zu: %f vs %f\n", i, Y[i], expected);
      return 1;
    }
    sum_ref += expected;
    sum_x += X0(i);
  }
  if (fabs(sum_in_task - sum_ref) > 1e-6 * fabs(sum_ref) + 1e-6)
  {
    fprintf(stderr, "sum in task FAILED: %f vs %f\n", sum_in_task, sum_ref);
    return 1;
  }
  if (fabs(sum_single - sum_x) > 1e-6 * fabs(sum_x) + 1e-6)
  {
    fprintf(stderr, "single-place sum FAILED: %f vs %f\n", sum_single, sum_x);
    return 1;
  }
  if (!contract_ok)
  {
    fprintf(stderr, "contract FAILED\n");
    return 1;
  }
  printf("sharded_task_view: Y = Y0 + (alpha+1) X0 verified on host; sums match; contract holds: PASSED\n");
  return 0;
}
