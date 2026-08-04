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
 * @brief data_place::replicated — one copy of a logical data per grid
 * member, read-only, fan-out on copy-in, per-place instance rebase in
 * parallel_for. Covered on the stream backend, the graph backend, and
 * inside a stackable conditional graph scope.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  const size_t n       = 1 << 20;
  const size_t nplaces = 2;

  ::std::vector<double> ref(n);
  for (size_t i = 0; i < n; i++)
  {
    ref[i] = 0.25 * static_cast<double>(i % 1024) + 1.0;
  }

  // ---- stream and graph backends
  for (int use_graph = 0; use_graph < 2; use_graph++)
  {
    context ctx;
    if (use_graph)
    {
      ctx = graph_ctx();
    }

    auto grid    = exec_place::repeat(exec_place::current_device(), nplaces);
    auto rep     = data_place::replicated(grid);
    auto lin     = ctx.logical_data(&ref[0], {n});
    auto lout    = ctx.logical_data(shape_of<slice<double>>(n));
    auto lplaces = ctx.logical_data(shape_of<slice<int>>(n));

    // read at the replicated place: each shard must see the payload through
    // its own replica, and results must match the reference everywhere
    ctx.parallel_for(blocked_partition(), grid, lin.shape(), lin.read(rep), lout.write(), lplaces.write())
        ->*[] __device__(size_t i, auto in, auto out, auto pl) {
              out(i) = 2.0 * in(i);
              int smid;
              asm("mov.u32 %0, %%smid;" : "=r"(smid));
              pl(i) = smid;
            };

    ctx.host_launch(lout.read())->*[&](auto out) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(out(i) == 2.0 * ref[i]);
      }
    };

    // read-only contract: any non-read access at a replicated place throws
    bool thrown = false;
    try
    {
      ctx.parallel_for(blocked_partition(), grid, lout.shape(), lout.rw(rep))->*[] __device__(size_t, auto) {};
    }
    catch (const ::std::invalid_argument&)
    {
      thrown = true;
    }
    EXPECT(thrown);

    ctx.finalize();
    printf("replicated data place: %s backend OK\n", use_graph ? "graph" : "stream");
  }

  // ---- green-context grid: member places are DISTINCT, so a replicated
  // dep pins one real instance per member (the repeat grid above
  // deduplicates to a single instance by place equality)
#  if _CCCL_CTK_AT_LEAST(12, 4)
  {
    green_context_helper gc(8, 0);
    if (gc.get_count() >= 2)
    {
      context ctx;
      ::std::vector<exec_place> places;
      places.push_back(exec_place::green_ctx(gc.get_view(0), true));
      places.push_back(exec_place::green_ctx(gc.get_view(1), true));
      auto ggrid = make_grid(mv(places));
      auto grep  = data_place::replicated(ggrid);
      EXPECT(grep.instance_count() == 2);
      EXPECT(grep.member(0) != grep.member(1));

      auto lin  = ctx.logical_data(&ref[0], {n});
      auto lout = ctx.logical_data(shape_of<slice<double>>(n));
      ctx.parallel_for(blocked_partition(), ggrid, lin.shape(), lin.read(grep), lout.write())
          ->*[] __device__(size_t i, auto in, auto out) {
                out(i) = 3.0 * in(i);
              };
      ctx.host_launch(lout.read())->*[&](auto out) {
        for (size_t i = 0; i < n; i++)
        {
          EXPECT(out(i) == 3.0 * ref[i]);
        }
      };
      // deferred form: the grid is bound at acquire from the launch's
      // execution place -- no grid repeated at the call site
      auto lout2 = ctx.logical_data(shape_of<slice<double>>(n));
      ctx.parallel_for(blocked_partition(), ggrid, lin.shape(), lin.read(data_place::replicated()), lout2.write())
          ->*[] __device__(size_t i, auto in, auto out) {
                out(i) = 5.0 * in(i);
              };
      ctx.host_launch(lout2.read())->*[&](auto out) {
        for (size_t i = 0; i < n; i++)
        {
          EXPECT(out(i) == 5.0 * ref[i]);
        }
      };

      // scalar degenerate: on a single-place task the deferred form
      // materializes to the place's affine data place
      auto lout3 = ctx.logical_data(shape_of<slice<double>>(n));
      ctx.parallel_for(lin.shape(), lin.read(data_place::replicated()), lout3.write())
          ->*[] __device__(size_t i, auto in, auto out) {
                out(i) = 7.0 * in(i);
              };
      ctx.host_launch(lout3.read())->*[&](auto out) {
        for (size_t i = 0; i < n; i++)
        {
          EXPECT(out(i) == 7.0 * ref[i]);
        }
      };

      ctx.finalize();
      printf("replicated data place: green-context grid (distinct member instances) OK\n");
      printf("replicated data place: deferred form (grid-bound at acquire + scalar degenerate) OK\n");
    }
    else
    {
      printf("replicated data place: green-context flavor skipped (single group)\n");
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 4)

  // ---- stackable conditional graph scope: replicas are ordinary
  // stream-ordered instances, so the auto-push (freeze + get within the
  // nested context) and pop-time lifetime need no special handling
  {
    stackable_ctx ctx;
    auto grid = exec_place::repeat(exec_place::current_device(), nplaces);
    auto rep  = data_place::replicated(grid);

    auto lin  = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("in");
    auto lacc = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("acc");
    ctx.parallel_for(lin.shape(), lin.write(), lacc.write())->*[] __device__(size_t i, auto in, auto acc) {
      in(i)  = static_cast<double>(i % 128);
      acc(i) = 0.0;
    };

    const size_t iters = 10;
    {
      auto rg = ctx.repeat_graph_scope(iters);
      ctx.parallel_for(blocked_partition(), grid, lin.shape(), lin.read(rep), lacc.rw())
          ->*[] __device__(size_t i, auto in, auto acc) {
                acc(i) += in(i);
              };
    }

    ctx.host_launch(lacc.read())->*[&](auto acc) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(acc(i) == static_cast<double>(iters) * static_cast<double>(i % 128));
      }
    };
    ctx.finalize();
    printf("replicated data place: stackable conditional scope OK\n");
  }

  printf("replicated_data_place: all checks passed\n");
  return 0;
#endif
}
