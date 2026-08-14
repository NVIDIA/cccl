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
 * @brief Frozen-import member walk: pushing a logical data at a concrete
 * replicated place imports EVERY member instance into the nested context
 * (population happens at the parent level), so a read at the replicated
 * place resolves in the nested context without issuing any copy -- in
 * particular no memcpy node lands in a conditional body graph.
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

  // ---- explicit push at the replicated place, then a conditional scope:
  // the walk runs at push time (repeat grid: equal members dedup to one
  // instance)
  {
    stackable_ctx ctx;
    auto grid = exec_place::repeat(exec_place::current_device(), nplaces);
    auto rep  = data_place::replicated(grid);

    auto lin  = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("fin");
    auto lacc = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("facc");
    ctx.parallel_for(lin.shape(), lin.write(), lacc.write())->*[] __device__(size_t i, auto in, auto acc) {
      in(i)  = static_cast<double>(i % 128);
      acc(i) = 0.0;
    };

    const size_t iters = 10;
    {
      auto rg = ctx.repeat_graph_scope(iters);
      // push INSIDE the scope (at root level a push is a no-op): the member
      // walk imports every replica; the accumulator is imported directly at
      // the grid's composite place so no instance needs a transfer inside
      // the scope
      lin.push(access_mode::read, rep);
      lacc.push(access_mode::rw, data_place::composite(blocked_partition(), grid));
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
    printf("frozen import: explicit push at replicated place OK\n");
  }

  // ---- read-only data: the auto-push sees the dependency's replicated
  // place and walks the members without an explicit push
  {
    stackable_ctx ctx;
    auto grid = exec_place::repeat(exec_place::current_device(), nplaces);
    auto rep  = data_place::replicated(grid);

    auto lro  = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("fro");
    auto lacc = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("froacc");
    ctx.parallel_for(lro.shape(), lro.write(), lacc.write())->*[] __device__(size_t i, auto in, auto acc) {
      in(i)  = static_cast<double>(i % 64);
      acc(i) = 0.0;
    };
    lro.set_read_only();

    const size_t iters = 6;
    {
      auto rg = ctx.repeat_graph_scope(iters);
      lacc.push(access_mode::rw, data_place::composite(blocked_partition(), grid));
      ctx.parallel_for(blocked_partition(), grid, lro.shape(), lro.read(rep), lacc.rw())
          ->*[] __device__(size_t i, auto in, auto acc) {
                acc(i) += in(i);
              };
    }

    ctx.host_launch(lacc.read())->*[&](auto acc) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(acc(i) == static_cast<double>(iters) * static_cast<double>(i % 64));
      }
    };
    ctx.finalize();
    printf("frozen import: auto-push walk for read-only data OK\n");
  }

  // ---- distinct member places (green contexts): the walk adopts one REAL
  // instance per member. A plain (non-conditional) graph scope keeps this
  // flavor clear of the driver rule that conditional bodies may not mix
  // CUDA contexts.
#  if _CCCL_CTK_AT_LEAST(12, 4)
  {
    ::std::optional<green_context_helper> gc_opt;
    try
    {
      gc_opt.emplace(8, 0);
    }
    catch (...)
    {}
    if (gc_opt && gc_opt->get_count() >= 2)
    {
      auto& gc = *gc_opt;
      stackable_ctx ctx;
      ::std::vector<exec_place> places;
      places.push_back(exec_place::green_ctx(gc.get_view(0), true));
      places.push_back(exec_place::green_ctx(gc.get_view(1), true));
      auto ggrid = make_grid(mv(places));
      auto grep  = data_place::replicated(ggrid);
      EXPECT(grep.member(0) != grep.member(1));

      auto lin  = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("gfin");
      auto lout = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("gfout");
      ctx.parallel_for(lin.shape(), lin.write(), lout.write())->*[] __device__(size_t i, auto in, auto out) {
        in(i)  = static_cast<double>(i % 32);
        out(i) = 0.0;
      };

      {
        auto gs = ctx.graph_scope();
        lin.push(access_mode::read, grep);
        lout.push(access_mode::rw, data_place::composite(blocked_partition(), ggrid));
        ctx.parallel_for(blocked_partition(), ggrid, lin.shape(), lin.read(grep), lout.rw())
            ->*[] __device__(size_t i, auto in, auto out) {
                  out(i) = 9.0 * in(i);
                };
      }

      ctx.host_launch(lout.read())->*[&](auto out) {
        for (size_t i = 0; i < n; i++)
        {
          EXPECT(out(i) == 9.0 * static_cast<double>(i % 32));
        }
      };
      ctx.finalize();
      printf("frozen import: distinct member instances adopted (green grid) OK\n");
    }
    else
    {
      printf("frozen import: green flavor skipped\n");
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 4)

  printf("replicated_frozen_import: all checks passed\n");
  return 0;
#endif
}
