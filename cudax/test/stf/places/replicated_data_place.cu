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

#if _CCCL_CTK_AT_LEAST(12, 4)
//! Replica instances are populated by copies issued INSIDE the nested
//! context of a stackable scope (the auto-push imports a single instance),
//! so the conditional body graph contains memcpy nodes. Some drivers reject
//! those at graph instantiation: probe for support so the stackable flavors
//! can waive instead of aborting.
bool conditional_body_memcpy_supported()
{
  cudaGraph_t g;
  cuda_safe_call(cudaGraphCreate(&g, 0));
  cudaGraphConditionalHandle handle;
  cuda_safe_call(cudaGraphConditionalHandleCreate(&handle, g, 1, cudaGraphCondAssignDefault));
  cudaGraphNodeParams np{};
  np.type               = cudaGraphNodeTypeConditional;
  np.conditional.handle = handle;
  np.conditional.type   = cudaGraphCondTypeWhile;
  np.conditional.size   = 1;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  const cudaGraphNode_t cnode = cuda_try<cudaGraphAddNode>(g, nullptr, nullptr, 0, &np);
#  else
  const cudaGraphNode_t cnode = cuda_try<cudaGraphAddNode>(g, nullptr, 0, &np);
#  endif
  (void) cnode;
  cudaGraph_t body = np.conditional.phGraph_out[0];

  void* a = nullptr;
  void* b = nullptr;
  cuda_safe_call(cudaMalloc(&a, 8));
  cuda_safe_call(cudaMalloc(&b, 8));
  cudaGraphNode_t cp;
  bool ok = (cudaGraphAddMemcpyNode1D(&cp, body, nullptr, 0, b, a, 8, cudaMemcpyDefault) == cudaSuccess);
  if (ok)
  {
    cudaGraphExec_t e;
    ok = (cudaGraphInstantiate(&e, g, 0) == cudaSuccess);
    if (ok)
    {
      cuda_safe_call(cudaGraphExecDestroy(e));
    }
  }
  cudaGetLastError(); // clear any probe failure
  cuda_safe_call(cudaFree(a));
  cuda_safe_call(cudaFree(b));
  cuda_safe_call(cudaGraphDestroy(g));
  return ok;
}

__global__ void probe_noop_kernel() {}

//! Conditional body graphs additionally require every kernel node to belong
//! to the SAME CUDA context ("all kernels ... must belong to the same CUDA
//! context"), and green contexts are distinct contexts. A stackable scope
//! running a green-grid task therefore mixes the primary context (the
//! condition-update kernel) with the grid's green contexts inside the body:
//! probe whether the driver accepts that at instantiation.
bool conditional_body_multi_context_supported(green_context_helper& gc)
{
  cudaGraph_t g;
  cuda_safe_call(cudaGraphCreate(&g, 0));
  cudaGraphConditionalHandle handle;
  cuda_safe_call(cudaGraphConditionalHandleCreate(&handle, g, 1, cudaGraphCondAssignDefault));
  cudaGraphNodeParams np{};
  np.type               = cudaGraphNodeTypeConditional;
  np.conditional.handle = handle;
  np.conditional.type   = cudaGraphCondTypeWhile;
  np.conditional.size   = 1;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  const cudaGraphNode_t cnode = cuda_try<cudaGraphAddNode>(g, nullptr, nullptr, 0, &np);
#  else
  const cudaGraphNode_t cnode = cuda_try<cudaGraphAddNode>(g, nullptr, 0, &np);
#  endif
  (void) cnode;
  cudaGraph_t body = np.conditional.phGraph_out[0];

  cudaKernelNodeParams kp{};
  kp.func           = (void*) probe_noop_kernel;
  kp.gridDim        = dim3(1);
  kp.blockDim       = dim3(1);
  kp.sharedMemBytes = 0;
  kp.kernelParams   = nullptr;
  kp.extra          = nullptr;

  // one kernel in the current (primary) context, then one per green context
  cudaGraphNode_t k;
  cuda_safe_call(cudaGraphAddKernelNode(&k, body, nullptr, 0, &kp));
  CUcontext prev;
  cuda_safe_call(cuCtxGetCurrent(&prev));
  for (size_t v = 0; v < 2; v++)
  {
    CUcontext gctx;
    cuda_safe_call(cuCtxFromGreenCtx(&gctx, gc.get_view(v).g_ctx));
    cuda_safe_call(cuCtxSetCurrent(gctx));
    cuda_safe_call(cudaGraphAddKernelNode(&k, body, nullptr, 0, &kp));
  }
  cuda_safe_call(cuCtxSetCurrent(prev));

  cudaGraphExec_t e;
  bool ok = (cudaGraphInstantiate(&e, g, 0) == cudaSuccess);
  if (ok)
  {
    cuda_safe_call(cudaGraphExecDestroy(e));
  }
  cudaGetLastError(); // clear any probe failure
  cuda_safe_call(cudaGraphDestroy(g));
  return ok;
}
#endif // _CCCL_CTK_AT_LEAST(12, 4)

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

  const bool stackable_ok = conditional_body_memcpy_supported();
  if (!stackable_ok)
  {
    printf("replicated data place: stackable flavors will be skipped (driver rejects memcpy nodes in conditional "
           "body graphs)\n");
  }

  // ---- stream and graph backends
  for (int use_graph = 0; use_graph < 2; use_graph++)
  {
    context ctx;
    if (use_graph)
    {
      ctx = graph_ctx();
    }

    auto grid = exec_place::repeat(exec_place::current_device(), nplaces);
    auto rep  = data_place::replicated(grid);
    auto lin  = ctx.logical_data(&ref[0], {n});
    auto lout = ctx.logical_data(shape_of<slice<double>>(n));
    auto tok  = ctx.token(); // a void_interface dep: the instances tuple is
                            // shorter than the deps tuple, which the
                            // replicated rebase must tolerate

    // read at the replicated place: each shard must see the payload through
    // its own replica, and results must match the reference everywhere
    ctx.parallel_for(blocked_partition(), grid, lin.shape(), lin.read(rep), lout.write(), tok.write())
        ->*[] __device__(size_t i, auto in, auto out) {
              out(i) = 2.0 * in(i);
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

    // deferred form on this backend: materialized against the launch's
    // own execution place at acquire
    auto lout_d = ctx.logical_data(shape_of<slice<double>>(n));
    ctx.parallel_for(blocked_partition(), grid, lin.shape(), lin.read(data_place::replicated()), lout_d.write())
        ->*[] __device__(size_t i, auto in, auto out) {
              out(i) = 6.0 * in(i);
            };
    ctx.host_launch(lout_d.read())->*[&](auto out) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(out(i) == 6.0 * ref[i]);
      }
    };

    // merged-mode contract: declaring the SAME data as replicated-read and
    // writable in one task merges the access modes past read; the combined
    // dependency must be rejected at acquisition
    bool thrown_merged = false;
    try
    {
      ctx.parallel_for(blocked_partition(), grid, lout.shape(), lout.read(rep), lout.rw())
          ->*[] __device__(size_t, auto, auto) {};
    }
    catch (const ::std::invalid_argument&)
    {
      thrown_merged = true;
    }
    EXPECT(thrown_merged);

    // ---- mutation cycle: "mutate the data at another place; the next
    // replicated read re-broadcasts" -- the coherence claim itself. Read at
    // the replicated place, mutate at the affine place, read replicated
    // again: the second generation must see the update through every
    // replica. In the graph backend the whole cycle (including the
    // re-broadcast copies) lands inside one captured graph.
    auto lgen = ctx.logical_data(shape_of<slice<double>>(n));
    auto lo1  = ctx.logical_data(shape_of<slice<double>>(n));
    auto lo2  = ctx.logical_data(shape_of<slice<double>>(n));
    ctx.parallel_for(lgen.shape(), lgen.write())->*[] __device__(size_t i, auto x) {
      x(i) = 1.0;
    };
    ctx.parallel_for(blocked_partition(), grid, lgen.shape(), lgen.read(rep), lo1.write())
        ->*[] __device__(size_t i, auto in, auto out) {
              out(i) = in(i);
            };
    ctx.parallel_for(lgen.shape(), lgen.rw())->*[] __device__(size_t i, auto x) {
      x(i) += 41.0;
    };
    ctx.parallel_for(blocked_partition(), grid, lgen.shape(), lgen.read(rep), lo2.write())
        ->*[] __device__(size_t i, auto in, auto out) {
              out(i) = in(i);
            };
    ctx.host_launch(lo1.read(), lo2.read())->*[&](auto o1, auto o2) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(o1(i) == 1.0);
        EXPECT(o2(i) == 42.0);
      }
    };
    ctx.finalize();

    printf("replicated data place: %s backend OK\n", use_graph ? "graph" : "stream");
  }

  // ---- green-context grid: member places are DISTINCT, so a replicated
  // dep pins one real instance per member (the repeat grid above
  // deduplicates to a single instance by place equality)
#  if _CCCL_CTK_AT_LEAST(12, 4)
  {
    ::std::optional<green_context_helper> gc_opt;
    try
    {
      gc_opt.emplace(8, 0);
    }
    catch (...)
    {
      printf("replicated data place: green-context flavors skipped (unsupported hardware)\n");
    }
    if (gc_opt && gc_opt->get_count() >= 2)
    {
      auto& gc = *gc_opt;
      ::std::vector<exec_place> places;
      places.push_back(exec_place::green_ctx(gc.get_view(0), true));
      places.push_back(exec_place::green_ctx(gc.get_view(1), true));
      auto ggrid = make_grid(mv(places));
      auto grep  = data_place::replicated(ggrid);
      EXPECT(grep.instance_count() == 2);
      EXPECT(grep.member(0) != grep.member(1));

      context ctx;
      for (int use_graph_g = 0; use_graph_g < 2; use_graph_g++)
      {
        if (use_graph_g)
        {
          ctx = graph_ctx();
        }
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
        printf("replicated data place: green-context grid (distinct member instances) %s OK\n",
               use_graph_g ? "[graph]" : "[stream]");
        printf("replicated data place: deferred form (grid-bound at acquire + scalar degenerate) %s OK\n",
               use_graph_g ? "[graph]" : "[stream]");
      }

      // ---- axis-grouped replication on a (2, 2) grid: axis 0 = the two
      // green domains (REPLICATED), axis 1 = two execution slots per domain
      // (SHARED). One instance per domain, shared by its two slots.
      {
        ::std::vector<exec_place> ps22;
        ps22.push_back(exec_place::green_ctx(gc.get_view(0), true)); // (0,0)
        ps22.push_back(exec_place::green_ctx(gc.get_view(1), true)); // (1,0)
        ps22.push_back(exec_place::green_ctx(gc.get_view(0), true)); // (0,1)
        ps22.push_back(exec_place::green_ctx(gc.get_view(1), true)); // (1,1)
        auto grid22 = make_grid(mv(ps22), dim4(2, 2));
        auto rep22  = data_place::replicated(grid22, replicate_over<0>);

        // projection math: one instance per axis-0 coordinate
        EXPECT(rep22.instance_count() == 2);
        EXPECT(rep22.instance_of(0) == 0); // (0,0)
        EXPECT(rep22.instance_of(1) == 1); // (1,0)
        EXPECT(rep22.instance_of(2) == 0); // (0,1) shares domain 0's instance
        EXPECT(rep22.instance_of(3) == 1); // (1,1) shares domain 1's instance
        EXPECT(rep22.member(0) != rep22.member(1));

        context gctx;
        // the grouped resolution is backend-agnostic: same launch on the
        // stream and graph backends
        for (int use_graph22 = 0; use_graph22 < 2; use_graph22++)
        {
          if (use_graph22)
          {
            gctx = graph_ctx();
          }
          auto lgin  = gctx.logical_data(&ref[0], {n});
          auto lgout = gctx.logical_data(shape_of<slice<double>>(n));
          gctx.parallel_for(blocked_partition(), grid22, lgin.shape(), lgin.read(rep22), lgout.write())
              ->*[] __device__(size_t i, auto in, auto out) {
                    out(i) = 4.0 * in(i);
                  };
          gctx.host_launch(lgout.read())->*[&](auto out) {
            for (size_t i = 0; i < n; i++)
            {
              EXPECT(out(i) == 4.0 * ref[i]);
            }
          };
          gctx.finalize();
        }

        // and inside a stackable conditional scope: the auto-push imports
        // the data once, the nested acquire pins one instance per axis-0
        // group
        const bool multi_ctx_ok = conditional_body_multi_context_supported(gc);
        if (!multi_ctx_ok)
        {
          printf("replicated data place: green stackable flavors skipped (driver requires a single CUDA context in "
                 "conditional body graphs)\n");
        }
        if (stackable_ok && multi_ctx_ok)
        {
          stackable_ctx sctx;
          auto lsin  = sctx.logical_data(shape_of<slice<double>>(n)).set_symbol("g22in");
          auto lsacc = sctx.logical_data(shape_of<slice<double>>(n)).set_symbol("g22acc");
          sctx.parallel_for(lsin.shape(), lsin.write(), lsacc.write())->*[] __device__(size_t i, auto in, auto acc) {
            in(i)  = static_cast<double>(i % 32);
            acc(i) = 0.0;
          };
          const size_t iters22 = 6;
          {
            auto rg = sctx.repeat_graph_scope(iters22);
            sctx.parallel_for(blocked_partition(), grid22, lsin.shape(), lsin.read(rep22), lsacc.rw())
                ->*[] __device__(size_t i, auto in, auto acc) {
                      acc(i) += in(i);
                    };
          }
          sctx.host_launch(lsacc.read())->*[&](auto acc) {
            for (size_t i = 0; i < n; i++)
            {
              EXPECT(acc(i) == static_cast<double>(iters22) * static_cast<double>(i % 32));
            }
          };
          sctx.finalize();
        }

        // shared axes require co-located fibers: a (2, 2) arrangement whose
        // axis-1 fiber crosses domains must be rejected at construction
        ::std::vector<exec_place> bad;
        bad.push_back(exec_place::green_ctx(gc.get_view(0), true)); // (0,0)
        bad.push_back(exec_place::green_ctx(gc.get_view(1), true)); // (1,0)
        bad.push_back(exec_place::green_ctx(gc.get_view(1), true)); // (0,1) != (0,0)
        bad.push_back(exec_place::green_ctx(gc.get_view(0), true)); // (1,1) != (1,0)
        auto bad_grid = make_grid(mv(bad), dim4(2, 2));
        bool thrown22 = false;
        try
        {
          auto r = data_place::replicated(bad_grid, replicate_over<0>);
        }
        catch (const ::std::invalid_argument&)
        {
          thrown22 = true;
        }
        EXPECT(thrown22);
        printf("replicated data place: axis-grouped replication on a (2,2) grid OK (stream/graph/stackable)\n");
      }
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
  if (stackable_ok)
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

    // the DEFERRED form through the stackable auto-push: the push imports
    // the data at the context's default place; the nested task's acquire
    // then materializes replicated() against its own execution place
    {
      auto rg = ctx.repeat_graph_scope(iters);
      ctx.parallel_for(blocked_partition(), grid, lin.shape(), lin.read(data_place::replicated()), lacc.rw())
          ->*[] __device__(size_t i, auto in, auto acc) {
                acc(i) += in(i);
              };
    }

    ctx.host_launch(lacc.read())->*[&](auto acc) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(acc(i) == 2.0 * static_cast<double>(iters) * static_cast<double>(i % 128));
      }
    };
    ctx.finalize();
    printf("replicated data place: stackable conditional scope (concrete + deferred deps) OK\n");
  }

  // ---- stackable + DISTINCT member places: true multi-replica through the
  // auto-push and a conditional scope (the repeat-grid section above only
  // exercises the equal-places dedup degenerate)
#  if _CCCL_CTK_AT_LEAST(12, 4)
  {
    ::std::optional<green_context_helper> gc_opt;
    try
    {
      gc_opt.emplace(8, 0);
    }
    catch (...)
    {}
    if (stackable_ok && gc_opt && gc_opt->get_count() >= 2 && conditional_body_multi_context_supported(*gc_opt))
    {
      auto& gc = *gc_opt;
      stackable_ctx ctx;
      ::std::vector<exec_place> places;
      places.push_back(exec_place::green_ctx(gc.get_view(0), true));
      places.push_back(exec_place::green_ctx(gc.get_view(1), true));
      auto ggrid = make_grid(mv(places));

      auto lin  = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("gin");
      auto lacc = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("gacc");
      ctx.parallel_for(lin.shape(), lin.write(), lacc.write())->*[] __device__(size_t i, auto in, auto acc) {
        in(i)  = static_cast<double>(i % 64);
        acc(i) = 0.0;
      };

      const size_t iters = 8;
      {
        auto rg = ctx.repeat_graph_scope(iters);
        ctx.parallel_for(blocked_partition(), ggrid, lin.shape(), lin.read(data_place::replicated()), lacc.rw())
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
      printf("replicated data place: stackable + distinct member replicas OK\n");
    }
    else
    {
      printf("replicated data place: stackable green flavor skipped\n");
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 4)

  printf("replicated_data_place: all checks passed\n");
  return 0;
#endif
}
