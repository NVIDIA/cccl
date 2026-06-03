//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

__global__ void scale_kernel(int cnt, double* data, double factor)
{
  const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  for (int i = tid; i < cnt; i += nthreads)
  {
    data[i] *= factor;
  }
}

__global__ void increment_kernel(int cnt, double* data)
{
  const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  for (int i = tid; i < cnt; i += nthreads)
  {
    data[i] += 1.0;
  }
}

C2H_TEST("stackable: push_graph / pop", "[stackable]")
{
  const size_t N = 256;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = static_cast<double>(i);
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  // Multiply by 2 inside a nested graph scope.
  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    scale_kernel<<<2, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d, 2.0);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_stackable_pop(ctx);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 2.0 * static_cast<double>(i)) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: pop_prologue relaunch accumulates N times", "[stackable][launchable]")
{
  const size_t N      = 256;
  const int relaunchN = 16;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    increment_kernel<<<2, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d);
    stf_task_end(t);
    stf_task_destroy(t);
  }

  // Two-phase pop: instantiate the graph, launch it relaunchN times, then
  // run the epilogue to release resources and unfreeze lA.
  stf_launchable_graph_handle lh = stf_stackable_pop_prologue(ctx);
  REQUIRE(lh != nullptr);
  for (int k = 0; k < relaunchN; ++k)
  {
    stf_launchable_graph_launch(lh);
  }
  stf_stackable_pop_epilogue(ctx);
  stf_launchable_graph_destroy(lh);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - static_cast<double>(relaunchN)) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: pop_prologue with zero launches unfreezes", "[stackable][launchable]")
{
  const size_t N = 128;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 7.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  // Push + submit work, but never launch the graph. The epilogue must still
  // release resources so that lA is unfrozen and reusable below.
  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    increment_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_launchable_graph_handle lh = stf_stackable_pop_prologue(ctx);
  REQUIRE(lh != nullptr);
  stf_stackable_pop_epilogue(ctx);
  stf_launchable_graph_destroy(lh);

  // Normal push_graph/pop still works after a zero-launch prologue+epilogue.
  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    scale_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d, 2.0);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_stackable_pop(ctx);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  // Zero-launch means the first graph never ran. The second scope doubled
  // the initial 7.0 to 14.0.
  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 14.0) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: launchable exec and stream accessors are non-null", "[stackable][launchable]")
{
  const size_t N = 64;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    increment_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_launchable_graph_handle lh = stf_stackable_pop_prologue(ctx);
  REQUIRE(lh != nullptr);

  // Accessors must be valid between prologue and epilogue. graph() must
  // return a live cudaGraph_t without forcing instantiation, exec() must
  // return a live cudaGraphExec_t, stream() is pure observation.
  REQUIRE(stf_launchable_graph_graph(lh) != nullptr);
  REQUIRE(stf_launchable_graph_exec(lh) != nullptr);
  REQUIRE(stf_launchable_graph_stream(lh) != nullptr);

  stf_launchable_graph_launch(lh);

  stf_stackable_pop_epilogue(ctx);
  stf_launchable_graph_destroy(lh);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 1.0) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: launchable graph() embed into outer graph", "[stackable][launchable]")
{
  const size_t N = 64;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    increment_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_launchable_graph_handle lh = stf_stackable_pop_prologue(ctx);
  REQUIRE(lh != nullptr);

  // Grab the underlying cudaGraph_t WITHOUT forcing instantiation and
  // without ever calling stf_launchable_graph_exec(). The child graph
  // built by the nested scope is embedded into an outer graph which is
  // instantiated and launched manually here.
  cudaGraph_t child_graph = stf_launchable_graph_graph(lh);
  REQUIRE(child_graph != nullptr);

  cudaStream_t support_stream = stf_launchable_graph_stream(lh);
  REQUIRE(support_stream != nullptr);

  cudaGraph_t outer = nullptr;
  REQUIRE(cudaGraphCreate(&outer, 0) == cudaSuccess);

  cudaGraphNode_t child_node = nullptr;
  REQUIRE(cudaGraphAddChildGraphNode(&child_node, outer, nullptr, 0, child_graph) == cudaSuccess);

  cudaGraphExec_t outer_exec = nullptr;
#if _CCCL_CTK_AT_LEAST(12, 0)
  REQUIRE(cudaGraphInstantiate(&outer_exec, outer, 0) == cudaSuccess);
#else
  REQUIRE(cudaGraphInstantiate(&outer_exec, outer, nullptr, nullptr, 0) == cudaSuccess);
#endif

  // Route the outer launch through the support stream: since graph() has
  // triggered the lazy dep-A sync on that stream, it is safe to drive
  // cudaGraphLaunch on it here.
  REQUIRE(cudaGraphLaunch(outer_exec, support_stream) == cudaSuccess);

  REQUIRE(cudaGraphExecDestroy(outer_exec) == cudaSuccess);
  REQUIRE(cudaGraphDestroy(outer) == cudaSuccess);

  stf_stackable_pop_epilogue(ctx);
  stf_launchable_graph_destroy(lh);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 1.0) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: shared pop_prologue dup/free releases only at last free", "[stackable][launchable]")
{
  const size_t N      = 128;
  const int relaunchN = 5;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    increment_kernel<<<2, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d);
    stf_task_end(t);
    stf_task_destroy(t);
  }

  stf_launchable_graph_shared h1 = nullptr;
  REQUIRE(stf_stackable_pop_prologue_shared(ctx, &h1) == 0);
  REQUIRE(h1 != nullptr);
  REQUIRE(stf_launchable_graph_shared_valid(h1) == 1);
  REQUIRE(stf_launchable_graph_shared_stream(h1) != nullptr);

  // Dup before launching anything: both handles must be able to drive the
  // same underlying graph.
  stf_launchable_graph_shared h2 = nullptr;
  REQUIRE(stf_launchable_graph_shared_dup(h1, &h2) == 0);
  REQUIRE(h2 != nullptr);
  REQUIRE(stf_launchable_graph_shared_valid(h2) == 1);

  for (int k = 0; k < relaunchN; ++k)
  {
    // Alternate between the two handles - both must work.
    if ((k & 1) == 0)
    {
      stf_launchable_graph_shared_launch(h1);
    }
    else
    {
      stf_launchable_graph_shared_launch(h2);
    }
  }

  // Free one handle; the other must still launch. No pop_epilogue yet.
  stf_launchable_graph_shared_free(h1);
  REQUIRE(stf_launchable_graph_shared_valid(h2) == 1);
  stf_launchable_graph_shared_launch(h2);

  // Free the last handle: pop_epilogue runs automatically here.
  stf_launchable_graph_shared_free(h2);

  // The context must be usable again after the shared release.
  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_enable_capture(t);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    scale_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d, 2.0);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_stackable_pop(ctx);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  // Each launch added +1; final scale doubled; relaunchN launches via h1/h2
  // plus one extra launch via h2 after free(h1) -> (relaunchN + 1) * 2.
  const double expected = 2.0 * (static_cast<double>(relaunchN) + 1.0);
  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - expected) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: shared pop_prologue tolerates NULL free", "[stackable][launchable]")
{
  // stf_launchable_graph_shared_free(NULL) must be a no-op just like the
  // other destroy entry points. The valid() probe returns 0 for NULL.
  stf_launchable_graph_shared_free(nullptr);
  REQUIRE(stf_launchable_graph_shared_valid(nullptr) == 0);
}

C2H_TEST("stackable: nested push_graph scopes", "[stackable]")
{
  const size_t N = 128;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  // Two nested scopes: each scope adds 1.0, so after popping both we expect 2.0.
  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    stf_task_start(t);
    double* d = static_cast<double*>(stf_task_get(t, 0));
    increment_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(static_cast<int>(N), d);
    stf_task_end(t);
    stf_task_destroy(t);

    stf_stackable_push_graph(ctx);
    {
      stf_task_handle t2 = stf_stackable_task_create(ctx);
      REQUIRE(t2 != nullptr);
      stf_stackable_task_add_dep(ctx, t2, lA, STF_RW);
      stf_task_start(t2);
      double* d2 = static_cast<double*>(stf_task_get(t2, 0));
      increment_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t2)>>>(static_cast<int>(N), d2);
      stf_task_end(t2);
      stf_task_destroy(t2);
    }
    stf_stackable_pop(ctx);
  }
  stf_stackable_pop(ctx);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 2.0) < 1e-10);
  }

  cudaFreeHost(host_data);
}

C2H_TEST("stackable: token + fence", "[stackable]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  stf_logical_data_handle tok = stf_stackable_token(ctx);
  REQUIRE(tok != nullptr);

  // Sequential task chain through the token: t1 (write) -> t2 (read).
  stf_task_handle t1 = stf_stackable_task_create(ctx);
  REQUIRE(t1 != nullptr);
  stf_stackable_task_add_dep(ctx, t1, tok, STF_WRITE);
  stf_task_start(t1);
  stf_task_end(t1);
  stf_task_destroy(t1);

  stf_task_handle t2 = stf_stackable_task_create(ctx);
  REQUIRE(t2 != nullptr);
  stf_stackable_task_add_dep(ctx, t2, tok, STF_READ);
  stf_task_start(t2);
  stf_task_end(t2);
  stf_task_destroy(t2);

  cudaStream_t fence = stf_stackable_ctx_fence(ctx);
  REQUIRE(cudaStreamSynchronize(fence) == cudaSuccess);

  stf_stackable_token_destroy(tok);
  stf_stackable_ctx_finalize(ctx);
}

#if _CCCL_CTK_AT_LEAST(12, 4)

// Smoke test for the while/repeat C-API surface: create+destroy each kind of
// scope without populating a body. Body-level integration is exercised at the
// C++ level by cudax/test/stf/local_stf/stackable_nested_repeat.cu and the
// graph_scope_test, but the C-API task-driven body still needs a follow-up to
// nail down the right capture path; tracked separately.
C2H_TEST("stackable: push_repeat / pop_repeat smoke", "[stackable][repeat]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  stf_repeat_scope_handle scope = stf_stackable_push_repeat(ctx, /*count=*/1);
  REQUIRE(scope != nullptr);
  stf_stackable_pop_repeat(scope);

  stf_stackable_ctx_finalize(ctx);
}

C2H_TEST("stackable: push_while / pop_while smoke", "[stackable][while]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  stf_while_scope_handle scope = stf_stackable_push_while(ctx);
  REQUIRE(scope != nullptr);

  // The conditional handle is observable as a uint64_t; just sanity-check it.
  REQUIRE(stf_while_scope_get_cond_handle(scope) != 0);

  stf_stackable_pop_while(scope);

  stf_stackable_ctx_finalize(ctx);
}

// Regression test mirroring probe_k_sweep.py: inside a while-scope body, chain
// K tasks that each do .rw() on the same persistent logical data, and make the
// loop execute exactly once. Sweep K=1..16 and expect every element of the
// accumulator to equal K. The equivalent Python probe fails deterministically
// when K is a multiple of 4 (drops exactly one update), so this test pins down
// whether the bug is in the C-API task path or somewhere above it.
C2H_TEST("stackable: while-body K chained rw tasks sweep", "[stackable][while][c-api]")
{
  const int Nd            = 128;
  const double tol_eps    = 1e-10;
  int total_mismatches    = 0;
  double total_off_by_one = 0.0;

  for (int K = 1; K <= 16; ++K)
  {
    stf_ctx_handle ctx = stf_stackable_ctx_create();
    REQUIRE(ctx != nullptr);

    // Accumulator: zero-initialized double[Nd].
    double* host_acc;
    cudaMallocHost(&host_acc, Nd * sizeof(double));
    for (int i = 0; i < Nd; i++)
    {
      host_acc[i] = 0.0;
    }
    stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_acc, Nd * sizeof(double));
    REQUIRE(lA != nullptr);

    // "done" flag: starts at 1.0, body drives it to 0.0 so while stops after 1
    // iteration. We use a double scalar to keep it consistent with the kernel
    // family used by the probe.
    double* host_done;
    cudaMallocHost(&host_done, sizeof(double));
    host_done[0]               = 1.0;
    stf_logical_data_handle lD = stf_stackable_logical_data(ctx, host_done, sizeof(double));
    REQUIRE(lD != nullptr);

    stf_while_scope_handle scope = stf_stackable_push_while(ctx);
    REQUIRE(scope != nullptr);
    {
      // K chained increments on lA, using the C-API raw task path that the
      // Python binding also uses.
      for (int k = 0; k < K; ++k)
      {
        stf_task_handle t = stf_stackable_task_create(ctx);
        REQUIRE(t != nullptr);
        stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
        stf_task_enable_capture(t);
        stf_task_start(t);
        double* d = static_cast<double*>(stf_task_get(t, 0));
        increment_kernel<<<1, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(Nd, d);
        stf_task_end(t);
        stf_task_destroy(t);
      }

      // Drive the done flag to 0.0 so the loop stops after 1 iteration.
      {
        stf_task_handle t = stf_stackable_task_create(ctx);
        REQUIRE(t != nullptr);
        stf_stackable_task_add_dep(ctx, t, lD, STF_WRITE);
        stf_task_enable_capture(t);
        stf_task_start(t);
        double* d = static_cast<double*>(stf_task_get(t, 0));
        scale_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>(1, d, 0.0);
        stf_task_end(t);
        stf_task_destroy(t);
      }

      // Continue while done > 0.5 (i.e. stop after we've zeroed it).
      stf_stackable_while_cond_scalar(ctx, scope, lD, STF_CMP_GT, 0.5, STF_DTYPE_FLOAT64);
    }
    stf_stackable_pop_while(scope);

    stf_stackable_logical_data_destroy(lA);
    stf_stackable_logical_data_destroy(lD);
    stf_stackable_ctx_finalize(ctx);

    const double expected = static_cast<double>(K);
    int mismatches        = 0;
    for (int i = 0; i < Nd; i++)
    {
      if (std::fabs(host_acc[i] - expected) > tol_eps)
      {
        ++mismatches;
      }
    }
    if (mismatches != 0)
    {
      fprintf(stderr,
              "[C-API K=%d] host_acc[0]=%g expected=%g (%d/%d mismatches)\n",
              K,
              host_acc[0],
              expected,
              mismatches,
              Nd);
      total_mismatches += mismatches;
      total_off_by_one += host_acc[0] - expected;
    }

    cudaFreeHost(host_acc);
    cudaFreeHost(host_done);
  }

  REQUIRE(total_mismatches == 0);
  (void) total_off_by_one;
}

#endif // _CCCL_CTK_AT_LEAST(12, 4)
