//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// Regression test for the C-facade stackable-token dispatch fix. Combining a
// stackable token with push_graph / pop_prologue used to abort inside STF with
// a "Data interface type mismatch" (assumed void_interface, actual
// mdspan<char, ..., layout_stride>) because the C API treated every stackable
// logical-data handle as a slice<char> and mis-cast tokens. The abort was a
// hard C-level abort, so the Python binding that drives this exact sequence
// could not catch it:
//
//     ctx = stf.stackable_context()
//     tok = ctx.token()
//     ctx.push()
//     with ctx.task(tok.write()): ...
//     with ctx.task(tok.read()):  ...
//     step_graph = ctx.pop_prologue_shared()
//
// These tests drive the same sequences through the C stackable API directly,
// so the path stays covered without any Python / Warp in the picture.
//
// The existing `stackable: token + fence` test uses tokens but outside any
// push_graph scope, and the existing pop_prologue tests use real logical_data;
// so the combination "tokens inside push_graph" is only exercised here.

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
__global__ void noop_kernel() {}
} // namespace

// Minimal case: a single token-only task inside a push_graph / pop scope.
// Does NOT use pop_prologue — just push_graph + pop — so token task-deps
// handling is covered independently of the prologue machinery.
C2H_TEST("stackable: token in push_graph scope (no prologue)", "[stackable][token][bug]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  stf_logical_data_handle tok = stf_stackable_token(ctx);
  REQUIRE(tok != nullptr);

  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, tok, STF_WRITE);
    stf_task_start(t);
    noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_stackable_pop(ctx);

  stf_stackable_token_destroy(tok);
  stf_stackable_ctx_finalize(ctx);
}

// Exact mirror of the Python run_stf_unified path:
//   ctx.push() -> task(tok.write()) -> task(tok.read()) -> pop_prologue(_shared)
// This used to abort inside pop_prologue(_shared)() with the void_interface vs
// mdspan<char> mismatch before the C-facade dispatch fix.
C2H_TEST("stackable: token write/read chain + pop_prologue", "[stackable][token][launchable][bug]")
{
  const int relaunchN = 4;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  stf_logical_data_handle tok = stf_stackable_token(ctx);
  REQUIRE(tok != nullptr);

  stf_stackable_push_graph(ctx);
  {
    // Writer task (equivalent of Python `tok.write()`).
    {
      stf_task_handle t = stf_stackable_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_stackable_task_add_dep(ctx, t, tok, STF_WRITE);
      stf_task_enable_capture(t);
      stf_task_start(t);
      noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
      stf_task_end(t);
      stf_task_destroy(t);
    }

    // Reader task (equivalent of Python `tok.read()`).
    {
      stf_task_handle t = stf_stackable_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_stackable_task_add_dep(ctx, t, tok, STF_READ);
      stf_task_enable_capture(t);
      stf_task_start(t);
      noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
      stf_task_end(t);
      stf_task_destroy(t);
    }
  }

  stf_launchable_graph_handle lh = stf_stackable_pop_prologue(ctx);
  REQUIRE(lh != nullptr);
  for (int k = 0; k < relaunchN; ++k)
  {
    stf_launchable_graph_launch(lh);
  }
  stf_stackable_pop_epilogue(ctx);
  stf_launchable_graph_destroy(lh);

  stf_stackable_token_destroy(tok);
  stf_stackable_ctx_finalize(ctx);
}

// Same as above but using the shared flavour of pop_prologue, which is what
// the Python `pop_prologue_shared()` binding calls into.
C2H_TEST("stackable: token write/read chain + pop_prologue_shared", "[stackable][token][launchable][bug]")
{
  const int relaunchN = 4;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  stf_logical_data_handle tok = stf_stackable_token(ctx);
  REQUIRE(tok != nullptr);

  stf_stackable_push_graph(ctx);
  {
    {
      stf_task_handle t = stf_stackable_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_stackable_task_add_dep(ctx, t, tok, STF_WRITE);
      stf_task_enable_capture(t);
      stf_task_start(t);
      noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
      stf_task_end(t);
      stf_task_destroy(t);
    }
    {
      stf_task_handle t = stf_stackable_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_stackable_task_add_dep(ctx, t, tok, STF_READ);
      stf_task_enable_capture(t);
      stf_task_start(t);
      noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
      stf_task_end(t);
      stf_task_destroy(t);
    }
  }

  stf_launchable_graph_shared h = nullptr;
  REQUIRE(stf_stackable_pop_prologue_shared(ctx, &h) == 0);
  REQUIRE(h != nullptr);
  for (int k = 0; k < relaunchN; ++k)
  {
    stf_launchable_graph_shared_launch(h);
  }
  // Last free drops the strong ref and runs pop_epilogue automatically.
  stf_launchable_graph_shared_free(h);

  stf_stackable_token_destroy(tok);
  stf_stackable_ctx_finalize(ctx);
}

// Sanity check: replacing the token with a real logical_data in the same
// push_graph + pop_prologue shape *should* work. This matches the
// `run_stf_unified_ld` workaround that the Python mockup confirmed OK.
C2H_TEST("stackable: logical_data write/read chain + pop_prologue (workaround)", "[stackable][launchable]")
{
  const size_t N      = 8;
  const int relaunchN = 4;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  uint8_t* host_dep;
  cudaMallocHost(&host_dep, N * sizeof(uint8_t));
  for (size_t i = 0; i < N; i++)
  {
    host_dep[i] = 0;
  }

  stf_logical_data_handle ld = stf_stackable_logical_data(ctx, host_dep, N * sizeof(uint8_t));
  REQUIRE(ld != nullptr);

  stf_stackable_push_graph(ctx);
  {
    {
      stf_task_handle t = stf_stackable_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_stackable_task_add_dep(ctx, t, ld, STF_RW);
      stf_task_enable_capture(t);
      stf_task_start(t);
      noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
      stf_task_end(t);
      stf_task_destroy(t);
    }
    {
      stf_task_handle t = stf_stackable_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_stackable_task_add_dep(ctx, t, ld, STF_READ);
      stf_task_enable_capture(t);
      stf_task_start(t);
      noop_kernel<<<1, 1, 0, (cudaStream_t) stf_task_get_custream(t)>>>();
      stf_task_end(t);
      stf_task_destroy(t);
    }
  }

  stf_launchable_graph_handle lh = stf_stackable_pop_prologue(ctx);
  REQUIRE(lh != nullptr);
  for (int k = 0; k < relaunchN; ++k)
  {
    stf_launchable_graph_launch(lh);
  }
  stf_stackable_pop_epilogue(ctx);
  stf_launchable_graph_destroy(lh);

  stf_stackable_logical_data_destroy(ld);
  stf_stackable_ctx_finalize(ctx);

  cudaFreeHost(host_dep);
}
