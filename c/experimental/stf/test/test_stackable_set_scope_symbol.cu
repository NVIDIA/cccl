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
// stf_stackable_set_scope_symbol names the current (innermost) scope of a
// stackable context. The name only affects DOT output and diagnostics, so these tests
// check that the call is accepted at the root scope and in nested scopes,
// before and after tasks, and that the computation is unchanged. The DOT
// label itself is checked by the Python test (python/cuda_stf/tests/stf/
// test_dot_symbols.py, every symbol-setting API against the DOT output), which
// runs the traced part in a subprocess because
// the DOT writer reads CUDASTF_DOT_FILE once per process.

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
__global__ void increment_kernel(int cnt, double* data)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt)
  {
    data[i] += 1.0;
  }
}

void increment_task(stf_ctx_handle ctx, stf_logical_data_handle lA, int n)
{
  stf_task_handle t = stf_stackable_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
  stf_task_enable_capture(t);
  stf_task_start(t);
  double* d = static_cast<double*>(stf_task_get(t, 0));
  increment_kernel<<<(n + 63) / 64, 64, 0, (cudaStream_t) stf_task_get_custream(t)>>>(n, d);
  stf_task_end(t);
  stf_task_destroy(t);
}
} // namespace

C2H_TEST("stackable: set_scope_symbol on the root scope", "[stackable][symbol]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);
  REQUIRE(stf_stackable_set_scope_symbol(ctx, "root") == STF_SUCCESS);

  stf_logical_data_handle tok = stf_stackable_token(ctx);
  REQUIRE(tok != nullptr);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, tok, STF_WRITE);
    stf_task_start(t);
    stf_task_end(t);
    stf_task_destroy(t);
  }
  // renaming after tasks were submitted is accepted too
  REQUIRE(stf_stackable_set_scope_symbol(ctx, "root-renamed") == STF_SUCCESS);

  stf_stackable_token_destroy(tok);
  stf_stackable_ctx_finalize(ctx);
}

C2H_TEST("stackable: set_scope_symbol names nested scopes", "[stackable][symbol]")
{
  const int N = 128;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  REQUIRE(cudaMallocHost(&host_data, N * sizeof(double)) == cudaSuccess);
  for (int i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  stf_stackable_push_graph(ctx);
  {
    REQUIRE(stf_stackable_set_scope_symbol(ctx, "outer") == STF_SUCCESS);
    increment_task(ctx, lA, N);

    stf_stackable_push_graph(ctx);
    {
      REQUIRE(stf_stackable_set_scope_symbol(ctx, "inner") == STF_SUCCESS);
      increment_task(ctx, lA, N);
    }
    stf_stackable_pop(ctx);

    // back in the outer scope: naming it again targets the outer scope, not the popped one
    REQUIRE(stf_stackable_set_scope_symbol(ctx, "outer-after-inner") == STF_SUCCESS);
    increment_task(ctx, lA, N);
  }
  stf_stackable_pop(ctx);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (int i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 3.0) < 1e-10);
  }
  REQUIRE(cudaFreeHost(host_data) == cudaSuccess);
}
