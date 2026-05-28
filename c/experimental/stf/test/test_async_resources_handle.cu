//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Focused tests for stf_async_resources_create/destroy() exercised through
// stf_ctx_create_ex(). They cover the contract documented in stf.h:
//
//   * A shared stf_async_resources_handle can be reused across multiple
//     contexts created via stf_ctx_create_ex().
//   * When the contexts are created with `has_stream = 1`, stf_ctx_finalize()
//     is non-blocking: the caller must cudaStreamSynchronize(user_stream)
//     before destroying the shared handle.
//
// Both backends (STF_BACKEND_STREAM and STF_BACKEND_GRAPH) are exercised
// because the graph backend additionally benefits from the handle's
// executable-graph cache.

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
// Writes `value` into every slot of `arr`. The inner busy loop widens the
// kernel window so a missing chain dependency between back-to-back contexts
// becomes observable: a slow ctx1 kernel must finish before ctx2's kernel
// commits its value.
__global__ void slow_set_kernel(int* arr, int n, int value, int iters)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
  {
    return;
  }
  int acc = 0;
  for (int i = 0; i < iters; ++i)
  {
    acc += (i * 1103515245 + 12345) & 0x7fffffff;
  }
  arr[tid] = value + (acc & 0);
}

// Submit one slow_set task into `ctx`, writing `value` everywhere in `d_arr`.
void submit_set(stf_ctx_handle ctx, int* d_arr, int n, int value, int iters)
{
  stf_logical_data_handle tok = stf_token(ctx);
  REQUIRE(tok != nullptr);
  stf_logical_data_set_symbol(tok, "tok");

  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_symbol(t, "slow_set");
  stf_task_add_dep(t, tok, STF_RW);
  stf_task_start(t);

  CUstream s = stf_task_get_custream(t);
  REQUIRE(s != nullptr);

  const int threads = 128;
  const int blocks  = (n + threads - 1) / threads;
  slow_set_kernel<<<blocks, threads, 0, (cudaStream_t) s>>>(d_arr, n, value, iters);

  stf_task_end(t);
  stf_task_destroy(t);

  stf_logical_data_destroy(tok);
}

// Run one ctx (created via stf_ctx_create_ex with a caller-provided stream
// and a shared async_resources handle) that issues a single slow_set task.
void run_ctx_with_handle(
  stf_backend_kind backend, cudaStream_t s, stf_async_resources_handle h, int* d_arr, int N, int value, int iters)
{
  stf_ctx_options opts{};
  opts.backend    = backend;
  opts.has_stream = 1;
  opts.stream     = s;
  opts.handle     = h;

  stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
  REQUIRE(ctx != nullptr);

  submit_set(ctx, d_arr, N, value, iters);

  // Non-blocking: this enqueues the remaining work and the resource-release
  // callback on `s`; it does not synchronize `s`.
  stf_ctx_finalize(ctx);
}

// Run a back-to-back ordering experiment: two contexts share a handle and a
// caller stream, write distinct values, and the final buffer must reflect
// the second context's value (ctx2-after-ctx1 ordering via the caller
// stream). Iterating amplifies any missed dependency.
void check_back_to_back_ordering(stf_backend_kind backend)
{
  constexpr int N     = 1 << 14;
  constexpr int ITERS = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  stf_async_resources_handle h = stf_async_resources_create();
  REQUIRE(h != nullptr);

  for (int iter = 0; iter < 20; ++iter)
  {
    run_ctx_with_handle(backend, s, h, d_arr, N, /*value=*/1, ITERS);
    run_ctx_with_handle(backend, s, h, d_arr, N, /*value=*/2, ITERS);

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    int h_arr[16]{};
    REQUIRE(cudaMemcpy(h_arr, d_arr, sizeof(h_arr), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < static_cast<int>(sizeof(h_arr) / sizeof(int)); ++i)
    {
      INFO("iter=" << iter << " i=" << i << " value=" << h_arr[i]);
      REQUIRE(h_arr[i] == 2);
    }
  }

  // Required before destroying `h`: stf_ctx_finalize() left resource-release
  // callbacks enqueued on `s`. The destroy call tears down the underlying
  // CUDA resources synchronously.
  REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
  stf_async_resources_destroy(h);

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}
} // namespace

C2H_TEST("stf_async_resources_handle: shared across back-to-back stream contexts on user stream",
         "[context][stream][async_resources_handle]")
{
  check_back_to_back_ordering(STF_BACKEND_STREAM);
}

C2H_TEST("stf_async_resources_handle: shared across back-to-back graph contexts on user stream",
         "[context][graph][async_resources_handle]")
{
  check_back_to_back_ordering(STF_BACKEND_GRAPH);
}

// Smoke check of the handle's lifetime API independent of any context:
//   * NULL is a no-op for stf_async_resources_destroy().
//   * Create/destroy without ever attaching the handle to a context works.
//   * Destroying the handle before having submitted any work via a context
//     (only after a context was created and finalized without tasks) is
//     safe.
C2H_TEST("stf_async_resources_handle: lifetime smoke (no work, NULL destroy)",
         "[context][async_resources_handle][lifetime]")
{
  stf_async_resources_destroy(nullptr);

  stf_async_resources_handle h = stf_async_resources_create();
  REQUIRE(h != nullptr);
  stf_async_resources_destroy(h);

  // Empty stream context bound to a user stream + handle, no submitted work.
  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  stf_async_resources_handle h2 = stf_async_resources_create();
  REQUIRE(h2 != nullptr);
  {
    stf_ctx_options opts{};
    opts.backend    = STF_BACKEND_STREAM;
    opts.has_stream = 1;
    opts.stream     = s;
    opts.handle     = h2;
    stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
    REQUIRE(ctx != nullptr);
    stf_ctx_finalize(ctx);
  }
  // Even without submitted tasks the finalize is non-blocking when the
  // context was created with `has_stream = 1`. Synchronize before destroy.
  REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
  stf_async_resources_destroy(h2);

  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}
