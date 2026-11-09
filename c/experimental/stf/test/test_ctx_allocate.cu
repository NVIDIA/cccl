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
// stf_ctx_allocate / stf_stackable_allocate: raw device memory served by a
// context's own allocators (cached or uncached). At the root of a stackable
// context and on a stream context the call is synchronous; inside a pushed
// scope the memory is owned by the scope (allocated on its support stream, no
// allocation node in the scope's graph, released at pop), and cached
// allocations are refused.

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
__global__ void fill_kernel(int n, float* out, float v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = v;
  }
}

__global__ void add_kernel(int n, const float* in, double* inout)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    inout[i] += in[i];
  }
}

constexpr int N = 128;

// Round trip through memory the context handed out: the pointer is usable on an
// arbitrary stream as soon as the call returns.
void check_round_trip(void* p)
{
  REQUIRE(p != nullptr);
  float host[N];
  for (int i = 0; i < N; i++)
  {
    host[i] = static_cast<float>(i);
  }
  REQUIRE(cudaMemcpy(p, host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
  float back[N] = {};
  REQUIRE(cudaMemcpy(back, p, sizeof(back), cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < N; i++)
  {
    REQUIRE(back[i] == static_cast<float>(i));
  }
}
} // namespace

C2H_TEST("ctx_allocate: stream context, cached and uncached", "[ctx][allocate]")
{
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  for (int cached = 0; cached < 2; cached++)
  {
    void* p = stf_ctx_allocate(ctx, N * sizeof(float), cached);
    check_round_trip(p);
    REQUIRE(stf_ctx_deallocate(ctx, p, N * sizeof(float), cached) == STF_SUCCESS);
  }

  // a cached block comes back on the next request of the same size
  void* a = stf_ctx_allocate(ctx, N * sizeof(float), 1);
  REQUIRE(a != nullptr);
  REQUIRE(stf_ctx_deallocate(ctx, a, N * sizeof(float), 1) == STF_SUCCESS);
  void* b = stf_ctx_allocate(ctx, N * sizeof(float), 1);
  REQUIRE(b == a);
  REQUIRE(stf_ctx_deallocate(ctx, b, N * sizeof(float), 1) == STF_SUCCESS);

  REQUIRE(stf_ctx_finalize(ctx) == STF_SUCCESS);
}

C2H_TEST("ctx_allocate: graph context is refused", "[ctx][allocate]")
{
  stf_ctx_handle ctx = stf_ctx_create_graph();
  REQUIRE(ctx != nullptr);
  REQUIRE(stf_ctx_allocate(ctx, 64, 0) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(stf_ctx_finalize(ctx) == STF_SUCCESS);
}

C2H_TEST("stackable_allocate: root behaves like the stream context", "[stackable][allocate]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);
  for (int cached = 0; cached < 2; cached++)
  {
    void* p = stf_stackable_allocate(ctx, N * sizeof(float), cached);
    check_round_trip(p);
    REQUIRE(stf_stackable_deallocate(ctx, p, N * sizeof(float), cached) == STF_SUCCESS);
  }
  REQUIRE(stf_stackable_ctx_finalize(ctx) == STF_SUCCESS);
}

C2H_TEST("stackable_allocate: scope-owned scratch used by the scope's tasks", "[stackable][allocate]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double host_data[N];
  for (int i = 0; i < N; i++)
  {
    host_data[i] = 1.0;
  }
  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  REQUIRE(stf_stackable_push_graph(ctx) == STF_SUCCESS);
  {
    // a library's temporary: allocated while the scope records, freed before the
    // scope ends; the scope keeps it alive until its graph has run
    void* scratch = stf_stackable_allocate(ctx, N * sizeof(float), 0);
    REQUIRE(scratch != nullptr);
    // cached allocations are refused inside a scope
    REQUIRE(stf_stackable_allocate(ctx, N * sizeof(float), 1) == nullptr);
    REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);

    stf_task_handle t1 = stf_stackable_task_create(ctx);
    REQUIRE(t1 != nullptr);
    stf_stackable_task_add_dep(ctx, t1, lA, STF_READ);
    stf_task_enable_capture(t1);
    stf_task_start(t1);
    fill_kernel<<<(N + 63) / 64, 64, 0, (cudaStream_t) stf_task_get_custream(t1)>>>(
      N, static_cast<float*>(scratch), 2.0f);
    stf_task_end(t1);
    stf_task_destroy(t1);

    stf_task_handle t2 = stf_stackable_task_create(ctx);
    REQUIRE(t2 != nullptr);
    stf_stackable_task_add_dep(ctx, t2, lA, STF_RW);
    stf_task_enable_capture(t2);
    stf_task_start(t2);
    double* d = static_cast<double*>(stf_task_get(t2, 0));
    add_kernel<<<(N + 63) / 64, 64, 0, (cudaStream_t) stf_task_get_custream(t2)>>>(
      N, static_cast<const float*>(scratch), d);
    stf_task_end(t2);
    stf_task_destroy(t2);

    REQUIRE(stf_stackable_deallocate(ctx, scratch, N * sizeof(float), 0) == STF_SUCCESS);
  }
  REQUIRE(stf_stackable_pop(ctx) == STF_SUCCESS);

  stf_stackable_logical_data_destroy(lA);
  REQUIRE(stf_stackable_ctx_finalize(ctx) == STF_SUCCESS);

  for (int i = 0; i < N; i++)
  {
    REQUIRE(host_data[i] == 3.0);
  }
}
