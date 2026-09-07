//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Minimal tests for stf_ctx_create_ex() with has_stream=1 (caller-provided
// CUDA stream) on the stream backend, with no async_resources handle shared
// across contexts. These verify that contexts created back-to-back on the same
// caller stream chain their work transitively through that stream.

#include <vector>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
// A device sink that is written but never read. Publishing the busy-loop
// result here gives the loop an observable side effect, so the compiler
// cannot optimize it away, without perturbing the result buffer.
__device__ unsigned g_busy_sink;

// Writes `value` into every slot of `arr`. The inner busy loop widens the
// kernel window so that a failure to chain ctx2-after-ctx1 is observable:
// ctx1 is still running when ctx2's kernel races in.
__global__ void slow_set_kernel(int* arr, int n, int value, int iters)
{
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n)
  {
    return;
  }
  // Busy loop to keep the kernel resident on the SM for a while. `acc` is
  // unsigned so the accumulation wraps with well-defined behavior.
  unsigned acc = 0;
  for (int i = 0; i < iters; ++i)
  {
    acc += (static_cast<unsigned>(i) * 1103515245u + 12345u) & 0x7fffffffu;
  }
  // Publish `acc` via an atomic: an observable, race-free side effect that
  // keeps the loop alive while the stored result stays exactly `value`.
  atomicAdd(&g_busy_sink, acc);
  arr[tid] = value;
}

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

// Submits `K` concurrent token-tasks in a single context; each writes `value`
// into its own slice of `d_arr`. Multiple independent tokens per context make
// STF spread kernels across several pool streams, so ordering depends on the
// caller-stream chaining contract.
void run_ctx_k_concurrent(cudaStream_t s, int* d_arr, int N, int K, int value, int iters)
{
  stf_ctx_options opts{};
  opts.backend    = STF_BACKEND_STREAM;
  opts.has_stream = 1;
  opts.stream     = s;
  opts.handle     = nullptr;

  stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
  REQUIRE(ctx != nullptr);

  const int per = N / K;
  for (int k = 0; k < K; ++k)
  {
    stf_logical_data_handle tok = stf_token(ctx);
    REQUIRE(tok != nullptr);

    stf_task_handle t = stf_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_task_add_dep(t, tok, STF_RW);
    stf_task_start(t);

    CUstream ts       = stf_task_get_custream(t);
    const int threads = 128;
    const int blocks  = (per + threads - 1) / threads;
    int* slice        = d_arr + k * per;
    slow_set_kernel<<<blocks, threads, 0, (cudaStream_t) ts>>>(slice, per, value, iters);

    stf_task_end(t);
    stf_task_destroy(t);
    stf_logical_data_destroy(tok);
  }

  stf_ctx_finalize(ctx);
}

// More faithful MLP mimic: K concurrent tokens, each with T chained tasks
// (sequential RW on the same token), so each token effectively owns a chain of
// T slow kernels on one pool stream.
void run_ctx_k_chains(cudaStream_t s, int* d_arr, int N, int K, int chain_len, int value, int iters)
{
  stf_ctx_options opts{};
  opts.backend    = STF_BACKEND_STREAM;
  opts.has_stream = 1;
  opts.stream     = s;
  opts.handle     = nullptr;

  stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
  REQUIRE(ctx != nullptr);

  const int per = N / K;
  std::vector<stf_logical_data_handle> toks(K);
  for (int k = 0; k < K; ++k)
  {
    toks[k] = stf_token(ctx);
    REQUIRE(toks[k] != nullptr);
  }

  for (int step = 0; step < chain_len; ++step)
  {
    for (int k = 0; k < K; ++k)
    {
      stf_task_handle t = stf_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_task_add_dep(t, toks[k], STF_RW);
      stf_task_start(t);

      CUstream ts       = stf_task_get_custream(t);
      const int threads = 128;
      const int blocks  = (per + threads - 1) / threads;
      int* slice        = d_arr + k * per;
      slow_set_kernel<<<blocks, threads, 0, (cudaStream_t) ts>>>(slice, per, value, iters);

      stf_task_end(t);
      stf_task_destroy(t);
    }
  }

  for (int k = 0; k < K; ++k)
  {
    stf_logical_data_destroy(toks[k]);
  }

  stf_ctx_finalize(ctx);
}
} // namespace

C2H_TEST("stf_ctx_create_ex: 1 token per context, back-to-back, stream-only", "[context][stream]")
{
  constexpr int N     = 1 << 14;
  constexpr int ITERS = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  for (int iter = 0; iter < 20; ++iter)
  {
    {
      stf_ctx_options opts{};
      opts.backend    = STF_BACKEND_STREAM;
      opts.has_stream = 1;
      opts.stream     = s;
      opts.handle     = nullptr;

      stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
      REQUIRE(ctx != nullptr);
      submit_set(ctx, d_arr, N, /*value=*/1, ITERS);
      stf_ctx_finalize(ctx);
    }
    {
      stf_ctx_options opts{};
      opts.backend    = STF_BACKEND_STREAM;
      opts.has_stream = 1;
      opts.stream     = s;
      opts.handle     = nullptr;

      stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
      REQUIRE(ctx != nullptr);
      submit_set(ctx, d_arr, N, /*value=*/2, ITERS);
      stf_ctx_finalize(ctx);
    }

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    int h_arr[16]{};
    REQUIRE(cudaMemcpy(h_arr, d_arr, sizeof(h_arr), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < static_cast<int>(sizeof(h_arr) / sizeof(int)); ++i)
    {
      INFO("iter=" << iter << " i=" << i << " value=" << h_arr[i]);
      REQUIRE(h_arr[i] == 2);
    }
  }

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}

C2H_TEST("stf_ctx_create_ex: K chains of T tasks per token, back-to-back, stream-only, no handle",
         "[context][stream][tokens][lifetime]")
{
  constexpr int N         = 1 << 16;
  constexpr int K         = 8;
  constexpr int CHAIN_LEN = 20;
  constexpr int ITERS     = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  for (int iter = 0; iter < 20; ++iter)
  {
    run_ctx_k_chains(s, d_arr, N, K, CHAIN_LEN, /*value=*/1, ITERS);
    run_ctx_k_chains(s, d_arr, N, K, CHAIN_LEN, /*value=*/2, ITERS);

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    std::vector<int> h_arr(N, 0);
    REQUIRE(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    int mismatches  = 0;
    int first_bad_i = -1;
    int first_bad_v = 0;
    for (int i = 0; i < N; ++i)
    {
      if (h_arr[i] != 2)
      {
        ++mismatches;
        if (first_bad_i < 0)
        {
          first_bad_i = i;
          first_bad_v = h_arr[i];
        }
      }
    }
    INFO("iter=" << iter << " mismatches=" << mismatches << " first_bad_idx=" << first_bad_i
                 << " first_bad_val=" << first_bad_v);
    REQUIRE(mismatches == 0);
  }

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}

C2H_TEST("stf_ctx_create_ex: K concurrent tokens per context, back-to-back, stream-only", "[context][stream][tokens]")
{
  constexpr int N     = 1 << 16;
  constexpr int K     = 8;
  constexpr int ITERS = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  for (int iter = 0; iter < 20; ++iter)
  {
    run_ctx_k_concurrent(s, d_arr, N, K, /*value=*/1, ITERS);
    run_ctx_k_concurrent(s, d_arr, N, K, /*value=*/2, ITERS);

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    std::vector<int> h_arr(N, 0);
    REQUIRE(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    int mismatches  = 0;
    int first_bad_i = -1;
    int first_bad_v = 0;
    for (int i = 0; i < N; ++i)
    {
      if (h_arr[i] != 2)
      {
        ++mismatches;
        if (first_bad_i < 0)
        {
          first_bad_i = i;
          first_bad_v = h_arr[i];
        }
      }
    }
    INFO("iter=" << iter << " mismatches=" << mismatches << " first_bad_idx=" << first_bad_i
                 << " first_bad_val=" << first_bad_v);
    REQUIRE(mismatches == 0);
  }

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}
