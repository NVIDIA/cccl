//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Ensure back-to-back stream_ctx instances on a caller stream are ordered.
 *
 * The test submits two stream_ctx instances back-to-back on the same
 * caller-provided stream, without an explicit synchronization between them.
 * Each context launches independent token chains on STF pool streams. The
 * second context writes value 2 into the same buffer written by the first
 * context, so observing any value other than 2 means the contexts were not
 * chained through the caller stream correctly.
 *
 * The explicit-sync and shared-handle variants exercise the same shape through
 * the two configurations that were already known to be safe.
 */

#include <cuda/experimental/stf.cuh>

#include <vector>

#include <cuda_runtime.h>

using namespace cuda::experimental::stf;

namespace
{
constexpr int N                 = 1 << 14;
constexpr int CHAIN_COUNT       = 8;
constexpr int CHAIN_LEN         = 40;
constexpr int OUTER             = 5;
constexpr long long BUSY_CYCLES = 5'000'000;

__global__ void slow_set_kernel(int* slice, int n, int value, long long ns)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
  {
    return;
  }
  const long long start = clock64();
  while (clock64() - start < ns)
  {
    // busy wait
  }
  slice[tid] = value;
}

void submit_token_chains(stream_ctx& ctx, int* d_arr, int value)
{
  std::vector<logical_data<void_interface>> toks;
  toks.reserve(CHAIN_COUNT);
  for (int k = 0; k < CHAIN_COUNT; ++k)
  {
    toks.push_back(ctx.token());
  }

  const int per_chain = N / CHAIN_COUNT;

  for (int step = 0; step < CHAIN_LEN; ++step)
  {
    for (int k = 0; k < CHAIN_COUNT; ++k)
    {
      int* slice = d_arr + k * per_chain;
      ctx.task(toks[k].rw())->*[=](cudaStream_t ts) {
        const int threads = 128;
        const int blocks  = (per_chain + threads - 1) / threads;
        slow_set_kernel<<<blocks, threads, 0, ts>>>(slice, per_chain, value, BUSY_CYCLES);
      };
    }
  }
}

bool has_mismatch(const std::vector<int>& values, int expected)
{
  return ::std::any_of(values.begin(), values.end(), [=](int x) { return x != expected; });
}

void validate_buffer(int* d_arr)
{
  std::vector<int> h_arr(N, 0);
  cuda_safe_call(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT(!has_mismatch(h_arr, 2));
}

void run_no_handle_no_sync_once()
{
  cudaStream_t stream{};
  cuda_safe_call(cudaStreamCreate(&stream));

  int* d_arr = nullptr;
  {
    cuda_safe_call(cudaMalloc(&d_arr, N * sizeof(int)));
    cuda_safe_call(cudaMemsetAsync(d_arr, 0, N * sizeof(int), stream));
  }

  {
    stream_ctx ctx(stream);
    submit_token_chains(ctx, d_arr, 1);
    ctx.finalize();
  }
  {
    stream_ctx ctx(stream);
    submit_token_chains(ctx, d_arr, 2);
    ctx.finalize();
  }

  cuda_safe_call(cudaStreamSynchronize(stream));
  validate_buffer(d_arr);

  cuda_safe_call(cudaFree(d_arr));
  cuda_safe_call(cudaStreamDestroy(stream));
}

void run_no_handle_sync_once()
{
  cudaStream_t stream{};
  cuda_safe_call(cudaStreamCreate(&stream));

  int* d_arr = nullptr;
  {
    cuda_safe_call(cudaMalloc(&d_arr, N * sizeof(int)));
    cuda_safe_call(cudaMemsetAsync(d_arr, 0, N * sizeof(int), stream));
  }

  {
    stream_ctx ctx(stream);
    submit_token_chains(ctx, d_arr, 1);
    ctx.finalize();
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  {
    stream_ctx ctx(stream);
    submit_token_chains(ctx, d_arr, 2);
    ctx.finalize();
  }

  cuda_safe_call(cudaStreamSynchronize(stream));
  validate_buffer(d_arr);

  cuda_safe_call(cudaFree(d_arr));
  cuda_safe_call(cudaStreamDestroy(stream));
}

void run_shared_handle_no_sync_once()
{
  cudaStream_t stream{};
  cuda_safe_call(cudaStreamCreate(&stream));

  int* d_arr = nullptr;
  {
    cuda_safe_call(cudaMalloc(&d_arr, N * sizeof(int)));
    cuda_safe_call(cudaMemsetAsync(d_arr, 0, N * sizeof(int), stream));
  }

  async_resources_handle handle;
  {
    stream_ctx ctx(stream, handle);
    submit_token_chains(ctx, d_arr, 1);
    ctx.finalize();
  }
  {
    stream_ctx ctx(stream, handle);
    submit_token_chains(ctx, d_arr, 2);
    ctx.finalize();
  }

  cuda_safe_call(cudaStreamSynchronize(stream));
  validate_buffer(d_arr);

  cuda_safe_call(cudaFree(d_arr));
  cuda_safe_call(cudaStreamDestroy(stream));
}

template <typename Test>
void repeat(Test test)
{
  for (int i = 0; i < OUTER; ++i)
  {
    test();
  }
}
} // namespace

int main()
{
  repeat(run_no_handle_no_sync_once);
  repeat(run_no_handle_sync_once);
  repeat(run_shared_handle_no_sync_once);
}
