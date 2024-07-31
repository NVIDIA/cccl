//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: nvrtc

#include <cuda/pipeline>
#include <cuda/std/type_traits>

// comes last to take the ABI version
#include <cuda_pipeline.h>

#include "cuda_space_selector.h"
#include "test_macros.h"
#include <cooperative_groups/memcpy_async.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)
TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

constexpr int nthreads        = 256;
constexpr size_t stages_count = 2; // Pipeline with two stages

// Simply copy shared memory to global out
__device__ __forceinline__ void compute(int* global_out, int const* shared_in)
{
  auto block = cooperative_groups::this_thread_block();
  for (int i = 0; i < static_cast<int>(block.size()); ++i)
  {
    global_out[i] = shared_in[i];
  }
}

__global__ void with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
  auto grid  = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

  // Two batches must fit in shared memory:
  constexpr int smem_size = stages_count * nthreads;
  __shared__ int shared[smem_size];
  size_t shared_offset[stages_count] = {0, block.size()}; // Offsets to each batch

  // Allocate shared storage for a two-stage cuda::pipeline:
  using pipeline_state = cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count>;
  __shared__ pipeline_state* shared_state;
  shared_memory_selector<pipeline_state, constructor_initializer> sel;
  shared_state  = sel.construct();
  auto pipeline = cuda::make_pipeline(block, shared_state);

  // Each thread processes `batch_sz` elements.
  // Compute offset of the batch `batch` of this thread block in global memory:
  auto block_batch = [&](size_t batch) -> int {
    return block.group_index().x * block.size() + grid.size() * batch;
  };

  // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for the block:
  if (batch_sz == 0)
  {
    return;
  }
  pipeline.producer_acquire();
  cuda::memcpy_async(block, shared + shared_offset[0], global_in + block_batch(0), sizeof(int) * block.size(), pipeline);
  pipeline.producer_commit();
  // Pipelined copy/compute:
  for (size_t batch = 1; batch < batch_sz; ++batch)
  {
    // Stage indices for the compute and copy stages:
    size_t compute_stage_idx = (batch - 1) % 2;
    size_t copy_stage_idx    = batch % 2;

    // This change fixes an unrelated bug. The global_idx that was passed to
    // the compute stage was wrong.
    size_t global_copy_idx    = block_batch(batch);
    size_t global_compute_idx = block_batch(batch - 1);

    // Collectively acquire the pipeline head stage from all producer threads:
    pipeline.producer_acquire();

    // Submit async copies to the pipeline's head stage to be
    // computed in the next loop iteration
    cuda::memcpy_async(
      block, shared + shared_offset[copy_stage_idx], global_in + global_copy_idx, sizeof(int) * block.size(), pipeline);

    // Collectively commit (advance) the pipeline's head stage
    pipeline.producer_commit();

    // Collectively wait for the operations commited to the
    // previous `compute` stage to complete:
    pipeline.consumer_wait();

    // Computation overlapped with the memcpy_async of the "copy" stage:
    compute(global_out + global_compute_idx, shared + shared_offset[compute_stage_idx]);
    // Diverge threads in block
    __nanosleep(block.thread_rank() * 10);
    // Collectively release the stage resources
    pipeline.consumer_release();
  }

  // Compute the data fetch by the last iteration
  pipeline.consumer_wait();
  compute(global_out + block_batch(batch_sz - 1), shared + shared_offset[(batch_sz - 1) % 2]);
  pipeline.consumer_release();
}

int main(int argc, char** argv)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (
      constexpr int batch_size = 10; constexpr size_t size = batch_size * nthreads; int* in, *out;

      if (cudaMallocManaged((void**) &in, sizeof(int) * size) != cudaSuccess
          || cudaMallocManaged((void**) &out, sizeof(int) * size) != cudaSuccess) {
        printf("Setup failed\n");
        return -1;
      }

      for (size_t i = 0; i < size; ++i) { in[i] = (int) i; }

      with_staging<<<1, nthreads>>>(out, in, size, batch_size);
      {
        auto result = cudaGetLastError();
        if (result != cudaSuccess)
        {
          printf("Kernel launch failed\n");
          printf("Error: %s\n", cudaGetErrorString(result));
          return -1;
        }
      }

      if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("Kernel failed while running\n");
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
      }

      const int max_errors = 10;
      int num_errors       = 0;
      for (size_t i = 0; i < size; ++i) {
        if (out[i] != (int) i)
        {
          printf("out[%d] did not match expected value %d. Got %d\r\n", (int) i, (int) i, out[i]);
          num_errors++;
        }
        if (num_errors == max_errors)
        {
          break;
        }
      }

      if (num_errors == 0) { printf("No errors.\r\n"); } else { return -1; }

      return 0;),
    return 0;)
}
