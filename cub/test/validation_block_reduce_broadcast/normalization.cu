// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce.cuh>

#include <cuda/functional>
#include <cuda/std/functional>

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

template <int BlockThreads, bool Broadcast>
__global__ void normalize_kernel(const float* input, float* output, int columns)
{
  using block_reduce_t = cub::BlockReduce<float, BlockThreads>;
  __shared__ typename block_reduce_t::TempStorage max_storage;
  __shared__ typename block_reduce_t::TempStorage sum_storage;
  __shared__ float max_slot, sum_slot;
  const auto row = static_cast<std::int64_t>(blockIdx.x) * columns;
  float maximum  = -INFINITY;
  for (int column = threadIdx.x; column < columns; column += BlockThreads)
  {
    maximum = fmaxf(maximum, input[row + column]);
  }
  if constexpr (Broadcast)
  {
    maximum = block_reduce_t(max_storage).ReduceBroadcast(maximum, cuda::maximum<>{});
  }
  else
  {
    maximum = block_reduce_t(max_storage).Reduce(maximum, cuda::maximum<>{});
    if (threadIdx.x == 0)
    {
      max_slot = maximum;
    }
    __syncthreads();
    maximum = max_slot;
  }
  float sum = 0;
  for (int column = threadIdx.x; column < columns; column += BlockThreads)
  {
    sum += expf(input[row + column] - maximum);
  }
  if constexpr (Broadcast)
  {
    sum = block_reduce_t(sum_storage).ReduceBroadcast(sum, cuda::std::plus<>{});
  }
  else
  {
    sum = block_reduce_t(sum_storage).Sum(sum);
    if (threadIdx.x == 0)
    {
      sum_slot = sum;
    }
    __syncthreads();
    sum = sum_slot;
  }
  for (int column = threadIdx.x; column < columns; column += BlockThreads)
  {
    output[row + column] = expf(input[row + column] - maximum) / sum;
  }
}

template <int BlockThreads>
__global__ void first_thread_kernel(const float* input, float* output)
{
  using block_reduce_t = cub::BlockReduce<float, BlockThreads>;
  __shared__ typename block_reduce_t::TempStorage storage;
  const float value = input[static_cast<std::int64_t>(blockIdx.x) * BlockThreads + threadIdx.x];
  const float sum   = block_reduce_t(storage).Sum(value);
  __syncthreads();
  const float maximum = block_reduce_t(storage).Reduce(value, cuda::maximum<>{});
  if (threadIdx.x == 0)
  {
    output[2 * blockIdx.x]     = sum;
    output[2 * blockIdx.x + 1] = maximum;
  }
}

template <int BlockThreads>
void launch(const at::Tensor& input, at::Tensor& output, bool first_thread, bool broadcast, cudaStream_t stream)
{
  const int columns = static_cast<int>(input.size(-1));
  const auto rows   = input.numel() / columns;
  if (first_thread)
  {
    first_thread_kernel<BlockThreads>
      <<<rows, BlockThreads, 0, stream>>>(input.data_ptr<float>(), output.data_ptr<float>());
  }
  else
  {
#if USE_BROADCAST
    if (broadcast)
    {
      normalize_kernel<BlockThreads, true>
        <<<rows, BlockThreads, 0, stream>>>(input.data_ptr<float>(), output.data_ptr<float>(), columns);
      return;
    }
#endif
    normalize_kernel<BlockThreads, false>
      <<<rows, BlockThreads, 0, stream>>>(input.data_ptr<float>(), output.data_ptr<float>(), columns);
  }
}

at::Tensor normalize(const at::Tensor& input, bool first_thread, bool broadcast)
{
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.is_contiguous(),
              "Expected contiguous CUDA float32 input");
  TORCH_CHECK(input.dim() >= 2 && input.size(-1) > 0, "Expected nonempty rows");
  const c10::cuda::CUDAGuard guard(input.device());
  const int columns = static_cast<int>(input.size(-1));
  TORCH_CHECK(!first_thread || columns == 32 || columns == 128 || columns == 256 || columns == 512,
              "First-thread control requires one element per thread");
  at::Tensor output = first_thread ? at::empty({input.numel() / columns, 2}, input.options()) : at::empty_like(input);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (columns <= 32)
  {
    launch<32>(input, output, first_thread, broadcast, stream);
  }
  else if (columns <= 128)
  {
    launch<128>(input, output, first_thread, broadcast, stream);
  }
  else if (columns <= 256)
  {
    launch<256>(input, output, first_thread, broadcast, stream);
  }
  else
  {
    launch<512>(input, output, first_thread, broadcast, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
