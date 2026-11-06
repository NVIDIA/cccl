// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host-side cost of enqueuing the warp sum benchmark kernel with <<<>>>.
// Counterpart of launch_overhead.py. Prints one JSON object.
//
// Launches go out in batches small enough to stay below the launch queue
// depth, with a stream synchronize between batches that is not timed, so the
// measurement is host enqueue time rather than GPU back-pressure.

#include <cub/warp/warp_reduce.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include <device_side_benchmark.cuh>

struct sum_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using WarpReduce = cub::WarpReduce<T>;
    __shared__ typename WarpReduce::TempStorage temp_storage[32];
    return WarpReduce{temp_storage[threadIdx.x / 32]}.Sum(thread_data);
  }
};

int main()
{
  constexpr int block_size = 256;
  constexpr int batch      = 256;
  constexpr int batches    = 200;
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  benchmark_kernel<block_size, 1, sum_op_t, int, true><<<1, block_size, 0, stream>>>(sum_op_t{});
  cudaStreamSynchronize(stream);

  std::vector<double> per_launch_us;
  for (int b = 0; b < batches; ++b)
  {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < batch; ++i)
    {
      benchmark_kernel<block_size, 1, sum_op_t, int, true><<<1, block_size, 0, stream>>>(sum_op_t{});
    }
    const auto stop = std::chrono::steady_clock::now();
    cudaStreamSynchronize(stream);
    per_launch_us.push_back(std::chrono::duration<double, std::micro>(stop - start).count() / batch);
  }
  if (cudaGetLastError() != cudaSuccess)
  {
    std::fprintf(stderr, "launch failed\n");
    return 1;
  }
  std::sort(per_launch_us.begin(), per_launch_us.end());
  std::printf(
    "{\"side\": \"cpp\", \"launch\": \"<<<>>>\", \"batch\": %d, \"batches\": %d, "
    "\"median_us\": %.4f, \"min_us\": %.4f, \"max_us\": %.4f}\n",
    batch,
    batches,
    per_launch_us[batches / 2],
    per_launch_us.front(),
    per_launch_us.back());
  cudaStreamDestroy(stream);
  return 0;
}
