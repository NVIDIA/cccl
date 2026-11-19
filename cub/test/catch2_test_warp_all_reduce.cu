// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/util_ptx.cuh>
#include <cub/warp/warp_all_reduce.cuh>

#include <thrust/memory.h>

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cmath>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/vector.h>

template <int Width, bool Physical, bool Divergent>
__global__ void all_reduce_sum_kernel(const float* input, float* output, float* native)
{
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int group = static_cast<int>(threadIdx.x) / Width;
  float value     = input[index];
  float expected  = value;
  if (!Divergent || group % 2 == 0)
  {
    const unsigned mask = Divergent ? cub::WarpMask<Width>(group % (32 / Width)) : 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < Width; offset *= 2)
    {
      expected += __shfl_xor_sync(mask, expected, offset, Width);
    }
    if constexpr (Physical)
    {
      value = cub::detail::WarpAllReduce<float, Width, true>{}.Sum(value);
    }
    else
    {
      value = cub::detail::WarpAllReduce<float, Width>{}.Sum(value);
    }
  }
  output[index] = value;
  native[index] = expected;
}

template <int Width, bool Physical, bool Divergent>
void check_all_reduce_sum(int distribution)
{
  constexpr int threads = 128;
  constexpr int blocks  = 3;
  constexpr int count   = threads * blocks;
  c2h::host_vector<float> input(count);
  for (int i = 0; i < count; ++i)
  {
    float value = static_cast<float>((i * 37) % 101 - 50) / 17.0f;
    if (distribution == 1)
    {
      value = std::ldexp(value, i % 33 - 16);
    }
    else if (distribution == 2)
    {
      constexpr float cancellation[] = {1.e10f, 1.0f, -1.e10f, -1.0f};
      value                          = cancellation[i % 4];
    }
    else if (distribution == 3)
    {
      value = i % 2 ? -0.0f : 0.0f;
    }
    else if (distribution == 4)
    {
      value = (i % 2 ? -1.0f : 1.0f) * cuda::std::numeric_limits<float>::denorm_min() * static_cast<float>(i % 31);
    }
    input[i] = value;
  }
  const c2h::device_vector<float> device_input = input;
  c2h::device_vector<float> device_output(count, cuda::std::numeric_limits<float>::quiet_NaN());
  c2h::device_vector<float> device_native(count, cuda::std::numeric_limits<float>::quiet_NaN());
  all_reduce_sum_kernel<Width, Physical, Divergent><<<blocks, threads>>>(
    thrust::raw_pointer_cast(device_input.data()),
    thrust::raw_pointer_cast(device_output.data()),
    thrust::raw_pointer_cast(device_native.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  const c2h::host_vector<float> output = device_output;
  const c2h::host_vector<float> native = device_native;
  for (int first = 0; first < count; first += Width)
  {
    double sum       = 0;
    double magnitude = 0;
    for (int lane = 0; lane < Width; ++lane)
    {
      sum += input[first + lane];
      magnitude += std::abs(static_cast<double>(input[first + lane]));
    }
    const bool participates = !Divergent || (first / Width) % 2 == 0;
    const double bound      = magnitude * cuda::std::numeric_limits<float>::epsilon() * 5
                            + Width * static_cast<double>(cuda::std::numeric_limits<float>::denorm_min());
    for (int lane = 0; lane < Width; ++lane)
    {
      const int index       = first + lane;
      const double expected = participates ? sum : input[index];
      REQUIRE(cuda::std::bit_cast<cuda::std::uint32_t>(output[index])
              == cuda::std::bit_cast<cuda::std::uint32_t>(native[index]));
      REQUIRE(std::abs(static_cast<double>(output[index]) - expected) <= bound);
    }
  }
}

using logical_warp_sizes = c2h::enum_type_list<int, 1, 2, 4, 8, 16, 32>;

CUB_TEST(
  "Warp all-reduce sum returns every logical lane's aggregate", "[allreduce][warp]", CUB_SMALL, logical_warp_sizes)
{
  constexpr int width    = c2h::get<0, TestType>::value;
  const int distribution = GENERATE(0, 1, 2, 3, 4);
  check_all_reduce_sum<width, false, false>(distribution);
  check_all_reduce_sum<width, true, false>(distribution);
}

CUB_TEST("Warp all-reduce sum permits independent logical groups", "[allreduce][warp]", CUB_SMALL, logical_warp_sizes)
{
  constexpr int width = c2h::get<0, TestType>::value;
  check_all_reduce_sum<width, false, true>(GENERATE(0, 1, 2, 3, 4));
}
