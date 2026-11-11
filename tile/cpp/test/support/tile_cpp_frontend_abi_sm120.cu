//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/coop/tile.cuh>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct = ::cuda::tiles;

inline constexpr size_t tile_size = 128;

__tile_global__ void frontend_abi_kernel(
  const __half* half_input, __half* half_output, const __nv_bfloat16* bfloat_input, __nv_bfloat16* bfloat_output)
{
  auto half_input_span    = ct::tensor_span{half_input, ct::shape<tile_size>{}};
  auto half_output_span   = ct::tensor_span{half_output, ct::shape<tile_size>{}};
  auto bfloat_input_span  = ct::tensor_span{bfloat_input, ct::shape<tile_size>{}};
  auto bfloat_output_span = ct::tensor_span{bfloat_output, ct::shape<tile_size>{}};
  auto half_input_view    = ct::partition_view{half_input_span, ct::shape<tile_size>{}};
  auto half_output_view   = ct::partition_view{half_output_span, ct::shape<tile_size>{}};
  auto bfloat_input_view  = ct::partition_view{bfloat_input_span, ct::shape<tile_size>{}};
  auto bfloat_output_view = ct::partition_view{bfloat_output_span, ct::shape<tile_size>{}};

  const auto half_sum   = ::cuda::coop::tile::sum(half_input_view.load(0));
  const auto bfloat_sum = ::cuda::coop::tile::sum(bfloat_input_view.load(0));

  // Both provider calls receive compiler-owned shared scratch through a
  // generic void*, exercising the !llvm.ptr<3> to !llvm.ptr frontend ABI map.
  half_output_view.store(half_sum, 0);
  bfloat_output_view.store(bfloat_sum, 0);
}

bool check_cuda(cudaError_t status, const char* operation)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

int main()
{
  constexpr size_t half_bytes     = tile_size * sizeof(__half);
  constexpr size_t bfloat_bytes   = tile_size * sizeof(__nv_bfloat16);
  constexpr float expected_half   = static_cast<float>(tile_size);
  constexpr float expected_bfloat = 2.0f * static_cast<float>(tile_size);

  const std::vector<__half> half_input(tile_size, __float2half_rn(1.0f));
  std::vector<__half> half_output(tile_size);
  const std::vector<__nv_bfloat16> bfloat_input(tile_size, __float2bfloat16_rn(2.0f));
  std::vector<__nv_bfloat16> bfloat_output(tile_size);

  __half* device_half_input           = nullptr;
  __half* device_half_output          = nullptr;
  __nv_bfloat16* device_bfloat_input  = nullptr;
  __nv_bfloat16* device_bfloat_output = nullptr;
  if (!check_cuda(cudaMalloc(&device_half_input, half_bytes), "cudaMalloc(half_input)")
      || !check_cuda(cudaMalloc(&device_half_output, half_bytes), "cudaMalloc(half_output)")
      || !check_cuda(cudaMalloc(&device_bfloat_input, bfloat_bytes), "cudaMalloc(bfloat_input)")
      || !check_cuda(cudaMalloc(&device_bfloat_output, bfloat_bytes), "cudaMalloc(bfloat_output)"))
  {
    return EXIT_FAILURE;
  }

  if (!check_cuda(cudaMemcpy(device_half_input, half_input.data(), half_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(half_input)")
      || !check_cuda(cudaMemcpy(device_bfloat_input, bfloat_input.data(), bfloat_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(bfloat_input)"))
  {
    return EXIT_FAILURE;
  }

  frontend_abi_kernel<<<1, 1>>>(device_half_input, device_half_output, device_bfloat_input, device_bfloat_output);
  if (!check_cuda(cudaGetLastError(), "frontend_abi_kernel launch")
      || !check_cuda(cudaDeviceSynchronize(), "frontend_abi_kernel execution")
      || !check_cuda(cudaMemcpy(half_output.data(), device_half_output, half_bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy(half_output)")
      || !check_cuda(cudaMemcpy(bfloat_output.data(), device_bfloat_output, bfloat_bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy(bfloat_output)"))
  {
    return EXIT_FAILURE;
  }

  for (size_t index = 0; index < tile_size; ++index)
  {
    if (__half2float(half_output[index]) != expected_half || __bfloat162float(bfloat_output[index]) != expected_bfloat)
    {
      std::cerr << "frontend ABI result mismatch at " << index << '\n';
      return EXIT_FAILURE;
    }
  }

  bool freed = true;
  freed &= check_cuda(cudaFree(device_bfloat_output), "cudaFree(bfloat_output)");
  freed &= check_cuda(cudaFree(device_bfloat_input), "cudaFree(bfloat_input)");
  freed &= check_cuda(cudaFree(device_half_output), "cudaFree(half_output)");
  freed &= check_cuda(cudaFree(device_half_input), "cudaFree(half_input)");
  if (!freed)
  {
    return EXIT_FAILURE;
  }

  std::cout << "PASS: C++ Tile frontend ABI f16/bf16\n";
  return EXIT_SUCCESS;
}
