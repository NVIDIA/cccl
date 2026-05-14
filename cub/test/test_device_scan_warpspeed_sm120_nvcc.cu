// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This test reproduces the sm120 NVCC codegen issue that requires disabling warpspeed scan on sm120 with NVCC < 13.4.
// See: https://github.com/NVIDIA/cccl/issues/8528

#undef CCCL_ENABLE_ASSERTIONS

#include <cub/device/device_scan.cuh>

#include <cstdio>

int main()
{
  int in_h[1] = {1};
  int* in{};
  int* out{};
  void* tmp{};
  size_t tmp_bytes{};

  cudaMalloc(&in, sizeof(in_h));
  cudaMalloc(&out, sizeof(in_h));
  cudaMemcpy(in, in_h, sizeof(in_h), cudaMemcpyHostToDevice);

  cub::DeviceScan::ExclusiveScan(tmp, tmp_bytes, in, out, cuda::std::plus<>{}, 0, 1);
  cudaMalloc(&tmp, tmp_bytes);
  cub::DeviceScan::ExclusiveScan(tmp, tmp_bytes, in, out, cuda::std::plus<>{}, 0, 1);

  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
  {
    std::printf("sync failed: %s\n", cudaGetErrorString(status));
    return 1;
  }

  int out_h{};
  cudaMemcpy(&out_h, out, sizeof(out_h), cudaMemcpyDeviceToHost);
  if (out_h != 0)
  {
    std::printf("wrong result: %d\n", out_h);
    return 1;
  }
}
