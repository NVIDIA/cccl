// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#undef CCCL_ENABLE_ASSERTIONS

#include <cub/device/device_scan.cuh>

#include <cstdio>

int main()
{
  int in_h[2] = {1, 1};
  int* in{};
  long long* out{};
  void* tmp{};
  size_t tmp_bytes{};

  cudaMalloc(&in, sizeof(in_h));
  cudaMalloc(&out, 3 * sizeof(*out));
  cudaMemcpy(in, in_h, sizeof(in_h), cudaMemcpyHostToDevice);
  cudaMemset(out, 0, 3 * sizeof(*out));

  cub::DeviceScan::InclusiveScan(tmp, tmp_bytes, in, out + 1, cuda::std::plus<>{}, 2);
  cudaMalloc(&tmp, tmp_bytes);
  cub::DeviceScan::InclusiveScan(tmp, tmp_bytes, in, out + 1, cuda::std::plus<>{}, 2);

  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
  {
    std::printf("sync failed: %s\n", cudaGetErrorString(status));
    return 1;
  }

  long long out_h[3]{};
  cudaMemcpy(out_h, out, sizeof(out_h), cudaMemcpyDeviceToHost);
  if (out_h[0] != 0 || out_h[1] != 1 || out_h[2] != 2)
  {
    std::printf("wrong result: {%lld, %lld, %lld}\n", out_h[0], out_h[1], out_h[2]);
    return 1;
  }
}
