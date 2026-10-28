// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This test is a minimal reproducer of an illegal memory access error observed in lookahead scan.
// More details here: https://github.com/NVIDIA/cccl/issues/8838

// Assertions hide the bug, so we need to disable them
#undef CCCL_ENABLE_ASSERTIONS

#include <cub/device/device_scan.cuh>
#include <cub/util_debug.cuh>

#include <cstdio>

#include "cub_non_catch2_test_memory.h"

CUB_TEST_MEMORY_CLASS(CUB_SMALL);

int main()
{
  int in_h[2] = {1, 1};
  int* in{};
  long long* out{};
  void* tmp{};
  size_t tmp_bytes{};

  CubDebugExit(cudaMalloc(&in, sizeof(in_h)));
  CubDebugExit(cudaMalloc(&out, 3 * sizeof(*out)));
  CubDebugExit(cudaMemcpy(in, in_h, sizeof(in_h), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemset(out, 0, 3 * sizeof(*out)));

  CubDebugExit(cub::DeviceScan::InclusiveScan(tmp, tmp_bytes, in, out + 1, cuda::std::plus<>{}, 2));
  CubDebugExit(cudaMalloc(&tmp, tmp_bytes));
  CubDebugExit(cub::DeviceScan::InclusiveScan(tmp, tmp_bytes, in, out + 1, cuda::std::plus<>{}, 2));

  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
  {
    std::printf("sync failed: %s\n", cudaGetErrorString(status));
    return 1;
  }

  long long out_h[3]{};
  CubDebugExit(cudaMemcpy(out_h, out, sizeof(out_h), cudaMemcpyDeviceToHost));
  if (out_h[0] != 0 || out_h[1] != 1 || out_h[2] != 2)
  {
    std::printf("wrong result: {%lld, %lld, %lld}\n", out_h[0], out_h[1], out_h[2]);
    return 1;
  }
}
