//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "cuda_smoke_common.cuh"

TEST_CASE("CUDA device is available", "[cuda_smoke]")
{
  int device_count = 0;
  CUDART_REQUIRE(cudaGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);

  CUDART_REQUIRE(cudaSetDevice(0));

  cudaDeviceProp props{};
  CUDART_REQUIRE(cudaGetDeviceProperties(&props, 0));
  REQUIRE(props.name[0] != '\0');

  REQUIRE(cudaGetLastError() == cudaSuccess);
}
