// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

//! @file
//! This file includes implementation of CUDA-specific utilities for custom Catch2 main. When CMake is configured to
//! include all the tests into a single executable, this file is only included into catch2_runner_helper.cu. When CMake
//! is configured to compile each test as a separate binary, this file is included into each test.

#include <cuda/__runtime/api_wrapper.h>

#include <iostream>

int device_guard(int device_id)
{
  int device_count{};
  if (_CCCL_LOG_CUDA_API(cudaGetDeviceCount, "Failed getting number of devices", &device_count) != cudaSuccess)
  {
    std::exit(-1);
  }

  if (device_id >= device_count || device_id < 0)
  {
    std::cerr << "Invalid device ID: " << device_id << std::endl;
    std::exit(-1);
  }

  return device_id;
}

void set_device(int device_id)
{
  if (_CCCL_LOG_CUDA_API(cudaSetDevice, "Failed to set requested device", device_guard(device_id)) != cudaSuccess)
  {
    std::cerr << "Failed to set device ID: " << device_id << std::endl;
    std::exit(-1);
  }
}
