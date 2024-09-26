//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <iostream>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int device_guard(int device_id)
{
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
  {
    std::cerr << "Can't query devices number." << std::endl;
    std::exit(-1);
  }

  if (device_id >= device_count || device_id < 0)
  {
    std::cerr << "Invalid device ID: " << device_id << std::endl;
    std::exit(-1);
  }

  return device_id;
}

int main(int argc, char* argv[])
{
  Catch::Session session;

  int device_id{};

  // Build a new parser on top of Catch's
  using namespace Catch::clara;
  auto cli = session.cli() | Opt(device_id, "device")["-d"]["--device"]("device id to use");
  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)
  {
    return returnCode;
  }

  cudaSetDevice(device_guard(device_id));
  return session.run(argc, argv);
}
