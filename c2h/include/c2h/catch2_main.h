// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <thrust/detail/config/device_system.h>

#include <iostream>

//! @file
//! This file includes a custom Catch2 main function. When CMake is configured to build each test as a separate
//! executable, this header is included into each test. On the other hand, when all the tests are compiled into a single
//! executable, this header is excluded from the tests and included into catch2_runner.cpp

#include <catch2/catch_session.hpp>

#ifdef C2H_CONFIG_MAIN
#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#    include <c2h/catch2_runner_helper.h>

#    ifndef C2H_EXCLUDE_CATCH2_HELPER_IMPL
#      include "catch2_runner_helper.inl"
#    endif // !C2H_EXCLUDE_CATCH2_HELPER_IMPL
#  endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

int main(int argc, char* argv[])
{
  Catch::Session session;

#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  int device_id{};

  // Build a new parser on top of Catch's
  using namespace Catch::Clara;
  auto cli = session.cli() | Opt(device_id, "device")["-d"]["--device"]("device id to use");
  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)
  {
    return returnCode;
  }

  set_device(device_id);
#  endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  return session.run();
}
#endif // C2H_CONFIG_MAIN
