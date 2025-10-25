//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include <fcntl.h>
#include <unistd.h>

void test_setup_cufile_driver()
{
  // Check if the current cufile.json was overridden.
  if (std::getenv("CUFILE_ENV_PATH_JSON") != nullptr)
  {
    return;
  }

  // Make the path to cufile.json in local src directory.
  std::string path{__FILE__};
  auto path_end = path.find_last_of('/') + 1;
  path.resize(path_end);
  path.append("cufile.json");

  // Set the environment variable.
  CUDAX_REQUIRE(setenv("CUFILE_ENV_PATH_JSON", path.c_str(), true) == 0);
}

void test_check_fd_is_valid(int fd)
{
  CUDAX_REQUIRE(fcntl(fd, F_GETFD) != -1);
}

void test_check_file_exists(const char* filename)
{
  CUDAX_REQUIRE(access(filename, F_OK) == 0);
}

void test_remove_file(const char* filename)
{
  test_check_file_exists(filename);
  CUDAX_REQUIRE(std::remove(filename) == 0);
}
