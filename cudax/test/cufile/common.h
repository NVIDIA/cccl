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
