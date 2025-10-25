//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/cufile.cuh>

#include <cstdio>
#include <stdexcept>

#include <testing.cuh>

#include "common.h"

void test_register_native_handle()
{
  constexpr auto filename = "cufile_driver_register_test_file";

  // 1. Check noexcept.
  STATIC_REQUIRE(!noexcept(cudax::cufile_driver.register_native_handle(cuda::std::declval<int>())));
  STATIC_REQUIRE(noexcept(cudax::cufile_driver.deregister_native_handle(cuda::std::declval<cudax::cufile_ref>())));

  // 2. Test return type.
  STATIC_REQUIRE(
    cuda::std::is_same_v<cudax::cufile_ref,
                         decltype(cudax::cufile_driver.register_native_handle(cuda::std::declval<int>()))>);
  STATIC_REQUIRE(
    cuda::std::
      is_same_v<void, decltype(cudax::cufile_driver.deregister_native_handle(cuda::std::declval<cudax::cufile_ref>()))>);

  FILE* cfile = std::fopen(filename, "w");
  test_check_file_exists(filename);

  int fd = fileno(cfile);
  CUDAX_REQUIRE(fd != -1);

  // 3. Register a file handle. Should return a valid cuFile handle.
  cudax::cufile_ref file = cudax::cufile_driver.register_native_handle(fd);
  CUDAX_REQUIRE(file.get() != nullptr);

  // 4. Reregistering the same file handle should result in an cufile_error.
  CHECK_THROWS_AS(cudax::cufile_driver.register_native_handle(fd), cudax::cufile_error);

  // 5. Deregister the cuFile handles. Can be called multiple times.
  cudax::cufile_driver.deregister_native_handle(file);
  cudax::cufile_driver.deregister_native_handle(file); // should be fine

  test_remove_file(filename);
}

C2H_CCCLRT_TEST("cuFile driver register", "[cufile][driver]")
{
  // 1. Test registering a native file handle.
  test_register_native_handle();
}
