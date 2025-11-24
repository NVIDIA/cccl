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

#include <testing.cuh>
#include <unistd.h>

#include "common.h"

// todo: test error states and resetting errno

C2H_CCCLRT_TEST("cuFile cufile", "[cufile][cufile]")
{
  constexpr auto filename = "cufile_test_file";

  // 1. Test public cufile types and properties.
  STATIC_REQUIRE(cuda::std::is_base_of_v<cudax::cufile_ref, cudax::cufile>);
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile::off_type, ::off_t>);
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile::native_handle_type, int>);

  // 2. Test default constructor.
  STATIC_REQUIRE(cuda::std::is_nothrow_default_constructible_v<cudax::cufile>);
  {
    cudax::cufile file;
    CUDAX_REQUIRE(file.get() == nullptr);
    CUDAX_REQUIRE(file.native_handle() == -1);
  }

  // 3. Test cufile(const char*, cufile_open_mode) constructor.
  STATIC_REQUIRE(cuda::std::is_constructible_v<cudax::cufile, const char*, cudax::cufile_open_mode>);
  {
    cudax::cufile file{filename, cudax::cufile_open_mode::out};
    CUDAX_REQUIRE(file.get() != nullptr);
    test_check_fd_is_valid(file.native_handle());
    test_check_file_exists(filename);

    // todo: test different open flags
  }
  test_remove_file(filename);

  // 4. Test copy constructor.
  STATIC_REQUIRE(!cuda::std::is_copy_constructible_v<cudax::cufile>);

  // 5. Test move constructor.
  STATIC_REQUIRE(cuda::std::is_nothrow_move_constructible_v<cudax::cufile>);
  {
    cudax::cufile file1{filename, cudax::cufile_open_mode::out};
    test_check_file_exists(filename);

    cudax::cufile file2{cuda::std::move(file1)};
    CUDAX_REQUIRE(file1.get() == nullptr);
    CUDAX_REQUIRE(file1.native_handle() == -1);
    CUDAX_REQUIRE(file2.get() != nullptr);
    test_check_fd_is_valid(file2.native_handle());
  }
  test_remove_file(filename);

  // 6. Test copy assignment.
  STATIC_REQUIRE(!cuda::std::is_copy_assignable_v<cudax::cufile>);

  // 7. Test move assignment.
  STATIC_REQUIRE(cuda::std::is_move_assignable_v<cudax::cufile>);
  {
    cudax::cufile file2;

    {
      cudax::cufile file1{filename, cudax::cufile_open_mode::out};

      // self move assignment
      file1 = cuda::std::move(file1);
      CUDAX_REQUIRE(file1.get() != nullptr);
      test_check_fd_is_valid(file1.native_handle());
      test_check_file_exists(filename);

      // move assignment
      file2 = cuda::std::move(file1);
      CUDAX_REQUIRE(file1.get() == nullptr);
      CUDAX_REQUIRE(file1.native_handle() == -1);
      CUDAX_REQUIRE(file2.get() != nullptr);
      test_check_fd_is_valid(file2.native_handle());
    }

    test_check_file_exists(filename);
  }
  test_remove_file(filename);

  // 8. Test destructor.
  STATIC_REQUIRE(cuda::std::is_nothrow_destructible_v<cudax::cufile>);

  // 9. Test is_open().
  STATIC_REQUIRE(noexcept(cuda::std::declval<cudax::cufile>().is_open()));
  STATIC_REQUIRE(cuda::std::is_same_v<bool, decltype(cuda::std::declval<cudax::cufile>().is_open())>);
  {
    cudax::cufile file;
    CUDAX_REQUIRE(!file.is_open());

    file = cudax::cufile{filename, cudax::cufile_open_mode::out};
    test_check_file_exists(filename);

    CUDAX_REQUIRE(file.is_open());
  }
  test_remove_file(filename);

  // 10. Test open_mode().
  STATIC_REQUIRE(!noexcept(cuda::std::declval<cudax::cufile>().open_mode()));
  STATIC_REQUIRE(
    cuda::std::is_same_v<cudax::cufile_open_mode, decltype(cuda::std::declval<cudax::cufile>().open_mode())>);
  {
    cudax::cufile file;
    CUDAX_REQUIRE(file.open_mode() == cudax::cufile_open_mode{});

    file = cudax::cufile{filename, cudax::cufile_open_mode::out};
    test_check_file_exists(filename);

    CUDAX_REQUIRE(file.open_mode() == cudax::cufile_open_mode::out);

    // todo: test other flags
  }
  test_remove_file(filename);

  // 11. Test open().
  STATIC_REQUIRE(!noexcept(cuda::std::declval<cudax::cufile>().open(
    cuda::std::declval<const char*>(), cuda::std::declval<cudax::cufile_open_mode>())));
  STATIC_REQUIRE(
    cuda::std::is_same_v<void,
                         decltype(cuda::std::declval<cudax::cufile>().open(
                           cuda::std::declval<const char*>(), cuda::std::declval<cudax::cufile_open_mode>()))>);
  {
    cudax::cufile file;
    CUDAX_REQUIRE(!file.is_open());

    file.open(filename, cudax::cufile_open_mode::out);
    CUDAX_REQUIRE(file.get() != nullptr);
    test_check_fd_is_valid(file.native_handle());
    test_check_file_exists(filename);
    CUDAX_REQUIRE(file.is_open());

    CHECK_THROWS_AS(file.open(filename, cudax::cufile_open_mode::out), std::runtime_error);
  }
  test_remove_file(filename);

  // 12. Test close().
  STATIC_REQUIRE(!noexcept(cuda::std::declval<cudax::cufile>().close()));
  STATIC_REQUIRE(cuda::std::is_same_v<void, decltype(cuda::std::declval<cudax::cufile>().close())>);
  {
    cudax::cufile file;

    file.close();
    CUDAX_REQUIRE(file.get() == nullptr);
    CUDAX_REQUIRE(file.native_handle() == -1);

    file.open(filename, cudax::cufile_open_mode::out);
    CUDAX_REQUIRE(file.get() != nullptr);
    test_check_fd_is_valid(file.native_handle());
    test_check_file_exists(filename);

    file.close();
    CUDAX_REQUIRE(file.get() == nullptr);
    CUDAX_REQUIRE(file.native_handle() == -1);

    file.close();
    CUDAX_REQUIRE(file.get() == nullptr);
    CUDAX_REQUIRE(file.native_handle() == -1);
  }
  test_remove_file(filename);

  // 13. Test native_handle().
  STATIC_REQUIRE(noexcept(cuda::std::declval<cudax::cufile>().native_handle()));
  STATIC_REQUIRE(cuda::std::is_same_v<int, decltype(cuda::std::declval<cudax::cufile>().native_handle())>);

  // 14. Test release().
  STATIC_REQUIRE(noexcept(cuda::std::declval<cudax::cufile>().release()));
  STATIC_REQUIRE(cuda::std::is_same_v<int, decltype(cuda::std::declval<cudax::cufile>().release())>);
  {
    cudax::cufile file;
    CUDAX_REQUIRE(file.release() == -1);

    file.open(filename, cudax::cufile_open_mode::out);
    CUDAX_REQUIRE(file.get() != nullptr);
    test_check_fd_is_valid(file.native_handle());

    int fd = file.release();
    CUDAX_REQUIRE(file.get() == nullptr);
    CUDAX_REQUIRE(file.native_handle() == -1);

    CUDAX_REQUIRE(close(fd) == 0);
  }
  test_remove_file(filename);

  // 15. Test from_native_handle(native_handle).
  STATIC_REQUIRE(!noexcept(cudax::cufile::from_native_handle(cuda::std::declval<int>())));
  STATIC_REQUIRE(
    cuda::std::is_same_v<cudax::cufile, decltype(cudax::cufile::from_native_handle(cuda::std::declval<int>()))>);
  {
    FILE* cfile = std::fopen(filename, "w");
    test_check_file_exists(filename);

    int fd = fileno(cfile);
    CUDAX_REQUIRE(fd != -1);

    cudax::cufile file = cudax::cufile::from_native_handle(fd);
    CUDAX_REQUIRE(file.get() != nullptr);
    CUDAX_REQUIRE(file.native_handle() == fd);
  }
  test_remove_file(filename);
}
