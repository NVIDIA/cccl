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

#include <cufile.h>
#include <testing.cuh>

#include "common.h"

C2H_CCCLRT_TEST("cuFile cufile_ref", "[cufile][cufile]")
{
  constexpr auto filename = "cufile_ref_test_file";

  // 1. Test public cufile_ref types and properties.
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_ref::off_type, ::off_t>);

  // 2. Test default constructor.
  STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cudax::cufile_ref>);

  // 3. Test cufile_ref(CUfileHandle_t) constructor.
  STATIC_REQUIRE(cuda::std::is_nothrow_constructible_v<cudax::cufile_ref, CUfileHandle_t>);
  STATIC_REQUIRE(cuda::std::is_convertible_v<CUfileHandle_t, cudax::cufile_ref>);
  {
    cudax::cufile file{filename, cudax::cufile_open_mode::out};

    cudax::cufile_ref file_ref{file.get()};
    CUDAX_REQUIRE(file_ref.get() == file.get());
  }
  test_remove_file(filename);

  // 4. Test copy constructor.
  STATIC_REQUIRE(cuda::std::is_trivially_copy_constructible_v<cudax::cufile_ref>);

  // 5. Test move constructor.
  STATIC_REQUIRE(cuda::std::is_trivially_move_constructible_v<cudax::cufile_ref>);

  // 6. Test copy assignment.
  STATIC_REQUIRE(cuda::std::is_trivially_copy_assignable_v<cudax::cufile_ref>);

  // 7. Test move assignment.
  STATIC_REQUIRE(cuda::std::is_trivially_move_assignable_v<cudax::cufile_ref>);

  // 8. Test destructor.
  STATIC_REQUIRE(cuda::std::is_trivially_destructible_v<cudax::cufile_ref>);

  // 9. Test get().
  STATIC_REQUIRE(noexcept(cuda::std::declval<cudax::cufile_ref>().get()));
  STATIC_REQUIRE(cuda::std::is_same_v<CUfileHandle_t, decltype(cuda::std::declval<cudax::cufile_ref>().get())>);
  {
    const cudax::cufile file;
    CUDAX_REQUIRE(cudax::cufile_ref{file}.get() == nullptr);
  }
  {
    cudax::cufile file{filename, cudax::cufile_open_mode::out};
    CUDAX_REQUIRE(cudax::cufile_ref{file}.get() != nullptr);
  }
  test_remove_file(filename);
}
