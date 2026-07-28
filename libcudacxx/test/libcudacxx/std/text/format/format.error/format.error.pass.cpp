//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <cuda/std/format>

// class format_error;

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>

#include <string>

#include "test_macros.h"

void test_format_error()
{
#if __cpp_lib_format >= 201907L
  static_assert(cuda::std::is_same_v<cuda::std::format_error, std::format_error>);
#endif // __cpp_lib_format >= 201907L

  static_assert(cuda::std::is_base_of_v<std::runtime_error, cuda::std::format_error>);
  static_assert(cuda::std::is_polymorphic_v<cuda::std::format_error>);

  {
    const char* msg = "format_error message c-string";
    cuda::std::format_error e(msg);
    assert(cuda::std::strcmp(e.what(), msg) == 0);
    cuda::std::format_error e2(e);
    assert(cuda::std::strcmp(e2.what(), msg) == 0);
    e2 = e;
    assert(cuda::std::strcmp(e2.what(), msg) == 0);
  }
  {
    std::string msg("format_error message std::string");
    cuda::std::format_error e(msg);
    assert(e.what() == msg);
    cuda::std::format_error e2(e);
    assert(e2.what() == msg);
    e2 = e;
    assert(e2.what() == msg);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_format_error();))
  return 0;
}
