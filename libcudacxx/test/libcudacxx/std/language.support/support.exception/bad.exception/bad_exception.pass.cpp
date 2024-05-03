//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test bad_exception

#include <cuda/std/__exception_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert((cuda::std::is_base_of<cuda::std::exception, cuda::std::bad_exception>::value),
                "cuda::std::is_base_of<cuda::std::exception, std::bad_exception>::value");
  static_assert(cuda::std::is_polymorphic<cuda::std::bad_exception>::value,
                "cuda::std::is_polymorphic<cuda::std::bad_exception>::value");
  cuda::std::bad_exception b;
  cuda::std::bad_exception b2 = b;
  b2                          = b;
  const char* w               = b2.what();
  const char* expected        = "cuda::std::bad_exception";
  for (size_t i = 0; i < 26; ++i)
  {
    assert(w[i] == expected[i]);
  }

  return 0;
}
