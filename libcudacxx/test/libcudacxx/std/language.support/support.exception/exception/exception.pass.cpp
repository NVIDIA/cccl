//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test exception

#include <cuda/std/__exception>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    static_assert(cuda::std::is_polymorphic<cuda::std::exception>::value,
                 "cuda::std::is_polymorphic<cuda::std::exception>::value");
    cuda::std::exception b;
    cuda::std::exception b2 = b;
    b2 = b;
    const char* w = b2.what();
    const char* expected = "cuda::std::exception";
    for (size_t i = 0; i < 22; ++i) {
      assert(w[i] == expected[i]);
    }

  return 0;
}
