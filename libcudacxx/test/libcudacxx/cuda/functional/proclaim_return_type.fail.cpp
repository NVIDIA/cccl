//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

#include <cuda/functional>

#include <assert.h>

int main(int argc, char** argv)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (auto f = cuda::proclaim_return_type<double>([] __device__() -> int {
       return 42;
     });

     assert(f() == 42);),
    static_assert(false);)

  return 0;
}
