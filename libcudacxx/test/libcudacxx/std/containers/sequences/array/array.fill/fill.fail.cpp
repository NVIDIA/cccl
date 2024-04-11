//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// void fill(const T& u);

#include <cuda/std/array>
#include <cuda/std/cassert>

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<const T, 0> C;
    C c = {};
    // expected-error-re@array:* {{{{(static_assert|static assertion)}} failed{{.*}}cannot fill zero-sized array of type
    // 'const T'}}
    c.fill(5.5); // expected-note {{requested here}}
  }

  return 0;
}
