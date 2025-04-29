//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> T& get(array<T, N>& a);

// Prevent -Warray-bounds from issuing a diagnostic when testing with clang verify.
// ADDITIONAL_COMPILE_OPTIONS_HOST: -Wno-array-bounds

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c                  = {1, 2, 3.5};
    cuda::std::get<3>(c) = 5.5; // expected-note {{requested here}}
    // expected-error-re@array:* {{{{(static_assert|static assertion)}} failed{{( due to requirement '3U[L]{0,2} <
    // 3U[L]{0,2}')?}}{{.*}}Index out of bounds in cuda::std::get<> (cuda::std::array)}}
  }

  return 0;
}
