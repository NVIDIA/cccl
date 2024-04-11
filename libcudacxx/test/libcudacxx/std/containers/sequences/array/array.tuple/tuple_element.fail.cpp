//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// tuple_element<I, array<T, N> >::type

#include <cuda/std/array>
#include <cuda/std/cassert>

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    cuda::std::tuple_element<3, C> foo; // expected-note {{requested here}}
    // expected-error-re@array:* {{{{(static_assert|static assertion)}} failed{{( due to requirement '3U[L]{0,2} <
    // 3U[L]{0,2}')?}}{{.*}}Index out of bounds in cuda::std::tuple_element<> (cuda::std::array)}}
  }

  return 0;
}
