//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
  {
    using namespace cuda::literals;

    auto cc = 90_cc;
    static_assert(cuda::std::is_same_v<decltype(cc), cuda::compute_capability>);
    assert(cc.get() == 90);
  }

  {
    using namespace cuda::compute_capability_literals;

    auto cc = 90_cc;
    static_assert(cuda::std::is_same_v<decltype(cc), cuda::compute_capability>);
    assert(cc.get() == 90);
  }

  {
    using namespace cuda::literals::compute_capability_literals;

    auto cc = 90_cc;
    static_assert(cuda::std::is_same_v<decltype(cc), cuda::compute_capability>);
    assert(cc.get() == 90);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
