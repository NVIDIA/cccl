//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// SIMD host/device functions are not supported in tile mode.

#include <cuda/std/cassert>
#include <cuda/std/simd>
#include <cuda/std/span>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <class T, class Flags>
TEST_FUNC void test_flags(Flags flags)
{
  using Vec = simd::vec<T, 4>;
  alignas(32) const volatile T input[4]{1, 2, 3, 4};
  alignas(32) volatile T output[4]{};
  const cuda::std::span<const volatile T, 4> source(input);

  const Vec constructed(source, flags);
  const auto values = simd::unchecked_load<Vec>(source, flags);
  simd::unchecked_store(values, output, flags);

  for (int i = 0; i < 4; ++i)
  {
    assert(constructed[i] == static_cast<T>(i + 1));
    assert(values[i] == static_cast<T>(i + 1));
    assert(output[i] == static_cast<T>(i + 1));
  }
}

TEST_FUNC bool test()
{
  test_flags<int>(simd::flag_default);
  test_flags<float>(simd::flag_aligned);
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
