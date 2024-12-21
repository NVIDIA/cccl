//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)
#  pragma warning(disable : 4018 4389) // various "signed/unsigned mismatch"
#endif

// constexpr iota_view(type_identity_t<W> value, type_identity_t<Bound> bound);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(0), SomeInt(10));
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    cuda::std::ranges::iota_view<SomeInt> io(SomeInt(0), cuda::std::unreachable_sentinel);
    assert(cuda::std::ranges::next(io.begin(), 10) != io.end());
  }

  {
    cuda::std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>> io(SomeInt(0), IntComparableWith(SomeInt(10)));
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    // This is allowed only when using the constructor (not the deduction guide).
    cuda::std::ranges::iota_view<int, unsigned> signedUnsigned(0, 10);
    assert(cuda::std::ranges::next(signedUnsigned.begin(), 10) == signedUnsigned.end());
  }

  {
    // This is allowed only when using the constructor (not the deduction guide).
    cuda::std::ranges::iota_view<unsigned, int> signedUnsigned(0, 10);
    assert(cuda::std::ranges::next(signedUnsigned.begin(), 10) == signedUnsigned.end());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
