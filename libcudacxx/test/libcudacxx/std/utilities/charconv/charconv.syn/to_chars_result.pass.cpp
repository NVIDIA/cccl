//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__charconv_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
  static_assert(!cuda::std::is_convertible_v<cuda::std::to_chars_result, bool>);
  static_assert(cuda::std::is_constructible_v<bool, cuda::std::to_chars_result>);

  cuda::std::to_chars_result x{nullptr, cuda::std::errc{}};

  auto [ptr, ec] = x;

  static_assert(cuda::std::is_same_v<decltype(ptr), char*>);
  assert(ptr == x.ptr);

  static_assert(cuda::std::is_same_v<decltype(ec), cuda::std::errc>);
  assert(ec == x.ec);

  {
    cuda::std::to_chars_result value{nullptr, cuda::std::errc{}};
    assert(bool(value) == true);
    static_assert(noexcept(bool(value)) == true);
  }

  {
    cuda::std::to_chars_result value{nullptr, cuda::std::errc::value_too_large};
    assert(bool(value) == false);
    static_assert(noexcept(bool(value)) == true);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
