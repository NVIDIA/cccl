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

__host__ __device__ constexpr void test_members()
{
  cuda::std::from_chars_result x{nullptr, cuda::std::errc{}};

  auto [ptr, ec] = x;

  static_assert(cuda::std::is_same_v<decltype(ptr), const char*>);
  assert(ptr == x.ptr);

  static_assert(cuda::std::is_same_v<decltype(ec), cuda::std::errc>);
  assert(ec == x.ec);
}

__host__ __device__ constexpr void test_operator_bool()
{
  static_assert(!cuda::std::is_convertible_v<cuda::std::from_chars_result, bool>);
  static_assert(cuda::std::is_constructible_v<bool, cuda::std::from_chars_result>);

  {
    cuda::std::from_chars_result value{nullptr, cuda::std::errc{}};
    assert(bool(value) == true);
    static_assert(noexcept(bool(value)) == true);
  }
  {
    cuda::std::from_chars_result value{nullptr, cuda::std::errc::value_too_large};
    assert(bool(value) == false);
    static_assert(noexcept(bool(value)) == true);
  }
}

__host__ __device__ constexpr void test_operator_eq_and_neq()
{
  const char a[]{'a'};
  const char b[]{'b'};

  {
    cuda::std::from_chars_result lhs{a, cuda::std::errc::value_too_large};
    cuda::std::from_chars_result rhs{a, cuda::std::errc::value_too_large};
    assert(lhs == rhs);
    assert(!(lhs != rhs));
  }
  {
    cuda::std::from_chars_result lhs{a, cuda::std::errc::value_too_large};
    cuda::std::from_chars_result rhs{a, cuda::std::errc::invalid_argument};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
  }
  {
    cuda::std::from_chars_result lhs{a, cuda::std::errc::value_too_large};
    cuda::std::from_chars_result rhs{b, cuda::std::errc::value_too_large};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
  }
  {
    cuda::std::from_chars_result lhs{a, cuda::std::errc::value_too_large};
    cuda::std::from_chars_result rhs{b, cuda::std::errc::invalid_argument};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
  }
}

__host__ __device__ constexpr bool test()
{
  test_members();
  test_operator_bool();
  test_operator_eq_and_neq();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
