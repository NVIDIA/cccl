//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr T&& optional<T>::operator*() &&;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  __host__ __device__ constexpr int test() const&
  {
    return 3;
  }
  __host__ __device__ constexpr int test() &
  {
    return 4;
  }
  __host__ __device__ constexpr int test() const&&
  {
    return 5;
  }
  __host__ __device__ constexpr int test() &&
  {
    return 6;
  }
};

struct Y
{
  __host__ __device__ constexpr int test() &&
  {
    return 7;
  }

  __host__ __device__ constexpr int test() &
  {
    return 42;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    optional<X> opt{};
    unused(opt);
    ASSERT_SAME_TYPE(decltype(*cuda::std::move(opt)), X&&);
    LIBCPP_STATIC_ASSERT(noexcept(*cuda::std::move(opt)), "");
    // ASSERT_NOT_NOEXCEPT(*cuda::std::move(opt));
    // FIXME: This assertion fails with GCC because it can see that
    // (A) operator*() is constexpr, and
    // (B) there is no path through the function that throws.
    // It's arguable if this is the correct behavior for the noexcept
    // operator.
    // Regardless this function should still be noexcept(false) because
    // it has a narrow contract.

    optional<X&> optref;
    unused(optref);
    ASSERT_SAME_TYPE(decltype(*cuda::std::move(optref)), X&);
    LIBCPP_STATIC_ASSERT(noexcept(*cuda::std::move(optref)), "");
    ASSERT_NOEXCEPT(*cuda::std::move(optref));
  }

  {
    optional<X> opt(X{});
    assert((*cuda::std::move(opt)).test() == 6);
  }

  {
    X val{};
    optional<X&> opt(val);
    assert((*cuda::std::move(opt)).test() == 4);
  }

  {
    optional<Y> opt(Y{});
    assert((*cuda::std::move(opt)).test() == 7);
  }

  {
    Y val{};
    optional<Y&> opt(val);
    assert((*cuda::std::move(opt)).test() == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
