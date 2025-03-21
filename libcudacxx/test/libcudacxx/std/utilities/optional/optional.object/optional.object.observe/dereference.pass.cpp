//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr T& optional<T>::operator*() &;

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
  __host__ __device__ constexpr int test()
  {
    return 7;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    optional<X> opt{};
    unused(opt);
    static_assert(cuda::std::is_same_v<decltype(*opt), X&>);
    static_assert(noexcept(*opt), "");
    // static_assert(!noexcept(*opt));
    // FIXME: This assertion fails with GCC because it can see that
    // (A) operator*() is constexpr, and
    // (B) there is no path through the function that throws.
    // It's arguable if this is the correct behavior for the noexcept
    // operator.
    // Regardless this function should still be noexcept(false) because
    // it has a narrow contract.

    optional<X&> optref;
    unused(optref);
    static_assert(cuda::std::is_same_v<decltype(*optref), X&>);
    static_assert(noexcept(*optref), "");
    static_assert(noexcept(*optref));
  }

  {
    optional<X> opt(X{});
    assert((*opt).test() == 4);
  }

  {
    X val{};
    optional<X&> opt(val);
    assert((*opt).test() == 4);
    assert(cuda::std::addressof(val) == cuda::std::addressof(*opt));
  }

  {
    optional<Y> opt(Y{});
    assert((*opt).test() == 7);
  }

  {
    Y val{};
    optional<Y&> opt(val);
    assert((*opt).test() == 7);
    assert(cuda::std::addressof(val) == cuda::std::addressof(*opt));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
