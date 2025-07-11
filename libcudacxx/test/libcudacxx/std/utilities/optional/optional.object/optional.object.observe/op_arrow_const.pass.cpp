//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr const T* optional<T>::operator->() const;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  __host__ __device__ constexpr int test() const
  {
    return 3;
  }
};

struct Y
{
  __host__ __device__ constexpr int test() const noexcept
  {
    return 2;
  }
};

struct Z
{
  __host__ __device__ constexpr const Z* operator&() const;
  __host__ __device__ constexpr int test() const
  {
    return 1;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    const cuda::std::optional<X> opt{};
    unused(opt);
    static_assert(cuda::std::is_same_v<decltype(opt.operator->()), X const*>);
    // static_assert(!noexcept(opt.operator->()));
    // FIXME: This assertion fails with GCC because it can see that
    // (A) operator->() is constexpr, and
    // (B) there is no path through the function that throws.
    // It's arguable if this is the correct behavior for the noexcept
    // operator.
    // Regardless this function should still be noexcept(false) because
    // it has a narrow contract.

    const cuda::std::optional<X&> optref;
    unused(optref);
    static_assert(cuda::std::is_same_v<decltype(optref.operator->()), X*>);
    static_assert(noexcept(optref.operator->()));
  }

  {
    const optional<X> opt(X{});
    assert(opt->test() == 3);
  }

  {
    X val{};
    const optional<X&> opt(val);
    assert(opt->test() == 3);
    assert(cuda::std::addressof(val) == opt.operator->());
  }

  {
    const optional<Y> opt(Y{});
    assert(opt->test() == 2);
  }

  {
    Y val{};
    const optional<Y&> opt(val);
    assert(opt->test() == 2);
    assert(cuda::std::addressof(val) == opt.operator->());
  }

  {
    const optional<Z> opt(Z{});
    assert(opt->test() == 1);
  }

  {
    Z val{};
    const optional<Z> opt(val);
    assert(opt->test() == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
