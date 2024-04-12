//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// constexpr optional<T>& operator=(const optional<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class Tp>
__host__ __device__ constexpr bool assign_empty(optional<Tp>&& lhs)
{
  const optional<Tp> rhs;
  lhs = rhs;
  return !lhs.has_value() && !rhs.has_value();
}

template <class Tp>
__host__ __device__ constexpr bool assign_value(optional<Tp>&& lhs)
{
  const optional<Tp> rhs(101);
  lhs = rhs;
  return lhs.has_value() && rhs.has_value() && *lhs == *rhs;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct X
{
  STATIC_MEMBER_VAR(throw_now, bool);

  X() = default;
  X(const X&)
  {
    if (throw_now())
    {
      TEST_THROW(6);
    }
  }
  X& operator=(X const&) = default;
};

void test_exceptions()
{
  optional<X> opt;
  optional<X> opt2(X{});
  assert(static_cast<bool>(opt2) == true);
  try
  {
    X::throw_now() = true;
    opt            = opt2;
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
    assert(static_cast<bool>(opt) == false);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  {
    using O = optional<int>;
#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(assign_empty(O{42}), "");
    static_assert(assign_value(O{42}), "");
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
    assert(assign_empty(O{42}));
    assert(assign_value(O{42}));
  }
  {
    using O = optional<TrivialTestTypes::TestType>;
#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(assign_empty(O{42}), "");
    static_assert(assign_value(O{42}), "");
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
    assert(assign_empty(O{42}));
    assert(assign_value(O{42}));
  }
  {
    using O = optional<TestTypes::TestType>;
    assert(assign_empty(O{42}));
    assert(assign_value(O{42}));
  }
  {
    using T = TestTypes::TestType;
    T::reset();
    optional<T> opt(3);
    const optional<T> opt2;
    assert(T::alive() == 1);
    opt = opt2;
    assert(T::alive() == 0);
    assert(!opt2.has_value());
    assert(!opt.has_value());
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
