//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11

// <cuda/std/optional>

// template <class U>
//   constexpr EXPLICIT optional(U&& u);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"

using cuda::std::optional;

struct ImplicitAny
{
  template <class U>
  __host__ __device__ constexpr ImplicitAny(U&&)
  {}
};

template <class To, class From>
__host__ __device__ constexpr bool implicit_conversion(optional<To>&& opt, const From& v)
{
  using O = optional<To>;
  static_assert(test_convertible<O, From>(), "");
  static_assert(!test_convertible<O, void*>(), "");
  static_assert(!test_convertible<O, From, int>(), "");
  return opt && *opt == static_cast<To>(v);
}

template <class To, class Input, class Expect>
__host__ __device__ constexpr bool explicit_conversion(Input&& in, const Expect& v)
{
  using O = optional<To>;
  static_assert(cuda::std::is_constructible<O, Input>::value, "");
  static_assert(!cuda::std::is_convertible<Input, O>::value, "");
  static_assert(!cuda::std::is_constructible<O, void*>::value, "");
  static_assert(!cuda::std::is_constructible<O, Input, int>::value, "");
  optional<To> opt(cuda::std::forward<Input>(in));
  return opt && *opt == static_cast<To>(v);
}

__host__ __device__ void test_implicit()
{
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    static_assert(implicit_conversion<long long>(42, 42), "");
  }
  {
    static_assert(implicit_conversion<double>(3.14, 3.14), "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    int x = 42;
    optional<void* const> o(&x);
    assert(*o == &x);
  }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    using T = TrivialTestTypes::TestType;
    static_assert(implicit_conversion<T>(42, 42), "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    using T = TestTypes::TestType;
    assert(implicit_conversion<T>(3, T(3)));
  }
  {
    using O = optional<ImplicitAny>;
    static_assert(!test_convertible<O, cuda::std::in_place_t>(), "");
    static_assert(!test_convertible<O, cuda::std::in_place_t&>(), "");
    static_assert(!test_convertible<O, const cuda::std::in_place_t&>(), "");
    static_assert(!test_convertible<O, cuda::std::in_place_t&&>(), "");
    static_assert(!test_convertible<O, const cuda::std::in_place_t&&>(), "");
  }
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct ImplicitThrow
{
  constexpr ImplicitThrow(int x)
  {
    if (x != -1)
    {
      TEST_THROW(6);
    }
  }
};

void test_exceptions_implicit()
{
  try
  {
    using T       = ImplicitThrow;
    optional<T> t = 42;
    assert(false);
    ((void) t);
  }
  catch (int)
  {}
}
#endif // !TEST_HAS_NO_EXCEPTIONS

__host__ __device__ void test_explicit()
{
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    using T = ExplicitTrivialTestTypes::TestType;
    static_assert(explicit_conversion<T>(42, 42), "");
  }
  {
    using T = ExplicitConstexprTestTypes::TestType;
    static_assert(explicit_conversion<T>(42, 42), "");
    static_assert(!cuda::std::is_convertible<int, T>::value, "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    using T = ExplicitTestTypes::TestType;
    T::reset();
    {
      assert(explicit_conversion<T>(42, 42));
      assert(T::alive() == 0);
    }
    T::reset();
    {
      optional<T> t(42);
      assert(T::alive() == 1);
      assert(T::value_constructed() == 1);
      assert(T::move_constructed() == 0);
      assert(T::copy_constructed() == 0);
      assert(t.value().value == 42);
    }
    assert(T::alive() == 0);
  }
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct ExplicitThrow
{
  constexpr explicit ExplicitThrow(int x)
  {
    if (x != -1)
    {
      TEST_THROW(6);
    }
  }
};
void test_exceptions_explicit()
{
  try
  {
    using T = ExplicitThrow;
    optional<T> t(42);
    assert(false);
  }
  catch (int)
  {}
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test_implicit();
  test_explicit();

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions_implicit();))
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions_explicit();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
