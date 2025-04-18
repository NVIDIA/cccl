//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/optional>

// template <class U>
//   constexpr EXPLICIT optional(U&& u);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const From' to 'short', possible loss of data

using cuda::std::optional;

struct ImplicitAny
{
  template <class U>
  __host__ __device__ constexpr ImplicitAny(U&&)
  {}
};

#ifdef CCCL_ENABLE_OPTIONAL_REF
template <class T>
struct ConvertibleToReference
{
  T val_;

  __host__ __device__ constexpr operator T&() noexcept
  {
    return val_;
  }

  __host__ __device__ constexpr operator const T&() const noexcept
  {
    return val_;
  }
};

template <class T>
struct ExplicitlyConvertibleToReference
{
  T val_;

  __host__ __device__ explicit constexpr operator T&() noexcept
  {
    return val_;
  }

  __host__ __device__ explicit constexpr operator const T&() const noexcept
  {
    return val_;
  }
};
#endif // CCCL_ENABLE_OPTIONAL_REF

template <class To>
__host__ __device__ constexpr optional<To> implicit_conversion(optional<To>&& opt)
{
  return opt;
}

enum class IsExplicit
{
  Yes,
  No
};

template <IsExplicit is_explicit, class To, class From>
__host__ __device__ constexpr void test([[maybe_unused]] From input)
{
  if constexpr (cuda::std::is_convertible_v<const From&, optional<To>>)
  {
    static_assert(is_explicit == IsExplicit::No);
    optional<To> opt = implicit_conversion<To>(cuda::std::as_const(input));
    assert(opt.has_value());
    assert(*opt == static_cast<To>(input));
  }
  else if constexpr (cuda::std::is_constructible_v<const From&, optional<To>>)
  {
    static_assert(is_explicit == IsExplicit::Yes);
    optional<To> opt{cuda::std::as_const(input)};
    assert(opt.has_value());
    assert(*opt == static_cast<To>(input));
  }
  else if constexpr (cuda::std::is_convertible_v<From, optional<To>>)
  {
    static_assert(is_explicit == IsExplicit::No);
    optional<To> opt = implicit_conversion<To>(cuda::std::move(input));
    assert(opt.has_value());
    assert(*opt == static_cast<To>(input));
  }
  else if constexpr (cuda::std::is_constructible_v<From, optional<To>>)
  {
    static_assert(is_explicit == IsExplicit::Yes);
    optional<To> opt{cuda::std::move(input)};
    assert(opt.has_value());
    assert(*opt == static_cast<To>(input));
  }
  else
  {
    optional<To> opt{input};
    assert(opt.has_value());
    assert(*opt == static_cast<To>(input));
    if constexpr (cuda::std::is_reference_v<To>)
    {
      assert(cuda::std::addressof(static_cast<To>(input)) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  // implicit conversions
  test<IsExplicit::No, int>(42);
  test<IsExplicit::No, double>(3.14);

  test<IsExplicit::No, short>(42);
  test<IsExplicit::No, float>(3.14);

  test<IsExplicit::No, TrivialTestTypes::TestType>(42);
  test<IsExplicit::No, ConstexprTestTypes::TestType>(42);

  // explicit conversions
  test<IsExplicit::Yes, ExplicitTrivialTestTypes::TestType>(42);
  test<IsExplicit::Yes, ExplicitConstexprTestTypes::TestType>(42);

#ifdef CCCL_ENABLE_OPTIONAL_REF
  {
    int val{42};
    test<IsExplicit::Yes, int&>(val);
    test<IsExplicit::No, const int&>(val);
  }
  {
    ConvertibleToReference<int> val{42};
    test<IsExplicit::No, int&>(val);
    test<IsExplicit::No, const int&>(val);
  }
  {
    const ConvertibleToReference<int> val{42};
    test<IsExplicit::No, int&>(val);
    test<IsExplicit::No, const int&>(val);
  }
  {
    ExplicitlyConvertibleToReference<int> val{42};
    test<IsExplicit::Yes, int&>(val);
    test<IsExplicit::Yes, const int&>(val);
  }
  {
    const ExplicitlyConvertibleToReference<int> val{42};
    test<IsExplicit::Yes, int&>(val);
    test<IsExplicit::Yes, const int&>(val);
  }
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

#if TEST_HAS_EXCEPTIONS()
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

template <class T>
void test_exceptions()
{
  try
  {
    if constexpr (cuda::std::is_convertible_v<int, optional<T>>)
    {
      optional<T> t = implicit_conversion<T>(42);
      unused(t);
      assert(false);
    }
    else
    {
      optional<T> t{42};
      unused(t);
      assert(false);
    }
  }
  catch (int)
  {}
}

void test_exceptions()
{
  test_exceptions<ImplicitThrow>();
  test_exceptions<ExplicitThrow>();
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  {
    using O = optional<ImplicitAny>;
    static_assert(!test_convertible<O, cuda::std::in_place_t>(), "");
    static_assert(!test_convertible<O, cuda::std::in_place_t&>(), "");
    static_assert(!test_convertible<O, const cuda::std::in_place_t&>(), "");
    static_assert(!test_convertible<O, cuda::std::in_place_t&&>(), "");
    static_assert(!test_convertible<O, const cuda::std::in_place_t&&>(), "");
  }

  {
    using T = ExplicitTestTypes::TestType;
    T::reset();
    {
      test<IsExplicit::Yes, T>(42);
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

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
