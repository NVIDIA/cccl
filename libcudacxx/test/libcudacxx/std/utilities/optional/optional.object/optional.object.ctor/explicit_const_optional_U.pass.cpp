//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class U>
//   explicit optional(const optional<U>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

class X
{
  int i_;

public:
  __host__ __device__ constexpr explicit X(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr X(const X& x)
      : i_(x.i_)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~X()
  {
    i_ = 0;
  }
  __host__ __device__ friend constexpr bool operator==(const X& x, const X& y)
  {
    return x.i_ == y.i_;
  }
};

class Y
{
  int i_;

public:
  __host__ __device__ constexpr explicit Y(int i)
      : i_(i)
  {}

  __host__ __device__ friend constexpr bool operator==(const Y& x, const Y& y)
  {
    return x.i_ == y.i_;
  }
};

#ifdef CCCL_ENABLE_OPTIONAL_REF
template <class T>
struct ConvertibleToReference
{
  T val_;

  __host__ __device__ constexpr operator const T&() const noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool operator==(const int& lhs, const ConvertibleToReference& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};

template <class T>
struct ExplicitlyConvertibleToReference
{
  T val_;

  __host__ __device__ explicit constexpr operator const T&() const noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool
  operator==(const int& lhs, const ExplicitlyConvertibleToReference& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};

template <class T>
struct ConvertibleToValue
{
  T val_;

  __host__ __device__ constexpr operator T() const noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool operator==(const int& lhs, const ConvertibleToValue& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};

template <class T>
struct ExplicitlyConvertibleToValue
{
  T val_;

  __host__ __device__ explicit constexpr operator T() const noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool operator==(const int& lhs, const ExplicitlyConvertibleToValue& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};
#endif // CCCL_ENABLE_OPTIONAL_REF

template <class T, class U>
__host__ __device__ constexpr void test()
{
  { // constructed from empty
    const optional<U> input{};
    optional<T> opt{input};
    assert(!input.has_value());
    assert(!opt.has_value());
  }
  { // constructed from non-empty
    cuda::std::remove_reference_t<U> val{42};
    const optional<U> input{val};
    optional<T> opt{input};
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == T{42});
  }
}

__host__ __device__ constexpr bool test()
{
  test<X, int>();
  test<Y, int>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int, int&>();
  test<int, const int&>();

  test<const int&, ConvertibleToReference<int>>();
  test<const int&, ExplicitlyConvertibleToReference<int>>();

  test<int, ConvertibleToReference<int>&>();
  test<int, ExplicitlyConvertibleToReference<int>&>();

  test<int, ConvertibleToValue<int>&>();
  test<int, const ConvertibleToValue<int>&>();

  test<int, ExplicitlyConvertibleToValue<int>&>();
  test<int, const ExplicitlyConvertibleToValue<int>&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

class TerminatesOnConstruction
{
  int i_;

public:
  __host__ __device__ explicit TerminatesOnConstruction(int)
  {
    cuda::std::terminate();
  }

  __host__ __device__ friend bool operator==(const TerminatesOnConstruction& x, const TerminatesOnConstruction& y)
  {
    return x.i_ == y.i_;
  }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
class Z
{
  int i_;

public:
  __host__ __device__ explicit Z(int i)
      : i_(i)
  {
    TEST_THROW(6);
  }

  __host__ __device__ friend bool operator==(const Z& x, const Z& y)
  {
    return x.i_ == y.i_;
  }
};

template <class T, class U>
void test_exception(const optional<U>& rhs)
{
  try
  {
    optional<T> lhs(rhs);
    unused(lhs);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}

void test_exceptions()
{
  typedef Z T;
  typedef int U;
  optional<U> rhs(3);
  test_exception<T>(rhs);
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  {
    using T = TerminatesOnConstruction;
    const optional<int> rhs;
    optional<T> lhs{rhs};
    assert(!lhs.has_value());
    assert(!rhs.has_value());
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
