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
//   optional(const optional<U>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

class X
{
  int i_;

public:
  __host__ __device__ constexpr X(int i)
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
  __host__ __device__ constexpr Y(int i)
      : i_(i)
  {}

  __host__ __device__ friend constexpr bool operator==(const Y& x, const Y& y)
  {
    return x.i_ == y.i_;
  }
};

struct B
{
  int val_;

  __host__ __device__ constexpr bool operator==(const int& other) const noexcept
  {
    return other == val_;
  }
};
class D : public B
{};

#ifdef CCCL_ENABLE_OPTIONAL_REF
template <class T>
struct ConvertibleToReference
{
  T val_;

  __host__ __device__ constexpr operator const T&() const noexcept
  {
    return val_;
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
#endif // CCCL_ENABLE_OPTIONAL_REF

template <class T, class U>
__host__ __device__ constexpr void test()
{
  { // constructed from empty
    const optional<U> input{};
    optional<T> opt = input;
    assert(!input.has_value());
    assert(!opt.has_value());
  }
  { // constructed from non-empty
    cuda::std::remove_reference_t<U> val{42};
    const optional<U> input{val};
    optional<T> opt = input;
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == 42);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      // optional<U> does not necessarily hold a reference so we cannot use addressof(val)
      assert(cuda::std::addressof(static_cast<T>(*input)) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, short>();
  test<X, int>();
  test<Y, int>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<B&, D&>();
  test<const int&, ConvertibleToReference<int>>();

  test<int, ConvertibleToReference<int>&>();
  test<int, ConvertibleToValue<int>&>();
  test<int, const ConvertibleToValue<int>&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

class TerminatesOnConstruction
{
  int i_;

public:
  __host__ __device__ TerminatesOnConstruction(int)
  {
    cuda::std::terminate();
  }

  __host__ __device__ friend bool operator==(const TerminatesOnConstruction& x, const TerminatesOnConstruction& y)
  {
    return x.i_ == y.i_;
  }
};

#if TEST_HAS_EXCEPTIONS()
class Z
{
  int i_;

public:
  Z(int i)
      : i_(i)
  {
    TEST_THROW(6);
  }

  friend bool operator==(const Z& x, const Z& y)
  {
    return x.i_ == y.i_;
  }
};

template <class T, class U>
void test_exception(const optional<U>& rhs)
{
  try
  {
    optional<T> lhs = rhs;
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
  optional<U> rhs(U{3});
  test_exception<T>(rhs);
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  {
    typedef TerminatesOnConstruction T;
    typedef int U;
    const optional<U> rhs;
    optional<T> lhs{rhs};
    assert(!lhs.has_value());
    assert(!rhs.has_value());
  }
  static_assert(!(cuda::std::is_constructible<optional<X>, const optional<Y>&>::value), "");

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
