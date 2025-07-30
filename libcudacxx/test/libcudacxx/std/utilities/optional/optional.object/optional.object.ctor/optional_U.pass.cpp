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
//   optional(optional<U>&& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using cuda::std::optional;

class X
{
  int i_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX20 X(int i)
      : i_(i)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 X(X&& x)
      : i_(cuda::std::exchange(x.i_, 0))
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

  __host__ __device__ constexpr operator T&() noexcept
  {
    return val_;
  }
};
#endif // CCCL_ENABLE_OPTIONAL_REF

template <class T, class U>
__host__ __device__ constexpr void test()
{
  { // constructed from empty
    optional<U> input{};
    optional<T> opt = cuda::std::move(input);
    assert(!input.has_value());
    assert(!opt.has_value());
  }
  { // constructed from non-empty
    cuda::std::remove_reference_t<U> val{42};
    optional<U> input{val};
    optional<T> opt = cuda::std::move(input);
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == 42);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(cuda::std::addressof(static_cast<T>(val)) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, short>();
  test<X, int>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<B&, D&>();
  test<int&, ConvertibleToReference<int>&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

struct TerminatesOnConstruction
{
  __host__ __device__ TerminatesOnConstruction(int)
  {
    cuda::std::terminate();
  }
};

#if TEST_HAS_EXCEPTIONS()
struct Z
{
  Z(int)
  {
    TEST_THROW(6);
  }
};

template <class T, class U>
void test_exception(optional<U>&& rhs)
{
  try
  {
    optional<T> lhs = cuda::std::move(rhs);
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
  optional<int> rhs(3);
  test_exception<Z>(cuda::std::move(rhs));
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  {
    using T = TerminatesOnConstruction;
    optional<int> input{};
    optional<T> lhs{cuda::std::move(input)};
    assert(!input.has_value());
    assert(!lhs.has_value());
  }

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  static_assert(!(cuda::std::is_constructible<optional<X>, optional<TerminatesOnConstruction>>::value), "");

  return 0;
}
