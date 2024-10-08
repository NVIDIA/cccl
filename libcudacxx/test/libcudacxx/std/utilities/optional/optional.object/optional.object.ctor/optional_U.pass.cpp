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

// template <class U>
//   optional(optional<U>&& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using cuda::std::optional;

template <class T, class U>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(optional<U>&& rhs)
{
  bool rhs_engaged = static_cast<bool>(rhs);
  optional<T> lhs  = cuda::std::move(rhs);
  assert(static_cast<bool>(lhs) == rhs_engaged);
}

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

struct TerminatesOnConstruction
{
  __host__ __device__ TerminatesOnConstruction(int)
  {
    cuda::std::terminate();
  }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Z
{
  Z(int)
  {
    TEST_THROW(6);
  }
};

template <class T, class U>
__host__ __device__ void test_exception(optional<U>&& rhs)
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
#endif // !TEST_HAS_NO_EXCEPTIONS

template <class T, class U>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_all()
{
  {
    optional<T> rhs;
    test<U>(cuda::std::move(rhs));
  }
  {
    optional<T> rhs(short{3});
    test<U>(cuda::std::move(rhs));
  }
  return true;
}

int main(int, char**)
{
  test_all<short, int>();
  test_all<int, X>();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test_all<short, int>());
  static_assert(test_all<int, X>());
#endif
  {
    optional<int> rhs;
    test<TerminatesOnConstruction>(cuda::std::move(rhs));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
  }
#endif // !TEST_HAS_NO_EXCEPTIONS

  static_assert(!(cuda::std::is_constructible<optional<X>, optional<TerminatesOnConstruction>>::value), "");

  return 0;
}
