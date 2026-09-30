//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_STD_TIME_REP_H
#define TEST_CUDA_STD_TIME_REP_H

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

class Rep
{
  int data_;

public:
  TEST_FUNC constexpr Rep()
      : data_(-1)
  {}
  TEST_FUNC explicit constexpr Rep(int i)
      : data_(i)
  {}

  TEST_FUNC bool constexpr operator==(int i) const
  {
    return data_ == i;
  }
  TEST_FUNC bool constexpr operator==(const Rep& r) const
  {
    return data_ == r.data_;
  }

  TEST_FUNC Rep& operator*=(Rep x)
  {
    data_ *= x.data_;
    return *this;
  }
  TEST_FUNC Rep& operator/=(Rep x)
  {
    data_ /= x.data_;
    return *this;
  }
};

// This is PR#41130

struct NotARep
{};

// Several duration operators take a Rep parameter. Before LWG3050 this
// parameter was constrained to be convertible from a non-const object,
// but the code always uses a const object. So the function was SFINAE'd
// away for this type. LWG3050 fixes the constraint to use a const
// object.
struct RepConstConvertibleLWG3050
{
  TEST_FUNC operator long() = delete;
  TEST_FUNC operator long() const
  {
    return 2;
  }
};

template <>
struct cuda::std::common_type<RepConstConvertibleLWG3050, int>
{
  using type = long;
};
template <>
struct cuda::std::common_type<int, RepConstConvertibleLWG3050>
{
  using type = long;
};

// cuda::std::chrono:::duration has only '*', '/' and '%' taking a "Rep" parameter

// Multiplication is commutative, division is not.
template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period> operator*(cuda::std::chrono::duration<Rep, Period> d, NotARep)
{
  return d;
}

template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period> operator*(NotARep, cuda::std::chrono::duration<Rep, Period> d)
{
  return d;
}

template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period> operator/(cuda::std::chrono::duration<Rep, Period> d, NotARep)
{
  return d;
}

template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period> operator%(cuda::std::chrono::duration<Rep, Period> d, NotARep)
{
  return d;
}

// op= is not commutative.
template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period>& operator*=(cuda::std::chrono::duration<Rep, Period>& d, NotARep)
{
  return d;
}

template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period>& operator/=(cuda::std::chrono::duration<Rep, Period>& d, NotARep)
{
  return d;
}

template <class Rep, class Period>
TEST_FUNC cuda::std::chrono::duration<Rep, Period>& operator%=(cuda::std::chrono::duration<Rep, Period>& d, NotARep)
{
  return d;
}

#endif // TEST_CUDA_STD_TIME_REP_H
