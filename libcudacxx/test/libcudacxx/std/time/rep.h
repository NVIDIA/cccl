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
  __host__ __device__ constexpr Rep()
      : data_(-1)
  {}
  __host__ __device__ explicit constexpr Rep(int i)
      : data_(i)
  {}

  __host__ __device__ bool constexpr operator==(int i) const
  {
    return data_ == i;
  }
  __host__ __device__ bool constexpr operator==(const Rep& r) const
  {
    return data_ == r.data_;
  }

  __host__ __device__ Rep& operator*=(Rep x)
  {
    data_ *= x.data_;
    return *this;
  }
  __host__ __device__ Rep& operator/=(Rep x)
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
  __host__ __device__ operator long() = delete;
  __host__ __device__ operator long() const
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
__host__ __device__ cuda::std::chrono::duration<Rep, Period>
operator*(cuda::std::chrono::duration<Rep, Period> d, NotARep)
{
  return d;
}

template <class Rep, class Period>
__host__ __device__ cuda::std::chrono::duration<Rep, Period>
operator*(NotARep, cuda::std::chrono::duration<Rep, Period> d)
{
  return d;
}

template <class Rep, class Period>
__host__ __device__ cuda::std::chrono::duration<Rep, Period>
operator/(cuda::std::chrono::duration<Rep, Period> d, NotARep)
{
  return d;
}

template <class Rep, class Period>
__host__ __device__ cuda::std::chrono::duration<Rep, Period>
operator%(cuda::std::chrono::duration<Rep, Period> d, NotARep)
{
  return d;
}

// op= is not commutative.
template <class Rep, class Period>
__host__ __device__ cuda::std::chrono::duration<Rep, Period>&
operator*=(cuda::std::chrono::duration<Rep, Period>& d, NotARep)
{
  return d;
}

template <class Rep, class Period>
__host__ __device__ cuda::std::chrono::duration<Rep, Period>&
operator/=(cuda::std::chrono::duration<Rep, Period>& d, NotARep)
{
  return d;
}

template <class Rep, class Period>
__host__ __device__ cuda::std::chrono::duration<Rep, Period>&
operator%=(cuda::std::chrono::duration<Rep, Period>& d, NotARep)
{
  return d;
}

#endif // TEST_CUDA_STD_TIME_REP_H
