//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_USER_DEFINED_INTEGRAL_H
#define SUPPORT_USER_DEFINED_INTEGRAL_H

template <class T>
struct UserDefinedIntegral
{
  __host__ __device__ constexpr UserDefinedIntegral()
      : value(0)
  {}
  __host__ __device__ constexpr UserDefinedIntegral(T v)
      : value(v)
  {}
  __host__ __device__ constexpr operator T() const
  {
    return value;
  }
  T value;
};

// Poison the arithmetic and comparison operations
template <class T, class U>
__host__ __device__ constexpr void operator+(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator-(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator*(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator/(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator==(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator!=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator<(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator>(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator<=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator>=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

#endif // SUPPORT_USER_DEFINED_INTEGRAL_H
