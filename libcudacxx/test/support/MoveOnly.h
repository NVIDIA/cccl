//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include <cuda/std/cstddef>

#include "test_macros.h"
// #include <functional>

class MoveOnly
{
  int data_;

public:
  __host__ __device__ TEST_CONSTEXPR MoveOnly(int data = 1)
      : data_(data)
  {}

  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR int get() const
  {
    return data_;
  }

  __host__ __device__ friend TEST_CONSTEXPR bool operator==(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator!=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ != y.data_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator<(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ < y.data_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator<=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ <= y.data_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator>(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ > y.data_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator>=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ >= y.data_;
  }

#if TEST_STD_VER > 2017 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
  __host__ __device__ friend constexpr auto operator<=>(const MoveOnly&, const MoveOnly&) = default;
#endif // TEST_STD_VER > 2017 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)

  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator+(const MoveOnly& x) const
  {
    return MoveOnly(data_ + x.data_);
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator*(const MoveOnly& x) const
  {
    return MoveOnly(data_ * x.data_);
  }

  template <class T>
  void operator,(T const&) = delete;
};

/*
template <>
struct cuda::std::hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    __host__ __device__ TEST_CONSTEXPR size_t operator()(const MoveOnly& x) const {return x.get();}
};
*/

#endif // MOVEONLY_H
