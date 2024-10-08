//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct NonTrivialCopy
{
  int data                = 0;
  bool copy_assigned_from = false;

  NonTrivialCopy() = default;

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy(NonTrivialCopy&& other) noexcept
      : data(other.data)
      , copy_assigned_from(false)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy(const NonTrivialCopy& other) noexcept
      : data(other.data)
      , copy_assigned_from(false)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy& operator=(const NonTrivialCopy& other) noexcept
  {
    data               = other.data;
    copy_assigned_from = true;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy& operator=(NonTrivialCopy&& other) noexcept
  {
    data               = other.data;
    copy_assigned_from = false;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy(const int val) noexcept
      : data(val)
      , copy_assigned_from(false)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialCopy& operator=(const int val) noexcept
  {
    data               = val;
    copy_assigned_from = false;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 friend bool
  operator==(const NonTrivialCopy& lhs, const NonTrivialCopy& rhs) noexcept
  {
    // NOTE: This uses implicit knowledge that the right hand side has been copied from
    return lhs.data == rhs.data && !lhs.copy_assigned_from && rhs.copy_assigned_from;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 bool operator==(const int& other) const noexcept
  {
    // NOTE: This uses implicit knowledge that the only elements we compare against were untouched
    return data == other && !copy_assigned_from;
  }
};

struct NonTrivialDestructor
{
  int data = 0;

  NonTrivialDestructor() = default;

  NonTrivialDestructor(NonTrivialDestructor&&) noexcept                 = default;
  NonTrivialDestructor(const NonTrivialDestructor&) noexcept            = default;
  NonTrivialDestructor& operator=(NonTrivialDestructor&&) noexcept      = default;
  NonTrivialDestructor& operator=(const NonTrivialDestructor&) noexcept = default;
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~NonTrivialDestructor() noexcept {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialDestructor(const int val) noexcept
      : data(val)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 NonTrivialDestructor& operator=(const int val) noexcept
  {
    data = val;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 friend bool
  operator==(const NonTrivialDestructor& lhs, const NonTrivialDestructor& rhs) noexcept
  {
    return lhs.data == rhs.data;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 bool operator==(const int& other) const noexcept
  {
    return data == other;
  }
};

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 __host__ __device__ void test()
{
  using value_type = typename cuda::std::iterator_traits<InIter>::value_type;
  {
    constexpr int N  = 1000;
    value_type ia[N] = {0};
    for (int i = 0; i < N; ++i)
    {
      ia[i] = i;
    }
    value_type ib[N] = {0};

    OutIter r = cuda::std::copy(InIter(ia), InIter(ia + N), OutIter(ib));
    assert(base(r) == ib + N);
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == ib[i]);
    }
  }
  {
    constexpr int N      = 5;
    value_type ia[N]     = {1, 2, 3, 4, 5};
    value_type ic[N + 2] = {6, 6, 6, 6, 6, 6, 6};

    auto p = cuda::std::copy(ia, ia + 5, ic);
    assert(p == (ic + N));
    for (int i = 0; i < N; ++i)
    {
      assert(ia[i] == ic[i]);
    }

    for (int i = N; i < N + 2; ++i)
    {
      assert(ic[i] == 6);
    }
  }
}
