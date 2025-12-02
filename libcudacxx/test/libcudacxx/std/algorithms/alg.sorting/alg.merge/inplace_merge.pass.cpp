//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: true

// <algorithm>

// template<BidirectionalIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   void
//   inplace_merge(Iter first, Iter middle, Iter last);

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/span>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR

#include <nv/target>

#include "count_new.h"
#include "test_iterators.h"
#include "test_macros.h"

struct S
{
  __host__ __device__ S()
      : i_(0)
  {}
  __host__ __device__ S(int i)
      : i_(i)
  {}

  __host__ __device__ S(const S& rhs)
      : i_(rhs.i_)
  {}
  __host__ __device__ S(S&& rhs)
      : i_(rhs.i_)
  {
    rhs.i_ = -1;
  }

  __host__ __device__ S& operator=(const S& rhs)
  {
    i_ = rhs.i_;
    return *this;
  }
  __host__ __device__ S& operator=(S&& rhs)
  {
    i_     = rhs.i_;
    rhs.i_ = -2;
    assert(this != &rhs);
    return *this;
  }
  __host__ __device__ S& operator=(int i)
  {
    i_ = i;
    return *this;
  }

  __host__ __device__ bool operator<(const S& rhs) const
  {
    return i_ < rhs.i_;
  }
  __host__ __device__ bool operator==(const S& rhs) const
  {
    return i_ == rhs.i_;
  }
  __host__ __device__ bool operator==(int i) const
  {
    return i_ == i;
  }

  __host__ __device__ void set(int i)
  {
    i_ = i;
  }

  int i_;
};

template <class Iter, class value_type = typename cuda::std::iterator_traits<Iter>::value_type*>
__host__ __device__ void test_one(unsigned N, unsigned M, value_type* ptr, cuda::std::minstd_rand& randomness)
{
  assert(M <= N);

  cuda::std::shuffle(ptr, ptr + N, randomness);
  cuda::std::sort(ptr, ptr + M);
  cuda::std::sort(ptr + M, ptr + N);
  cuda::std::inplace_merge(Iter(ptr), Iter(ptr + M), Iter(ptr + N));
  if (N > 0)
  {
    assert(ptr[0] == 0);
    assert(ptr[N - 1] == static_cast<value_type>(N - 1));
    assert(cuda::std::is_sorted(ptr, ptr + N));
  }
}

template <class Iter, class value_type = typename cuda::std::iterator_traits<Iter>::value_type*>
__host__ __device__ void test(unsigned N, value_type* ptr, cuda::std::minstd_rand& randomness)
{
  test_one<Iter>(N, 0, ptr, randomness);
  test_one<Iter>(N, N / 4, ptr, randomness);
  test_one<Iter>(N, N / 2, ptr, randomness);
  test_one<Iter>(N, 3 * N / 4, ptr, randomness);
  test_one<Iter>(N, N, ptr, randomness);
}

template <class Iter, class value_type = typename cuda::std::iterator_traits<Iter>::value_type*>
__host__ __device__ void test(value_type* ptr, cuda::std::minstd_rand& randomness)
{
  test_one<Iter>(1, 0, ptr, randomness);
  test_one<Iter>(1, 1, ptr, randomness);
  test_one<Iter>(2, 0, ptr, randomness);
  test_one<Iter>(2, 1, ptr, randomness);
  test_one<Iter>(2, 2, ptr, randomness);
  test_one<Iter>(3, 0, ptr, randomness);
  test_one<Iter>(3, 1, ptr, randomness);
  test_one<Iter>(3, 2, ptr, randomness);
  test_one<Iter>(3, 3, ptr, randomness);
  test<Iter>(4, ptr, randomness);
  test<Iter>(100, ptr, randomness);
  // test<Iter>(1000);
}

int main(int, char**)
{
  constexpr size_t N = 100;
  int* arr           = new int[N];
  for (size_t i = 0; i < N; ++i)
  {
    arr[i] = int(i);
  }
  cuda::std::minstd_rand randomness{};

  test<bidirectional_iterator<int*>>(arr, randomness);
  test<random_access_iterator<int*>>(arr, randomness);
  test<int*>(arr, randomness);

  test<bidirectional_iterator<S*>>(reinterpret_cast<S*>(arr), randomness);
  test<random_access_iterator<S*>>(reinterpret_cast<S*>(arr), randomness);
  test<S*>(reinterpret_cast<S*>(arr), randomness);

#if defined(_LIBCUDACXX_HAS_VECTOR)
#  if !defined(TEST_HAS_NO_EXCEPTIONS)
  {
    cuda::std::vector<int> vec(150, 3);
    getGlobalMemCounter()->throw_after = 0;
    cuda::std::inplace_merge(vec.begin(), vec.begin() + 100, vec.end());
    assert(cuda::std::all_of(vec.begin(), vec.end(), [](int i) {
      return i == 3;
    }));
  }
#  endif // !defined(TEST_HAS_NO_EXCEPTIONS)
#endif // _LIBCUDACXX_HAS_VECTOR

  delete[] arr;

  return 0;
}
