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

// template<BidirectionalIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Compare>
//   void
//   inplace_merge(Iter first, Iter middle, Iter last, Compare comp);

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/memory>
#include <cuda/std/span>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR

#include "counting_predicates.h"
#include "test_iterators.h"
#include "test_macros.h"

struct indirect_less
{
  template <class P>
  __host__ __device__ constexpr bool operator()(const P& x, const P& y) const noexcept
  {
    return *x < *y;
  }
};

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
  __host__ __device__ bool operator>(const S& rhs) const
  {
    return i_ > rhs.i_;
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
  cuda::std::sort(ptr, ptr + M, cuda::std::greater<value_type>());
  cuda::std::sort(ptr + M, ptr + N, cuda::std::greater<value_type>());
  binary_counting_predicate<cuda::std::greater<value_type>, value_type, value_type> pred(
    (cuda::std::greater<value_type>()));
  cuda::std::inplace_merge(Iter(ptr), Iter(ptr + M), Iter(ptr + N), cuda::std::ref(pred));
  if (N > 0)
  {
    assert(ptr[0] == static_cast<int>(N) - 1);
    assert(ptr[N - 1] == 0);
    assert(cuda::std::is_sorted(ptr, ptr + N, cuda::std::greater<value_type>()));
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
  test<Iter>(20, ptr, randomness);
  test<Iter>(100, ptr, randomness);
  // test<Iter>(1000);
}

struct less_by_first
{
  template <typename Pair>
  __host__ __device__ bool operator()(const Pair& lhs, const Pair& rhs) const
  {
    return cuda::std::less<typename Pair::first_type>()(lhs.first, rhs.first);
  }
};

#if defined(_LIBCUDACXX_HAS_VECTOR)
__host__ __device__ void test_PR31166()
{
  using P  = cuda::std::pair<int, int>;
  using V  = cuda::std::vector<P>;
  P vec[5] = {P(1, 0), P(2, 0), P(2, 1), P(2, 2), P(2, 3)};
  for (int i = 0; i < 5; ++i)
  {
    V res(vec, vec + 5);
    cuda::std::inplace_merge(res.begin(), res.begin() + i, res.end(), less_by_first());
    assert(res.size() == 5);
    assert(cuda::std::equal(res.begin(), res.end(), vec));
  }
}
#endif // _LIBCUDACXX_HAS_VECTOR

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
  test_PR31166();
#endif // _LIBCUDACXX_HAS_VECTOR

  delete[] arr;
  return 0;
}
