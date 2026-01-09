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

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter> && CopyConstructible<Compare>
//   constexpr void stable_sort(Iter first, Iter last, Compare comp); // constexpr since C++26
//
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=200000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/memory>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
struct nearly_vector
{
  __host__ __device__ nearly_vector(size_t N)
      : size_(N)
      , ptr_(::new T[size_])
  {}

  __host__ __device__ ~nearly_vector()
  {
    ::delete[] ptr_;
  }

  __host__ __device__ T* begin() noexcept
  {
    return ptr_;
  }
  __host__ __device__ const T* begin() const noexcept
  {
    return ptr_;
  }
  __host__ __device__ T* end() noexcept
  {
    return ptr_ + size_;
  }
  __host__ __device__ const T* end() const noexcept
  {
    return ptr_ + size_;
  }

  __host__ __device__ size_t size() const noexcept
  {
    return size_;
  }

  __host__ __device__ T& operator[](size_t index) noexcept
  {
    return ptr_[index];
  }
  __host__ __device__ const T& operator[](size_t index) const noexcept
  {
    return ptr_[index];
  }

  size_t size_;
  T* ptr_;
};

struct indirect_less
{
  template <class P>
  __host__ __device__ bool operator()(const P& x, const P& y) const
  {
    return *x < *y;
  }
};

struct first_only
{
  __host__ __device__ bool operator()(const cuda::std::pair<int, int>& x, const cuda::std::pair<int, int>& y) const
  {
    return x.first < y.first;
  }
};

using Pair = cuda::std::pair<int, int>;

__host__ __device__ nearly_vector<Pair> generate_sawtooth(int N, int M)
{
  nearly_vector<Pair> v(N);
  int x   = 0;
  int ver = 0;
  for (int i = 0; i < N; ++i)
  {
    v[i] = Pair(x, ver);
    if (++x == M)
    {
      x = 0;
      ++ver;
    }
  }
  return v;
}

__host__ __device__ bool test()
{
  int const N = 1000;
  int const M = 10;

  // test sawtooth pattern
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::stable_sort(v.begin(), v.end(), first_only());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  // Test sorting a sequence where subsequences of elements are not sorted with <,
  // but everything is already sorted with respect to the first element. This ensures
  // that we don't change the order of "equivalent" elements.
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::minstd_rand randomness;
    for (int i = 0; i < N - M; i += M)
    {
      cuda::std::shuffle(v.begin() + i, v.begin() + i + M, randomness);
    }
    cuda::std::stable_sort(v.begin(), v.end(), first_only());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  {
    nearly_vector<cuda::std::unique_ptr<int>> v(1000);
    for (int i = 0; static_cast<cuda::std::size_t>(i) < v.size(); ++i)
    {
      v[i].reset(new int(i));
    }
    cuda::std::stable_sort(v.begin(), v.end(), indirect_less());
    assert(cuda::std::is_sorted(v.begin(), v.end(), indirect_less()));
    assert(*v[0] == 0);
    assert(*v[1] == 1);
    assert(*v[2] == 2);
  }

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
