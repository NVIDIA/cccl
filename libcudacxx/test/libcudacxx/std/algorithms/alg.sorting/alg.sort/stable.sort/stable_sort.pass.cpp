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

// template <class RandomAccessIterator>
//     constexpr void               // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last);

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=200000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/utility>

#include "count_new.h"
#include "test_macros.h"

template <class T>
struct nearly_vector
{
  __host__ __device__ nearly_vector(size_t N)
      : size_(N)
      , ptr_(::new T[size_])
  {
    for (size_t i = 0; i != size_; ++i)
    {
      ptr_[i] = 0;
    }
  }

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

template <class T = int>
__host__ __device__ nearly_vector<T> generate_sawtooth(int N, int M)
{
  // Populate a sequence of length N with M different numbers
  nearly_vector<T> v(N);
  T x = 0;
  for (int i = 0; i < N; ++i)
  {
    v[i] = x;
    if (++x == M)
    {
      x = 0;
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
    cuda::std::stable_sort(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  // test random pattern
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::minstd_rand randomness;
    cuda::std::shuffle(v.begin(), v.end(), randomness);
    cuda::std::stable_sort(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  // test sorted pattern
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::sort(v.begin(), v.end());

    cuda::std::stable_sort(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  // test reverse sorted pattern
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::sort(v.begin(), v.end());
    cuda::std::reverse(v.begin(), v.end());

    cuda::std::stable_sort(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  // test swap ranges 2 pattern
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::sort(v.begin(), v.end());
    cuda::std::swap_ranges(v.begin(), v.begin() + (N / 2), v.begin() + (N / 2));

    cuda::std::stable_sort(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  // test reverse swap ranges 2 pattern
  {
    auto v = generate_sawtooth(N, M);
    cuda::std::sort(v.begin(), v.end());
    cuda::std::reverse(v.begin(), v.end());
    cuda::std::swap_ranges(v.begin(), v.begin() + (N / 2), v.begin() + (N / 2));

    cuda::std::stable_sort(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
