//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: true

// This test did pass but is very slow when run using qemu. ~7 minutes on a
// Neoverse N1 (AArch64) server core.
// REQUIRES: long_tests

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   void
//   sort(Iter first, Iter last);

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/numeric>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
struct nearly_vector
{
  using value_type = T;
  using iterator   = T*;

  __host__ __device__ nearly_vector(size_t N)
      : size_(N)
      , ptr_(::new T[size_])
  {
    for (size_t i = 0; i != size_; ++i)
    {
      ptr_[i] = T{};
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

template <class Container, class RI>
__host__ __device__ void test_sort_helper(RI f, RI l)
{
  if (f != l)
  {
    Container save(l - f);
    do
    {
      cuda::std::copy(f, l, save.begin());
      cuda::std::sort(save.begin(), save.end());
      assert(cuda::std::is_sorted(save.begin(), save.end()));
      assert(cuda::std::is_permutation(save.begin(), save.end(), f));
    } while (cuda::std::next_permutation(f, l));
  }
}

template <class T>
__host__ __device__ void set_value(T& dest, int value)
{
  dest = value;
}

__host__ __device__ void set_value(cuda::std::pair<int, int>& dest, int value)
{
  dest.first  = value;
  dest.second = value;
}

template <class Container, class RI>
__host__ __device__ void test_sort_driver_driver(RI f, RI l, int start, RI real_last)
{
  for (RI i = l; i > f + start;)
  {
    set_value(*--i, start);
    if (f == i)
    {
      test_sort_helper<Container>(f, real_last);
    }
    if (start > 0)
    {
      test_sort_driver_driver<Container>(f, i, start - 1, real_last);
    }
  }
}

template <class Container, class RI>
__host__ __device__ void test_sort_driver(RI f, RI l, int start)
{
  test_sort_driver_driver<Container>(f, l, start, l);
}

template <class Container, int sa>
__host__ __device__ void test_sort_()
{
  Container ia(sa);
  for (int i = 0; i < sa; ++i)
  {
    test_sort_driver<Container>(ia.begin(), ia.end(), i);
  }
}

template <class T>
__host__ __device__ T increment_or_reset(T value, int max_value)
{
  return value == max_value - 1 ? 0 : value + 1;
}

__host__ __device__ cuda::std::pair<int, int> increment_or_reset(cuda::std::pair<int, int> value, int max_value)
{
  int new_value = value.first + 1;
  if (new_value == max_value)
  {
    new_value = 0;
  }
  return cuda::std::make_pair(new_value, new_value);
}

template <class Container, int N>
__host__ __device__ void test_larger_sorts(int M)
{
  using Iter      = typename Container::iterator;
  using ValueType = typename Container::value_type;
  assert(N != 0);
  assert(M != 0);
  // create container of length N filled with M different objects
  Container array(N);
  ValueType x = ValueType();
  for (int i = 0; i < N; ++i)
  {
    array[i] = x;
    x        = increment_or_reset(x, M);
  }
  Container original = array;
  Iter iter          = array.begin();
  Iter original_iter = original.begin();

  cuda::std::minstd_rand randomness;

  // test saw tooth pattern
  cuda::std::sort(iter, iter + N);
  assert(cuda::std::is_sorted(iter, iter + N));
  assert(cuda::std::is_permutation(iter, iter + N, original_iter));
  // test random pattern
  cuda::std::shuffle(iter, iter + N, randomness);
  cuda::std::sort(iter, iter + N);
  assert(cuda::std::is_sorted(iter, iter + N));
  assert(cuda::std::is_permutation(iter, iter + N, original_iter));
  // test sorted pattern
  cuda::std::sort(iter, iter + N);
  assert(cuda::std::is_sorted(iter, iter + N));
  assert(cuda::std::is_permutation(iter, iter + N, original_iter));
  // test reverse sorted pattern
  cuda::std::reverse(iter, iter + N);
  cuda::std::sort(iter, iter + N);
  assert(cuda::std::is_sorted(iter, iter + N));
  assert(cuda::std::is_permutation(iter, iter + N, original_iter));
  // test swap ranges 2 pattern
  cuda::std::swap_ranges(iter, iter + N / 2, iter + N / 2);
  cuda::std::sort(iter, iter + N);
  assert(cuda::std::is_sorted(iter, iter + N));
  assert(cuda::std::is_permutation(iter, iter + N, original_iter));
  // test reverse swap ranges 2 pattern
  cuda::std::reverse(iter, iter + N);
  cuda::std::swap_ranges(iter, iter + N / 2, iter + N / 2);
  cuda::std::sort(iter, iter + N);
  assert(cuda::std::is_sorted(iter, iter + N));
  assert(cuda::std::is_permutation(iter, iter + N, original_iter));
}

template <class Container, int N>
__host__ __device__ void test_larger_sorts()
{
  test_larger_sorts<Container, N>(1);
  test_larger_sorts<Container, N>(2);
  test_larger_sorts<Container, N>(3);
  test_larger_sorts<Container, N>(N / 2 - 1);
  test_larger_sorts<Container, N>(N / 2);
  test_larger_sorts<Container, N>(N / 2 + 1);
  test_larger_sorts<Container, N>(N - 2);
  test_larger_sorts<Container, N>(N - 1);
  test_larger_sorts<Container, N>(N);
}

__host__ __device__ void test_pointer_sort()
{
  static const int array_size = 10;
  const int v[array_size]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int* pv[array_size];
  for (int i = 0; i < array_size; i++)
  {
    pv[i] = &v[array_size - 1 - i];
  }
  cuda::std::sort(pv, pv + array_size);
  assert(*pv[0] == v[0]);
  assert(*pv[1] == v[1]);
  assert(*pv[array_size - 1] == v[array_size - 1]);
}

template <class Container>
__host__ __device__ void run_sort_tests()
{
  // test null range
  using ValueType = typename Container::value_type;
  ValueType d     = ValueType();
  cuda::std::sort(&d, &d);

  // exhaustively test all possibilities up to length 8
  test_sort_<Container, 1>();
  test_sort_<Container, 2>();
  test_sort_<Container, 3>();
  test_sort_<Container, 4>();
  test_sort_<Container, 5>();
  test_sort_<Container, 6>();
  test_sort_<Container, 7>();
  test_sort_<Container, 8>();

  test_larger_sorts<Container, 256>();
  test_larger_sorts<Container, 257>();
}

int main(int, char**)
{
  // test various combinations of contiguous/non-contiguous containers with
  // arithmetic/non-arithmetic types
  run_sort_tests<nearly_vector<int>>();
  run_sort_tests<nearly_vector<cuda::std::pair<int, int>>>();

  test_pointer_sort();

  return 0;
}
