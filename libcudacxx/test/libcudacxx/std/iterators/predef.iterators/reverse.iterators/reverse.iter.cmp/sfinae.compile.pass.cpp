//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <cuda/std/iterator>
//
// reverse_iterator
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator==(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator!=(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator<(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator>(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator<=(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template <class Iterator1, class Iterator2>
// constexpr bool                          // constexpr in C++17
// operator>=(const reverse_iterator<Iterator1>& x, const reverse_iterator<Iterator2>& y);
//
// template<class Iterator1, three_way_comparable_with<Iterator1> Iterator2>
//  constexpr compare_three_way_result_t<Iterator1, Iterator2>
//    operator<=>(const reverse_iterator<Iterator1>& x,
//                const reverse_iterator<Iterator2>& y);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

struct IterBase
{
  using iterator_category = cuda::std::bidirectional_iterator_tag;
  using value_type        = int;
  using difference_type   = ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
};

template <class T>
concept HasEqual = requires(T t) { t == t; };
template <class T>
concept HasNotEqual = requires(T t) { t != t; };
template <class T>
concept HasLess = requires(T t) { t < t; };
template <class T>
concept HasLessOrEqual = requires(T t) { t <= t; };
template <class T>
concept HasGreater = requires(T t) { t > t; };
template <class T>
concept HasGreaterOrEqual = requires(T t) { t >= t; };
template <class T>
concept HasSpaceship = requires(T t) { t <=> t; };

// operator ==

struct NoEqualityCompIter : IterBase
{
  __host__ __device__ bool operator!=(NoEqualityCompIter) const;
  __host__ __device__ bool operator<(NoEqualityCompIter) const;
  __host__ __device__ bool operator>(NoEqualityCompIter) const;
  __host__ __device__ bool operator<=(NoEqualityCompIter) const;
  __host__ __device__ bool operator>=(NoEqualityCompIter) const;
};

static_assert(HasEqual<cuda::std::reverse_iterator<int*>>);
static_assert(!HasEqual<cuda::std::reverse_iterator<NoEqualityCompIter>>);
static_assert(HasNotEqual<cuda::std::reverse_iterator<NoEqualityCompIter>>);
static_assert(HasLess<cuda::std::reverse_iterator<NoEqualityCompIter>>);
static_assert(HasLessOrEqual<cuda::std::reverse_iterator<NoEqualityCompIter>>);
static_assert(HasGreater<cuda::std::reverse_iterator<NoEqualityCompIter>>);
static_assert(HasGreaterOrEqual<cuda::std::reverse_iterator<NoEqualityCompIter>>);

__host__ __device__ void Foo()
{
  cuda::std::reverse_iterator<NoEqualityCompIter> i;
  unused(i);
}

// operator !=

struct NoInequalityCompIter : IterBase
{
  __host__ __device__ bool operator<(NoInequalityCompIter) const;
  __host__ __device__ bool operator>(NoInequalityCompIter) const;
  __host__ __device__ bool operator<=(NoInequalityCompIter) const;
  __host__ __device__ bool operator>=(NoInequalityCompIter) const;
};

static_assert(HasNotEqual<cuda::std::reverse_iterator<int*>>);
static_assert(!HasNotEqual<cuda::std::reverse_iterator<NoInequalityCompIter>>);
static_assert(!HasEqual<cuda::std::reverse_iterator<NoInequalityCompIter>>);
static_assert(HasLess<cuda::std::reverse_iterator<NoInequalityCompIter>>);
static_assert(HasLessOrEqual<cuda::std::reverse_iterator<NoInequalityCompIter>>);
static_assert(HasGreater<cuda::std::reverse_iterator<NoInequalityCompIter>>);
static_assert(HasGreaterOrEqual<cuda::std::reverse_iterator<NoInequalityCompIter>>);

// operator <

struct NoGreaterCompIter : IterBase
{
  __host__ __device__ bool operator==(NoGreaterCompIter) const;
  __host__ __device__ bool operator!=(NoGreaterCompIter) const;
  __host__ __device__ bool operator<(NoGreaterCompIter) const;
  __host__ __device__ bool operator<=(NoGreaterCompIter) const;
  __host__ __device__ bool operator>=(NoGreaterCompIter) const;
};

static_assert(HasLess<cuda::std::reverse_iterator<int*>>);
static_assert(!HasLess<cuda::std::reverse_iterator<NoGreaterCompIter>>);
static_assert(HasEqual<cuda::std::reverse_iterator<NoGreaterCompIter>>);
static_assert(HasNotEqual<cuda::std::reverse_iterator<NoGreaterCompIter>>);
static_assert(HasLessOrEqual<cuda::std::reverse_iterator<NoGreaterCompIter>>);
static_assert(HasGreater<cuda::std::reverse_iterator<NoGreaterCompIter>>);
static_assert(HasGreaterOrEqual<cuda::std::reverse_iterator<NoGreaterCompIter>>);

// operator >

struct NoLessCompIter : IterBase
{
  __host__ __device__ bool operator==(NoLessCompIter) const;
  __host__ __device__ bool operator!=(NoLessCompIter) const;
  __host__ __device__ bool operator>(NoLessCompIter) const;
  __host__ __device__ bool operator<=(NoLessCompIter) const;
  __host__ __device__ bool operator>=(NoLessCompIter) const;
};

static_assert(HasGreater<cuda::std::reverse_iterator<int*>>);
static_assert(!HasGreater<cuda::std::reverse_iterator<NoLessCompIter>>);
static_assert(HasEqual<cuda::std::reverse_iterator<NoLessCompIter>>);
static_assert(HasNotEqual<cuda::std::reverse_iterator<NoLessCompIter>>);
static_assert(HasLess<cuda::std::reverse_iterator<NoLessCompIter>>);
static_assert(HasLessOrEqual<cuda::std::reverse_iterator<NoLessCompIter>>);
static_assert(HasGreaterOrEqual<cuda::std::reverse_iterator<NoLessCompIter>>);

// operator <=

struct NoGreaterOrEqualCompIter : IterBase
{
  __host__ __device__ bool operator==(NoGreaterOrEqualCompIter) const;
  __host__ __device__ bool operator!=(NoGreaterOrEqualCompIter) const;
  __host__ __device__ bool operator<(NoGreaterOrEqualCompIter) const;
  __host__ __device__ bool operator>(NoGreaterOrEqualCompIter) const;
  __host__ __device__ bool operator<=(NoGreaterOrEqualCompIter) const;
};

static_assert(HasLessOrEqual<cuda::std::reverse_iterator<int*>>);
static_assert(!HasLessOrEqual<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(HasEqual<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(HasNotEqual<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(HasLess<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(HasGreater<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(HasGreaterOrEqual<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);

// operator >=

struct NoLessOrEqualCompIter : IterBase
{
  __host__ __device__ bool operator==(NoLessOrEqualCompIter) const;
  __host__ __device__ bool operator!=(NoLessOrEqualCompIter) const;
  __host__ __device__ bool operator<(NoLessOrEqualCompIter) const;
  __host__ __device__ bool operator>(NoLessOrEqualCompIter) const;
  __host__ __device__ bool operator>=(NoLessOrEqualCompIter) const;
};

static_assert(HasGreaterOrEqual<cuda::std::reverse_iterator<int*>>);
static_assert(!HasGreaterOrEqual<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert(HasEqual<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert(HasNotEqual<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert(HasLess<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert(HasLessOrEqual<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);
static_assert(HasGreater<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
// operator <=>
static_assert(cuda::std::three_way_comparable_with<int*, int*>);
static_assert(HasSpaceship<cuda::std::reverse_iterator<int*>>);
static_assert(!cuda::std::three_way_comparable_with<NoEqualityCompIter, NoEqualityCompIter>);
static_assert(!HasSpaceship<cuda::std::reverse_iterator<NoEqualityCompIter>>);
static_assert(!cuda::std::three_way_comparable_with<NoInequalityCompIter, NoInequalityCompIter>);
static_assert(!HasSpaceship<cuda::std::reverse_iterator<NoInequalityCompIter>>);
static_assert(!cuda::std::three_way_comparable_with<NoGreaterCompIter, NoGreaterCompIter>);
static_assert(!HasSpaceship<cuda::std::reverse_iterator<NoGreaterCompIter>>);
static_assert(!cuda::std::three_way_comparable_with<NoLessCompIter, NoLessCompIter>);
static_assert(!HasSpaceship<cuda::std::reverse_iterator<NoLessCompIter>>);
static_assert(!cuda::std::three_way_comparable_with<NoGreaterOrEqualCompIter, NoGreaterOrEqualCompIter>);
static_assert(!HasSpaceship<cuda::std::reverse_iterator<NoGreaterOrEqualCompIter>>);
static_assert(!cuda::std::three_way_comparable_with<NoLessOrEqualCompIter, NoLessOrEqualCompIter>);
static_assert(!HasSpaceship<cuda::std::reverse_iterator<NoLessOrEqualCompIter>>);
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

int main(int, char**)
{
  return 0;
}
