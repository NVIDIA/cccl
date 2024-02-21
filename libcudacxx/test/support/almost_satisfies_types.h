//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef ALMOST_SATISFIES_TYPES_H
#define ALMOST_SATISFIES_TYPES_H

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "test_iterators.h"

template <class T, class U = sentinel_wrapper<T>>
class UncheckedRange
{
public:
  __host__ __device__ T begin();
  __host__ __device__ U end();
};

static_assert(cuda::std::ranges::contiguous_range<UncheckedRange<int*, int*>>);

// almost an input_iterator
class InputIteratorNotDerivedFrom
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = void;

  __host__ __device__ InputIteratorNotDerivedFrom& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
};

using InputRangeNotDerivedFrom = UncheckedRange<InputIteratorNotDerivedFrom>;

static_assert(cuda::std::input_or_output_iterator<InputIteratorNotDerivedFrom>);
static_assert(cuda::std::indirectly_readable<InputIteratorNotDerivedFrom>);
static_assert(!cuda::std::input_iterator<InputIteratorNotDerivedFrom>);
static_assert(!cuda::std::ranges::input_range<InputRangeNotDerivedFrom>);

class InputIteratorNotIndirectlyReadable
{
public:
  using difference_type   = long;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ InputIteratorNotIndirectlyReadable& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
};

using InputRangeNotIndirectlyReadable = UncheckedRange<InputIteratorNotIndirectlyReadable>;

static_assert(cuda::std::input_or_output_iterator<InputIteratorNotIndirectlyReadable>);
static_assert(!cuda::std::indirectly_readable<InputIteratorNotIndirectlyReadable>);
static_assert(!cuda::std::input_iterator<InputIteratorNotIndirectlyReadable>);
static_assert(!cuda::std::ranges::input_range<InputIteratorNotIndirectlyReadable>);

class InputIteratorNotInputOrOutputIterator
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ int& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
};

using InputRangeNotInputOrOutputIterator = UncheckedRange<InputIteratorNotInputOrOutputIterator>;

static_assert(!cuda::std::input_or_output_iterator<InputIteratorNotInputOrOutputIterator>);
static_assert(cuda::std::indirectly_readable<InputIteratorNotInputOrOutputIterator>);
static_assert(!cuda::std::input_iterator<InputIteratorNotInputOrOutputIterator>);
static_assert(!cuda::std::ranges::input_range<InputRangeNotInputOrOutputIterator>);

// almost an indirect_unary_predicate
class IndirectUnaryPredicateNotCopyConstructible
{
public:
  IndirectUnaryPredicateNotCopyConstructible(const IndirectUnaryPredicateNotCopyConstructible&) = delete;
  __host__ __device__ bool operator()(int) const;
};

static_assert(cuda::std::predicate<IndirectUnaryPredicateNotCopyConstructible, int&>);
static_assert(!cuda::std::indirect_unary_predicate<IndirectUnaryPredicateNotCopyConstructible, int*>);

class IndirectUnaryPredicateNotPredicate
{
public:
  __host__ __device__ bool operator()(int&&) const;
};

static_assert(!cuda::std::predicate<IndirectUnaryPredicateNotPredicate, int&>);
static_assert(!cuda::std::indirect_unary_predicate<IndirectUnaryPredicateNotPredicate, int*>);

// almost a sentinel_for cpp20_input_iterator
class SentinelForNotSemiregular
{
public:
  SentinelForNotSemiregular() = delete;
  using difference_type       = long;
  __host__ __device__ SentinelForNotSemiregular& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
  __host__ __device__ friend bool operator==(const SentinelForNotSemiregular&, const cpp20_input_iterator<int*>&);
#if TEST_STD_VER < 2020
  __host__ __device__ friend bool operator==(const cpp20_input_iterator<int*>&, const SentinelForNotSemiregular&);
  __host__ __device__ friend bool operator!=(const SentinelForNotSemiregular&, const cpp20_input_iterator<int*>&);
  __host__ __device__ friend bool operator!=(const cpp20_input_iterator<int*>&, const SentinelForNotSemiregular&);
#endif
};

using InputRangeNotSentinelSemiregular  = UncheckedRange<cpp20_input_iterator<int*>, SentinelForNotSemiregular>;
using OutputRangeNotSentinelSemiregular = UncheckedRange<cpp20_output_iterator<int*>, SentinelForNotSemiregular>;

static_assert(cuda::std::input_or_output_iterator<SentinelForNotSemiregular>);
static_assert(!cuda::std::semiregular<SentinelForNotSemiregular>);
static_assert(!cuda::std::sentinel_for<SentinelForNotSemiregular, cpp20_input_iterator<int*>>);

// almost a sentinel_for cpp20_input_iterator
class SentinelForNotWeaklyEqualityComparableWith
{
public:
  using difference_type = long;
  __host__ __device__ SentinelForNotWeaklyEqualityComparableWith& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
};

using InputRangeNotSentinelEqualityComparableWith =
  UncheckedRange<cpp20_input_iterator<int*>, SentinelForNotWeaklyEqualityComparableWith>;
using OutputRangeNotSentinelEqualityComparableWith =
  UncheckedRange<cpp20_output_iterator<int*>, SentinelForNotWeaklyEqualityComparableWith>;

static_assert(cuda::std::input_or_output_iterator<SentinelForNotWeaklyEqualityComparableWith>);
static_assert(cuda::std::semiregular<SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!cuda::std::sentinel_for<SentinelForNotWeaklyEqualityComparableWith, cpp20_input_iterator<int*>>);

class WeaklyIncrementableNotMovable
{
public:
  using difference_type = long;
  __host__ __device__ WeaklyIncrementableNotMovable& operator++();
  __host__ __device__ void operator++(int);
  WeaklyIncrementableNotMovable(const WeaklyIncrementableNotMovable&) = delete;
};

static_assert(!cuda::std::movable<WeaklyIncrementableNotMovable>);
static_assert(!cuda::std::weakly_incrementable<WeaklyIncrementableNotMovable>);

// almost a forward_iterator
class ForwardIteratorNotDerivedFrom
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ ForwardIteratorNotDerivedFrom& operator++();
  __host__ __device__ ForwardIteratorNotDerivedFrom operator++(int);
  __host__ __device__ const int& operator*() const;
#if TEST_STD_VER > 2017
  bool operator==(const ForwardIteratorNotDerivedFrom&) const = default;
#else
  __host__ __device__ bool operator==(const ForwardIteratorNotDerivedFrom&) const;
  __host__ __device__ bool operator!=(const ForwardIteratorNotDerivedFrom&) const;
#endif
};

using ForwardRangeNotDerivedFrom = UncheckedRange<ForwardIteratorNotDerivedFrom>;

static_assert(cuda::std::input_iterator<ForwardIteratorNotDerivedFrom>);
static_assert(cuda::std::incrementable<ForwardIteratorNotDerivedFrom>);
static_assert(cuda::std::sentinel_for<ForwardIteratorNotDerivedFrom, ForwardIteratorNotDerivedFrom>);
static_assert(!cuda::std::forward_iterator<ForwardIteratorNotDerivedFrom>);

class ForwardIteratorNotIncrementable
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::forward_iterator_tag;

  __host__ __device__ ForwardIteratorNotIncrementable& operator++();
  __host__ __device__ int operator++(int);
  __host__ __device__ const int& operator*() const;
#if TEST_STD_VER > 2017
  bool operator==(const ForwardIteratorNotIncrementable&) const = default;
#else
  __host__ __device__ bool operator==(const ForwardIteratorNotIncrementable&) const;
  __host__ __device__ bool operator!=(const ForwardIteratorNotIncrementable&) const;
#endif
};

using ForwardRangeNotIncrementable = UncheckedRange<ForwardIteratorNotIncrementable>;

static_assert(cuda::std::input_iterator<ForwardIteratorNotIncrementable>);
static_assert(!cuda::std::incrementable<ForwardIteratorNotIncrementable>);
static_assert(cuda::std::sentinel_for<ForwardIteratorNotIncrementable, ForwardIteratorNotIncrementable>);
static_assert(!cuda::std::forward_iterator<ForwardIteratorNotIncrementable>);

using ForwardRangeNotSentinelSemiregular = UncheckedRange<forward_iterator<int*>, SentinelForNotSemiregular>;
using ForwardRangeNotSentinelEqualityComparableWith =
  UncheckedRange<forward_iterator<int*>, SentinelForNotWeaklyEqualityComparableWith>;

class BidirectionalIteratorNotDerivedFrom
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::forward_iterator_tag;

  __host__ __device__ BidirectionalIteratorNotDerivedFrom& operator++();
  __host__ __device__ BidirectionalIteratorNotDerivedFrom operator++(int);
  __host__ __device__ BidirectionalIteratorNotDerivedFrom& operator--();
  __host__ __device__ BidirectionalIteratorNotDerivedFrom operator--(int);
  __host__ __device__ int& operator*() const;

#if TEST_STD_VER > 2017
  bool operator==(const BidirectionalIteratorNotDerivedFrom&) const = default;
#else
  __host__ __device__ bool operator==(const BidirectionalIteratorNotDerivedFrom&) const;
  __host__ __device__ bool operator!=(const BidirectionalIteratorNotDerivedFrom&) const;
#endif
};

using BidirectionalRangeNotDerivedFrom = UncheckedRange<BidirectionalIteratorNotDerivedFrom>;
using BidirectionalRangeNotSentinelSemiregular =
  UncheckedRange<bidirectional_iterator<int*>, SentinelForNotSemiregular>;
using BidirectionalRangeNotSentinelWeaklyEqualityComparableWith =
  UncheckedRange<bidirectional_iterator<int*>, SentinelForNotWeaklyEqualityComparableWith>;

static_assert(cuda::std::forward_iterator<BidirectionalIteratorNotDerivedFrom>);
static_assert(!cuda::std::bidirectional_iterator<BidirectionalIteratorNotDerivedFrom>);
static_assert(!cuda::std::ranges::bidirectional_range<BidirectionalRangeNotDerivedFrom>);

class BidirectionalIteratorNotDecrementable
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ BidirectionalIteratorNotDecrementable& operator++();
  __host__ __device__ BidirectionalIteratorNotDecrementable operator++(int);
  __host__ __device__ int& operator*() const;

#if TEST_STD_VER > 2017
  bool operator==(const BidirectionalIteratorNotDecrementable&) const = default;
#else
  __host__ __device__ bool operator==(const BidirectionalIteratorNotDecrementable&) const;
  __host__ __device__ bool operator!=(const BidirectionalIteratorNotDecrementable&) const;
#endif
};

using BidirectionalRangeNotDecrementable = UncheckedRange<BidirectionalIteratorNotDecrementable>;

static_assert(cuda::std::forward_iterator<BidirectionalIteratorNotDecrementable>);
static_assert(!cuda::std::bidirectional_iterator<BidirectionalIteratorNotDecrementable>);
static_assert(!cuda::std::ranges::bidirectional_range<BidirectionalRangeNotDecrementable>);

class PermutableNotForwardIterator
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ PermutableNotForwardIterator& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ int& operator*() const;
};

using PermutableRangeNotForwardIterator = UncheckedRange<PermutableNotForwardIterator>;

static_assert(cuda::std::input_iterator<PermutableNotForwardIterator>);
static_assert(!cuda::std::forward_iterator<PermutableNotForwardIterator>);
static_assert(!cuda::std::permutable<PermutableNotForwardIterator>);

class PermutableNotSwappable
{
public:
  class NotSwappable
  {
    NotSwappable(NotSwappable&&) = delete;
  };

  using difference_type   = long;
  using value_type        = NotSwappable;
  using iterator_category = cuda::std::contiguous_iterator_tag;

  __host__ __device__ PermutableNotSwappable& operator++();
  __host__ __device__ PermutableNotSwappable operator++(int);
  __host__ __device__ NotSwappable& operator*() const;

#if TEST_STD_VER > 2017
  bool operator==(const PermutableNotSwappable&) const = default;
#else
  __host__ __device__ bool operator==(const PermutableNotSwappable&) const;
  __host__ __device__ bool operator!=(const PermutableNotSwappable&) const;
#endif
};

using PermutableRangeNotSwappable = UncheckedRange<PermutableNotSwappable>;

static_assert(cuda::std::input_iterator<PermutableNotSwappable>);
static_assert(cuda::std::forward_iterator<PermutableNotSwappable>);
static_assert(!cuda::std::permutable<PermutableNotSwappable>);
static_assert(!cuda::std::indirectly_swappable<PermutableNotSwappable>);

class OutputIteratorNotInputOrOutputIterator
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ int& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ int& operator*();
};

using OutputRangeNotInputOrOutputIterator = UncheckedRange<InputIteratorNotInputOrOutputIterator>;

static_assert(!cuda::std::input_or_output_iterator<OutputIteratorNotInputOrOutputIterator>);
static_assert(cuda::std::indirectly_writable<OutputIteratorNotInputOrOutputIterator, int>);
static_assert(!cuda::std::output_iterator<OutputIteratorNotInputOrOutputIterator, int>);
static_assert(!cuda::std::ranges::input_range<OutputRangeNotInputOrOutputIterator>);

class OutputIteratorNotIndirectlyWritable
{
public:
  using difference_type   = long;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ OutputIteratorNotIndirectlyWritable& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
};

using OutputRangeNotIndirectlyWritable = UncheckedRange<OutputIteratorNotIndirectlyWritable>;

static_assert(cuda::std::input_or_output_iterator<OutputIteratorNotIndirectlyWritable>);
static_assert(!cuda::std::indirectly_writable<OutputIteratorNotIndirectlyWritable, int>);
static_assert(!cuda::std::output_iterator<OutputIteratorNotIndirectlyWritable, int>);
static_assert(!cuda::std::ranges::output_range<OutputIteratorNotIndirectlyWritable, int>);

class IndirectBinaryPredicateNotIndirectlyReadable
{
public:
  using difference_type   = long;
  using iterator_category = cuda::std::input_iterator_tag;

  __host__ __device__ int& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ const int& operator*() const;
};

using InputRangeIndirectBinaryPredicateNotIndirectlyReadable =
  UncheckedRange<cpp20_input_iterator<int*>, IndirectBinaryPredicateNotIndirectlyReadable>;

static_assert(
  !cuda::std::indirect_binary_predicate<cuda::std::ranges::equal_to, IndirectBinaryPredicateNotIndirectlyReadable, int*>);

class RandomAccessIteratorNotDerivedFrom
{
  using Self = RandomAccessIteratorNotDerivedFrom;

public:
  using value_type      = int;
  using difference_type = long;
  using pointer         = int*;
  using reference       = int&;
  // Deliberately not using the `cuda::std::random_access_iterator_tag` category.
  using iterator_category = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ reference operator*() const;
  __host__ __device__ reference operator[](difference_type) const;

  __host__ __device__ Self& operator++();
  __host__ __device__ Self& operator--();
  __host__ __device__ Self operator++(int);
  __host__ __device__ Self operator--(int);

  __host__ __device__ Self& operator+=(difference_type);
  __host__ __device__ Self& operator-=(difference_type);
  __host__ __device__ friend Self operator+(Self, difference_type);
  __host__ __device__ friend Self operator+(difference_type, Self);
  __host__ __device__ friend Self operator-(Self, difference_type);
  __host__ __device__ friend difference_type operator-(Self, Self);

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const Self&) const = default;
#else
  __host__ __device__ bool operator==(const Self&) const;
  __host__ __device__ bool operator!=(const Self&) const;

  __host__ __device__ bool operator<(const Self&) const;
  __host__ __device__ bool operator<=(const Self&) const;
  __host__ __device__ bool operator>(const Self&) const;
  __host__ __device__ bool operator>=(const Self&) const;
#endif
};

static_assert(cuda::std::bidirectional_iterator<RandomAccessIteratorNotDerivedFrom>);
static_assert(!cuda::std::random_access_iterator<RandomAccessIteratorNotDerivedFrom>);

using RandomAccessRangeNotDerivedFrom = UncheckedRange<RandomAccessIteratorNotDerivedFrom>;

class RandomAccessIteratorBadIndex
{
  using Self = RandomAccessIteratorBadIndex;

public:
  using value_type        = int;
  using difference_type   = long;
  using pointer           = int*;
  using reference         = int&;
  using iterator_category = cuda::std::random_access_iterator_tag;

  __host__ __device__ reference operator*() const;
  // Deliberately returning a type different from `reference`.
  __host__ __device__ const int& operator[](difference_type) const;

  __host__ __device__ Self& operator++();
  __host__ __device__ Self& operator--();
  __host__ __device__ Self operator++(int);
  __host__ __device__ Self operator--(int);

  __host__ __device__ Self& operator+=(difference_type);
  __host__ __device__ Self& operator-=(difference_type);
  __host__ __device__ friend Self operator+(Self, difference_type);
  __host__ __device__ friend Self operator+(difference_type, Self);
  __host__ __device__ friend Self operator-(Self, difference_type);
  __host__ __device__ friend difference_type operator-(Self, Self);

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const Self&) const = default;
#else
  __host__ __device__ bool operator==(const Self&) const;
  __host__ __device__ bool operator!=(const Self&) const;

  __host__ __device__ bool operator<(const Self&) const;
  __host__ __device__ bool operator<=(const Self&) const;
  __host__ __device__ bool operator>(const Self&) const;
  __host__ __device__ bool operator>=(const Self&) const;
#endif
};

static_assert(cuda::std::bidirectional_iterator<RandomAccessIteratorBadIndex>);
static_assert(!cuda::std::random_access_iterator<RandomAccessIteratorBadIndex>);

using RandomAccessRangeBadIndex = UncheckedRange<RandomAccessIteratorBadIndex>;

template <class Iter>
class ComparatorNotCopyable
{
public:
  ComparatorNotCopyable(ComparatorNotCopyable&&)                 = default;
  ComparatorNotCopyable& operator=(ComparatorNotCopyable&&)      = default;
  ComparatorNotCopyable(const ComparatorNotCopyable&)            = delete;
  ComparatorNotCopyable& operator=(const ComparatorNotCopyable&) = delete;

  __host__ __device__ bool operator()(Iter&, Iter&) const;
};

#endif // ALMOST_SATISFIES_TYPES_H
