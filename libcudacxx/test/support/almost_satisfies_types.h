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
  TEST_FUNC T begin();
  TEST_FUNC U end();
};

static_assert(cuda::std::ranges::contiguous_range<UncheckedRange<int*, int*>>);

// almost an input_iterator
class InputIteratorNotDerivedFrom
{
public:
  using difference_type   = long;
  using value_type        = int;
  using iterator_category = void;

  TEST_FUNC InputIteratorNotDerivedFrom& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
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

  TEST_FUNC InputIteratorNotIndirectlyReadable& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
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

  TEST_FUNC int& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
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
  TEST_FUNC bool operator()(int) const;
};

static_assert(cuda::std::predicate<IndirectUnaryPredicateNotCopyConstructible, int&>);
static_assert(!cuda::std::indirect_unary_predicate<IndirectUnaryPredicateNotCopyConstructible, int*>);

class IndirectUnaryPredicateNotPredicate
{
public:
  TEST_FUNC bool operator()(int&&) const;
};

static_assert(!cuda::std::predicate<IndirectUnaryPredicateNotPredicate, int&>);
static_assert(!cuda::std::indirect_unary_predicate<IndirectUnaryPredicateNotPredicate, int*>);

// almost a sentinel_for cpp20_input_iterator
class SentinelForNotSemiregular
{
public:
  SentinelForNotSemiregular() = delete;
  using difference_type       = long;
  TEST_FUNC SentinelForNotSemiregular& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
  TEST_FUNC friend bool operator==(const SentinelForNotSemiregular&, const cpp20_input_iterator<int*>&);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator==(const cpp20_input_iterator<int*>&, const SentinelForNotSemiregular&);
  TEST_FUNC friend bool operator!=(const SentinelForNotSemiregular&, const cpp20_input_iterator<int*>&);
  TEST_FUNC friend bool operator!=(const cpp20_input_iterator<int*>&, const SentinelForNotSemiregular&);
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
  TEST_FUNC SentinelForNotWeaklyEqualityComparableWith& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
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
  TEST_FUNC WeaklyIncrementableNotMovable& operator++();
  TEST_FUNC void operator++(int);
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

  TEST_FUNC ForwardIteratorNotDerivedFrom& operator++();
  TEST_FUNC ForwardIteratorNotDerivedFrom operator++(int);
  TEST_FUNC const int& operator*() const;
#if TEST_STD_VER > 2017
  bool operator==(const ForwardIteratorNotDerivedFrom&) const = default;
#else
  TEST_FUNC bool operator==(const ForwardIteratorNotDerivedFrom&) const;
  TEST_FUNC bool operator!=(const ForwardIteratorNotDerivedFrom&) const;
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

  TEST_FUNC ForwardIteratorNotIncrementable& operator++();
  TEST_FUNC int operator++(int);
  TEST_FUNC const int& operator*() const;
#if TEST_STD_VER > 2017
  bool operator==(const ForwardIteratorNotIncrementable&) const = default;
#else
  TEST_FUNC bool operator==(const ForwardIteratorNotIncrementable&) const;
  TEST_FUNC bool operator!=(const ForwardIteratorNotIncrementable&) const;
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

  TEST_FUNC BidirectionalIteratorNotDerivedFrom& operator++();
  TEST_FUNC BidirectionalIteratorNotDerivedFrom operator++(int);
  TEST_FUNC BidirectionalIteratorNotDerivedFrom& operator--();
  TEST_FUNC BidirectionalIteratorNotDerivedFrom operator--(int);
  TEST_FUNC int& operator*() const;

#if TEST_STD_VER > 2017
  bool operator==(const BidirectionalIteratorNotDerivedFrom&) const = default;
#else
  TEST_FUNC bool operator==(const BidirectionalIteratorNotDerivedFrom&) const;
  TEST_FUNC bool operator!=(const BidirectionalIteratorNotDerivedFrom&) const;
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

  TEST_FUNC BidirectionalIteratorNotDecrementable& operator++();
  TEST_FUNC BidirectionalIteratorNotDecrementable operator++(int);
  TEST_FUNC int& operator*() const;

#if TEST_STD_VER > 2017
  bool operator==(const BidirectionalIteratorNotDecrementable&) const = default;
#else
  TEST_FUNC bool operator==(const BidirectionalIteratorNotDecrementable&) const;
  TEST_FUNC bool operator!=(const BidirectionalIteratorNotDecrementable&) const;
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

  TEST_FUNC PermutableNotForwardIterator& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC int& operator*() const;
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

  TEST_FUNC PermutableNotSwappable& operator++();
  TEST_FUNC PermutableNotSwappable operator++(int);
  TEST_FUNC NotSwappable& operator*() const;

#if TEST_STD_VER > 2017
  bool operator==(const PermutableNotSwappable&) const = default;
#else
  TEST_FUNC bool operator==(const PermutableNotSwappable&) const;
  TEST_FUNC bool operator!=(const PermutableNotSwappable&) const;
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

  TEST_FUNC int& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC int& operator*();
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

  TEST_FUNC OutputIteratorNotIndirectlyWritable& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
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

  TEST_FUNC int& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC const int& operator*() const;
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

  TEST_FUNC reference operator*() const;
  TEST_FUNC reference operator[](difference_type) const;

  TEST_FUNC Self& operator++();
  TEST_FUNC Self& operator--();
  TEST_FUNC Self operator++(int);
  TEST_FUNC Self operator--(int);

  TEST_FUNC Self& operator+=(difference_type);
  TEST_FUNC Self& operator-=(difference_type);
  TEST_FUNC friend Self operator+(Self, difference_type);
  TEST_FUNC friend Self operator+(difference_type, Self);
  TEST_FUNC friend Self operator-(Self, difference_type);
  TEST_FUNC friend difference_type operator-(Self, Self);

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Self&) const = default;
#else
  TEST_FUNC bool operator==(const Self&) const;
  TEST_FUNC bool operator!=(const Self&) const;

  TEST_FUNC bool operator<(const Self&) const;
  TEST_FUNC bool operator<=(const Self&) const;
  TEST_FUNC bool operator>(const Self&) const;
  TEST_FUNC bool operator>=(const Self&) const;
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

  TEST_FUNC reference operator*() const;
  // Deliberately returning a type different from `reference`.
  TEST_FUNC const int& operator[](difference_type) const;

  TEST_FUNC Self& operator++();
  TEST_FUNC Self& operator--();
  TEST_FUNC Self operator++(int);
  TEST_FUNC Self operator--(int);

  TEST_FUNC Self& operator+=(difference_type);
  TEST_FUNC Self& operator-=(difference_type);
  TEST_FUNC friend Self operator+(Self, difference_type);
  TEST_FUNC friend Self operator+(difference_type, Self);
  TEST_FUNC friend Self operator-(Self, difference_type);
  TEST_FUNC friend difference_type operator-(Self, Self);

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Self&) const = default;
#else
  TEST_FUNC bool operator==(const Self&) const;
  TEST_FUNC bool operator!=(const Self&) const;

  TEST_FUNC bool operator<(const Self&) const;
  TEST_FUNC bool operator<=(const Self&) const;
  TEST_FUNC bool operator>(const Self&) const;
  TEST_FUNC bool operator>=(const Self&) const;
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

  TEST_FUNC bool operator()(Iter&, Iter&) const;
};

#endif // ALMOST_SATISFIES_TYPES_H
