//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class T>
// struct iterator_traits;

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>

#include "iterator_traits_cpp17_iterators.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Traits, class = void>
inline constexpr bool has_iterator_concept_v = false;

template <class Traits>
inline constexpr bool has_iterator_concept_v<Traits, cuda::std::void_t<typename Traits::iterator_concept>> = true;

template <class It, class Traits, cuda::std::enable_if_t<cuda::std::is_pointer_v<It>, int> = 0>
TEST_FUNC constexpr void test_iter_concept()
{
  static_assert(cuda::std::same_as<typename Traits::iterator_concept, cuda::std::contiguous_iterator_tag>);
}

template <class It, class Traits, cuda::std::enable_if_t<!cuda::std::is_pointer_v<It>, int> = 0>
TEST_FUNC constexpr void test_iter_concept()
{
  static_assert(!has_iterator_concept_v<Traits>);
}

template <class Iter, class Category, class ValueType, class DiffType, class RefType, class PtrType>
TEST_FUNC constexpr bool test()
{
  using Traits = cuda::std::iterator_traits<Iter>;
  static_assert(cuda::std::same_as<typename Traits::iterator_category, Category>);
  static_assert(cuda::std::same_as<typename Traits::value_type, ValueType>);
  static_assert(cuda::std::same_as<typename Traits::difference_type, DiffType>);
  static_assert(cuda::std::same_as<typename Traits::reference, RefType>);
  static_assert(cuda::std::same_as<typename Traits::pointer, PtrType>);

  test_iter_concept<Iter, Traits>();

  return true;
}

template <class Iter, class Category>
TEST_FUNC constexpr bool testIOIterator()
{
  return test<Iter, Category, void, cuda::std::ptrdiff_t, void, void>();
}

template <class Iter, class Category, class ValueType>
TEST_FUNC constexpr bool testConst()
{
  return test<Iter, Category, ValueType, cuda::std::ptrdiff_t, const ValueType&, const ValueType*>();
}

template <class Iter, class Category, class ValueType>
TEST_FUNC constexpr bool testMutable()
{
  return test<Iter, Category, ValueType, cuda::std::ptrdiff_t, ValueType&, ValueType*>();
}

// Standard types.

// The Standard does not specify whether iterator_traits<It>::iterator_concept
// exists for any particular non-pointer type, we assume it is present
// only for pointers.
//
static_assert(testMutable<cuda::std::array<int, 10>::iterator, cuda::std::random_access_iterator_tag, int>());
static_assert(testConst<cuda::std::array<int, 10>::const_iterator, cuda::std::random_access_iterator_tag, int>());

// Local test iterators.

struct AllMembers
{
  struct iterator_category
  {};
  struct value_type
  {};
  struct difference_type
  {};
  struct reference
  {};
  struct pointer
  {};
};
using AllMembersTraits = cuda::std::iterator_traits<AllMembers>;
static_assert(cuda::std::same_as<AllMembersTraits::iterator_category, AllMembers::iterator_category>);
static_assert(cuda::std::same_as<AllMembersTraits::value_type, AllMembers::value_type>);
static_assert(cuda::std::same_as<AllMembersTraits::difference_type, AllMembers::difference_type>);
static_assert(cuda::std::same_as<AllMembersTraits::reference, AllMembers::reference>);
static_assert(cuda::std::same_as<AllMembersTraits::pointer, AllMembers::pointer>);
static_assert(!has_iterator_concept_v<AllMembersTraits>);

struct NoPointerMember
{
  struct iterator_category
  {};
  struct value_type
  {};
  struct difference_type
  {};
  struct reference
  {};
  // ignored, because NoPointerMember is not a LegacyInputIterator:
  TEST_FUNC value_type* operator->() const;
};
using NoPointerMemberTraits = cuda::std::iterator_traits<NoPointerMember>;
static_assert(cuda::std::same_as<NoPointerMemberTraits::iterator_category, NoPointerMember::iterator_category>);
static_assert(cuda::std::same_as<NoPointerMemberTraits::value_type, NoPointerMember::value_type>);
static_assert(cuda::std::same_as<NoPointerMemberTraits::difference_type, NoPointerMember::difference_type>);
static_assert(cuda::std::same_as<NoPointerMemberTraits::reference, NoPointerMember::reference>);
static_assert(cuda::std::same_as<NoPointerMemberTraits::pointer, void>);
static_assert(!has_iterator_concept_v<NoPointerMemberTraits>);

struct IterConcept
{
  struct iterator_category
  {};
  struct value_type
  {};
  struct difference_type
  {};
  struct reference
  {};
  struct pointer
  {};
  // iterator_traits does NOT pass through the iterator_concept of the type itself.
  struct iterator_concept
  {};
};
using IterConceptTraits = cuda::std::iterator_traits<IterConcept>;
static_assert(cuda::std::same_as<IterConceptTraits::iterator_category, IterConcept::iterator_category>);
static_assert(cuda::std::same_as<IterConceptTraits::value_type, IterConcept::value_type>);
static_assert(cuda::std::same_as<IterConceptTraits::difference_type, IterConcept::difference_type>);
static_assert(cuda::std::same_as<IterConceptTraits::reference, IterConcept::reference>);
static_assert(cuda::std::same_as<IterConceptTraits::pointer, IterConcept::pointer>);
static_assert(!has_iterator_concept_v<IterConceptTraits>);

struct LegacyInput
{
  struct iterator_category
  {};
  struct value_type
  {};
  struct reference
  {
    TEST_FUNC operator value_type() const;
  };

  TEST_FUNC friend bool operator==(LegacyInput, LegacyInput);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyInput, LegacyInput);
#endif
  TEST_FUNC reference operator*() const;
  TEST_FUNC LegacyInput& operator++();
  TEST_FUNC LegacyInput operator++(int);
};
template <>
struct cuda::std::incrementable_traits<LegacyInput>
{
  using difference_type = short;
};
using LegacyInputTraits = cuda::std::iterator_traits<LegacyInput>;
static_assert(cuda::std::same_as<LegacyInputTraits::iterator_category, LegacyInput::iterator_category>);
static_assert(cuda::std::same_as<LegacyInputTraits::value_type, LegacyInput::value_type>);
static_assert(cuda::std::same_as<LegacyInputTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyInputTraits::reference, LegacyInput::reference>);
static_assert(cuda::std::same_as<LegacyInputTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyInputTraits>);

struct LegacyInputArrow
{
  struct iterator_category
  {};
  struct value_type
  {};
  struct reference
  {
    TEST_FUNC operator value_type() const;
  };

  TEST_FUNC friend bool operator==(LegacyInputArrow, LegacyInputArrow);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyInputArrow, LegacyInputArrow);
#endif
  // Otherwise, if decltype(​declval<I&>().operator->()) is well-formed, then pointer names that type.
  TEST_FUNC int* operator->();
  TEST_FUNC reference operator*() const;
  TEST_FUNC LegacyInputArrow& operator++();
  TEST_FUNC LegacyInputArrow operator++(int);
};
template <>
struct cuda::std::incrementable_traits<LegacyInputArrow>
{
  using difference_type = short;
};
using LegacyInputArrowTraits = cuda::std::iterator_traits<LegacyInputArrow>;
static_assert(cuda::std::same_as<LegacyInputArrowTraits::iterator_category, LegacyInputArrow::iterator_category>);
static_assert(cuda::std::same_as<LegacyInputArrowTraits::value_type, LegacyInputArrow::value_type>);
static_assert(cuda::std::same_as<LegacyInputArrowTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyInputArrowTraits::reference, LegacyInputArrow::reference>);
static_assert(cuda::std::same_as<LegacyInputArrowTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<LegacyInputArrowTraits>);

struct LegacyInputPointer
{
  struct iterator_category
  {};
  struct value_type
  {};
  struct reference
  {
    TEST_FUNC operator value_type() const;
  };
  // If the qualified-id I​::​pointer is valid and denotes a type, then pointer names that type.
  struct pointer
  {};

  TEST_FUNC friend bool operator==(LegacyInputPointer, LegacyInputPointer);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyInputPointer, LegacyInputPointer);
#endif
  // Otherwise, if decltype(​declval<I&>().operator->()) is well-formed, then pointer names that type.
  TEST_FUNC int* operator->();
  TEST_FUNC reference operator*() const;
  TEST_FUNC LegacyInputPointer& operator++();
  TEST_FUNC LegacyInputPointer operator++(int);
};
template <>
struct cuda::std::incrementable_traits<LegacyInputPointer>
{
  using difference_type = short;
};
using LegacyInputPointerTraits = cuda::std::iterator_traits<LegacyInputPointer>;
static_assert(cuda::std::same_as<LegacyInputPointerTraits::iterator_category, LegacyInputPointer::iterator_category>);
static_assert(cuda::std::same_as<LegacyInputPointerTraits::value_type, LegacyInputPointer::value_type>);
static_assert(cuda::std::same_as<LegacyInputPointerTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyInputPointerTraits::reference, LegacyInputPointer::reference>);
static_assert(cuda::std::same_as<LegacyInputPointerTraits::pointer, LegacyInputPointer::pointer>);
static_assert(!has_iterator_concept_v<LegacyInputPointerTraits>);

struct LegacyInputNoValueType
{
  struct not_value_type
  {};
  using difference_type = int; // or any signed integral type
  struct reference
  {
    TEST_FUNC operator not_value_type&() const;
  };

  TEST_FUNC friend bool operator==(LegacyInputNoValueType, LegacyInputNoValueType);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyInputNoValueType, LegacyInputNoValueType);
#endif
  TEST_FUNC reference operator*() const;
  TEST_FUNC LegacyInputNoValueType& operator++();
  TEST_FUNC LegacyInputNoValueType operator++(int);
};
template <>
struct cuda::std::indirectly_readable_traits<LegacyInputNoValueType>
{
  using value_type = LegacyInputNoValueType::not_value_type;
};
using LegacyInputNoValueTypeTraits = cuda::std::iterator_traits<LegacyInputNoValueType>;
static_assert(cuda::std::same_as<LegacyInputNoValueTypeTraits::iterator_category, cuda::std::input_iterator_tag>);
static_assert(cuda::std::same_as<LegacyInputNoValueTypeTraits::value_type, LegacyInputNoValueType::not_value_type>);
static_assert(cuda::std::same_as<LegacyInputNoValueTypeTraits::difference_type, int>);
static_assert(cuda::std::same_as<LegacyInputNoValueTypeTraits::reference, LegacyInputNoValueType::reference>);
static_assert(cuda::std::same_as<LegacyInputNoValueTypeTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyInputNoValueTypeTraits>);

struct LegacyForward
{
  struct not_value_type
  {};

  TEST_FUNC friend bool operator==(LegacyForward, LegacyForward);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyForward, LegacyForward);
#endif
  TEST_FUNC const not_value_type& operator*() const;
  TEST_FUNC LegacyForward& operator++();
  TEST_FUNC LegacyForward operator++(int);
};
template <>
struct cuda::std::indirectly_readable_traits<LegacyForward>
{
  using value_type = LegacyForward::not_value_type;
};
template <>
struct cuda::std::incrementable_traits<LegacyForward>
{
  using difference_type = short; // or any signed integral type
};
using LegacyForwardTraits = cuda::std::iterator_traits<LegacyForward>;
static_assert(cuda::std::same_as<LegacyForwardTraits::iterator_category, cuda::std::forward_iterator_tag>);
static_assert(cuda::std::same_as<LegacyForwardTraits::value_type, LegacyForward::not_value_type>);
static_assert(cuda::std::same_as<LegacyForwardTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyForwardTraits::reference, const LegacyForward::not_value_type&>);
static_assert(cuda::std::same_as<LegacyForwardTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyForwardTraits>);

struct LegacyBidirectional
{
  struct value_type
  {};

  TEST_FUNC friend bool operator==(LegacyBidirectional, LegacyBidirectional);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyBidirectional, LegacyBidirectional);
#endif
  TEST_FUNC const value_type& operator*() const;
  TEST_FUNC LegacyBidirectional& operator++();
  TEST_FUNC LegacyBidirectional operator++(int);
  TEST_FUNC LegacyBidirectional& operator--();
  TEST_FUNC LegacyBidirectional operator--(int);
  TEST_FUNC friend short operator-(LegacyBidirectional, LegacyBidirectional);
};
using LegacyBidirectionalTraits = cuda::std::iterator_traits<LegacyBidirectional>;
static_assert(cuda::std::same_as<LegacyBidirectionalTraits::iterator_category, cuda::std::bidirectional_iterator_tag>,
              "");
static_assert(cuda::std::same_as<LegacyBidirectionalTraits::value_type, LegacyBidirectional::value_type>);
static_assert(cuda::std::same_as<LegacyBidirectionalTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyBidirectionalTraits::reference, const LegacyBidirectional::value_type&>);
static_assert(cuda::std::same_as<LegacyBidirectionalTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyBidirectionalTraits>);

// Almost a random access iterator except it is missing operator-(It, It).
struct MinusNotDeclaredIter
{
  struct value_type
  {};

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  auto operator<=>(const MinusNotDeclaredIter&) const = default; // nvbug3908399
#else
  TEST_FUNC friend bool operator==(const MinusNotDeclaredIter&, const MinusNotDeclaredIter&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(const MinusNotDeclaredIter&, const MinusNotDeclaredIter&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<(const MinusNotDeclaredIter&, const MinusNotDeclaredIter&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<=(const MinusNotDeclaredIter&, const MinusNotDeclaredIter&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>(const MinusNotDeclaredIter&, const MinusNotDeclaredIter&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>=(const MinusNotDeclaredIter&, const MinusNotDeclaredIter&) noexcept
  {
    return true;
  }
#endif

  TEST_FUNC const value_type& operator*() const;
  TEST_FUNC const value_type& operator[](long) const;
  TEST_FUNC MinusNotDeclaredIter& operator++();
  TEST_FUNC MinusNotDeclaredIter operator++(int);
  TEST_FUNC MinusNotDeclaredIter& operator--();
  TEST_FUNC MinusNotDeclaredIter operator--(int);
  TEST_FUNC MinusNotDeclaredIter& operator+=(long);
  TEST_FUNC MinusNotDeclaredIter& operator-=(long);

  // Providing difference_type does not fully compensate for missing operator-(It, It).
  TEST_FUNC friend MinusNotDeclaredIter operator-(MinusNotDeclaredIter, int);
  TEST_FUNC friend MinusNotDeclaredIter operator+(MinusNotDeclaredIter, int);
  TEST_FUNC friend MinusNotDeclaredIter operator+(int, MinusNotDeclaredIter);
};
template <>
struct cuda::std::incrementable_traits<MinusNotDeclaredIter>
{
  using difference_type = short;
};
using MinusNotDeclaredIterTraits = cuda::std::iterator_traits<MinusNotDeclaredIter>;
static_assert(cuda::std::same_as<MinusNotDeclaredIterTraits::iterator_category, cuda::std::bidirectional_iterator_tag>,
              "");
static_assert(cuda::std::same_as<MinusNotDeclaredIterTraits::value_type, MinusNotDeclaredIter::value_type>);
static_assert(cuda::std::same_as<MinusNotDeclaredIterTraits::difference_type, short>);
static_assert(cuda::std::same_as<MinusNotDeclaredIterTraits::reference, const MinusNotDeclaredIter::value_type&>);
static_assert(cuda::std::same_as<MinusNotDeclaredIterTraits::pointer, void>);
static_assert(!has_iterator_concept_v<MinusNotDeclaredIterTraits>);

struct WrongSubscriptReturnType
{
  struct value_type
  {};

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  auto operator<=>(const WrongSubscriptReturnType&) const = default; // nvbug3908399
#else
  TEST_FUNC friend bool operator==(const WrongSubscriptReturnType&, const WrongSubscriptReturnType&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(const WrongSubscriptReturnType&, const WrongSubscriptReturnType&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<(const WrongSubscriptReturnType&, const WrongSubscriptReturnType&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<=(const WrongSubscriptReturnType&, const WrongSubscriptReturnType&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>(const WrongSubscriptReturnType&, const WrongSubscriptReturnType&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>=(const WrongSubscriptReturnType&, const WrongSubscriptReturnType&) noexcept
  {
    return true;
  }
#endif

  // The type of it[n] is not convertible to the type of *it; therefore, this is not random-access.
  TEST_FUNC value_type& operator*() const;
  TEST_FUNC const value_type& operator[](long) const;
  TEST_FUNC WrongSubscriptReturnType& operator++();
  TEST_FUNC WrongSubscriptReturnType operator++(int);
  TEST_FUNC WrongSubscriptReturnType& operator--();
  TEST_FUNC WrongSubscriptReturnType operator--(int);
  TEST_FUNC WrongSubscriptReturnType& operator+=(long);
  TEST_FUNC WrongSubscriptReturnType& operator-=(long);
  TEST_FUNC friend short operator-(WrongSubscriptReturnType, WrongSubscriptReturnType);
  TEST_FUNC friend WrongSubscriptReturnType operator-(WrongSubscriptReturnType, int);
  TEST_FUNC friend WrongSubscriptReturnType operator+(WrongSubscriptReturnType, int);
  TEST_FUNC friend WrongSubscriptReturnType operator+(int, WrongSubscriptReturnType);
};
using WrongSubscriptReturnTypeTraits = cuda::std::iterator_traits<WrongSubscriptReturnType>;
static_assert(
  cuda::std::same_as<WrongSubscriptReturnTypeTraits::iterator_category, cuda::std::bidirectional_iterator_tag>, "");
static_assert(cuda::std::same_as<WrongSubscriptReturnTypeTraits::value_type, WrongSubscriptReturnType::value_type>);
static_assert(cuda::std::same_as<WrongSubscriptReturnTypeTraits::difference_type, short>);
static_assert(cuda::std::same_as<WrongSubscriptReturnTypeTraits::reference, WrongSubscriptReturnType::value_type&>);
static_assert(cuda::std::same_as<WrongSubscriptReturnTypeTraits::pointer, void>);
static_assert(!has_iterator_concept_v<WrongSubscriptReturnTypeTraits>);

struct LegacyRandomAccess
{
  struct value_type
  {};

  TEST_FUNC friend bool operator==(LegacyRandomAccess, LegacyRandomAccess);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyRandomAccess, LegacyRandomAccess);
#endif
  TEST_FUNC friend bool operator<(LegacyRandomAccess, LegacyRandomAccess);
  TEST_FUNC friend bool operator<=(LegacyRandomAccess, LegacyRandomAccess);
  TEST_FUNC friend bool operator>(LegacyRandomAccess, LegacyRandomAccess);
  TEST_FUNC friend bool operator>=(LegacyRandomAccess, LegacyRandomAccess);
  TEST_FUNC const value_type& operator*() const;
  TEST_FUNC const value_type& operator[](long) const;
  TEST_FUNC LegacyRandomAccess& operator++();
  TEST_FUNC LegacyRandomAccess operator++(int);
  TEST_FUNC LegacyRandomAccess& operator--();
  TEST_FUNC LegacyRandomAccess operator--(int);
  TEST_FUNC LegacyRandomAccess& operator+=(long);
  TEST_FUNC LegacyRandomAccess& operator-=(long);
  TEST_FUNC friend short operator-(LegacyRandomAccess, LegacyRandomAccess);
  TEST_FUNC friend LegacyRandomAccess operator-(LegacyRandomAccess, int);
  TEST_FUNC friend LegacyRandomAccess operator+(LegacyRandomAccess, int);
  TEST_FUNC friend LegacyRandomAccess operator+(int, LegacyRandomAccess);
};
using LegacyRandomAccessTraits = cuda::std::iterator_traits<LegacyRandomAccess>;
static_assert(cuda::std::same_as<LegacyRandomAccessTraits::iterator_category, cuda::std::random_access_iterator_tag>,
              "");
static_assert(cuda::std::same_as<LegacyRandomAccessTraits::value_type, LegacyRandomAccess::value_type>);
static_assert(cuda::std::same_as<LegacyRandomAccessTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyRandomAccessTraits::reference, const LegacyRandomAccess::value_type&>);
static_assert(cuda::std::same_as<LegacyRandomAccessTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyRandomAccessTraits>);

struct LegacyRandomAccessSpaceship
{
  struct not_value_type
  {};
  struct ReferenceConvertible
  {
    TEST_FUNC operator not_value_type&() const;
  };

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  auto operator<=>(const LegacyRandomAccessSpaceship&) const = default; // nvbug3908399
#else
  TEST_FUNC friend bool operator==(const LegacyRandomAccessSpaceship&, const LegacyRandomAccessSpaceship&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(const LegacyRandomAccessSpaceship&, const LegacyRandomAccessSpaceship&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<(const LegacyRandomAccessSpaceship&, const LegacyRandomAccessSpaceship&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<=(const LegacyRandomAccessSpaceship&, const LegacyRandomAccessSpaceship&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>(const LegacyRandomAccessSpaceship&, const LegacyRandomAccessSpaceship&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>=(const LegacyRandomAccessSpaceship&, const LegacyRandomAccessSpaceship&) noexcept
  {
    return true;
  }
#endif

  TEST_FUNC not_value_type& operator*() const;
  TEST_FUNC ReferenceConvertible operator[](long) const;
  TEST_FUNC LegacyRandomAccessSpaceship& operator++();
  TEST_FUNC LegacyRandomAccessSpaceship operator++(int);
  TEST_FUNC LegacyRandomAccessSpaceship& operator--();
  TEST_FUNC LegacyRandomAccessSpaceship operator--(int);
  TEST_FUNC LegacyRandomAccessSpaceship& operator+=(long);
  TEST_FUNC LegacyRandomAccessSpaceship& operator-=(long);
  TEST_FUNC friend short operator-(LegacyRandomAccessSpaceship, LegacyRandomAccessSpaceship);
  TEST_FUNC friend LegacyRandomAccessSpaceship operator-(LegacyRandomAccessSpaceship, int);
  TEST_FUNC friend LegacyRandomAccessSpaceship operator+(LegacyRandomAccessSpaceship, int);
  TEST_FUNC friend LegacyRandomAccessSpaceship operator+(int, LegacyRandomAccessSpaceship);
};
template <>
struct cuda::std::indirectly_readable_traits<LegacyRandomAccessSpaceship>
{
  using value_type = LegacyRandomAccessSpaceship::not_value_type;
};
template <>
struct cuda::std::incrementable_traits<LegacyRandomAccessSpaceship>
{
  using difference_type = short; // or any signed integral type
};
using LegacyRandomAccessSpaceshipTraits = cuda::std::iterator_traits<LegacyRandomAccessSpaceship>;
static_assert(
  cuda::std::same_as<LegacyRandomAccessSpaceshipTraits::iterator_category, cuda::std::random_access_iterator_tag>, "");
static_assert(
  cuda::std::same_as<LegacyRandomAccessSpaceshipTraits::value_type, LegacyRandomAccessSpaceship::not_value_type>, "");
static_assert(cuda::std::same_as<LegacyRandomAccessSpaceshipTraits::difference_type, short>);
static_assert(
  cuda::std::same_as<LegacyRandomAccessSpaceshipTraits::reference, LegacyRandomAccessSpaceship::not_value_type&>, "");
static_assert(cuda::std::same_as<LegacyRandomAccessSpaceshipTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyRandomAccessSpaceshipTraits>);

// For output iterators, value_type, difference_type, and reference may be void.
struct BareLegacyOutput
{
  struct Empty
  {};
  TEST_FUNC Empty operator*() const;
  TEST_FUNC BareLegacyOutput& operator++();
  TEST_FUNC BareLegacyOutput operator++(int);
};
using BareLegacyOutputTraits = cuda::std::iterator_traits<BareLegacyOutput>;
static_assert(cuda::std::same_as<BareLegacyOutputTraits::iterator_category, cuda::std::output_iterator_tag>);
static_assert(cuda::std::same_as<BareLegacyOutputTraits::value_type, void>);
static_assert(cuda::std::same_as<BareLegacyOutputTraits::difference_type, void>);
static_assert(cuda::std::same_as<BareLegacyOutputTraits::reference, void>);
static_assert(cuda::std::same_as<BareLegacyOutputTraits::pointer, void>);
static_assert(!has_iterator_concept_v<BareLegacyOutputTraits>);

// The operator- means we get difference_type.
struct LegacyOutputWithMinus
{
  struct Empty
  {};
  TEST_FUNC Empty operator*() const;
  TEST_FUNC LegacyOutputWithMinus& operator++();
  TEST_FUNC LegacyOutputWithMinus operator++(int);
  TEST_FUNC friend short operator-(LegacyOutputWithMinus, LegacyOutputWithMinus);
  // Lacking operator==, this is a LegacyIterator but not a LegacyInputIterator.
};
using LegacyOutputWithMinusTraits = cuda::std::iterator_traits<LegacyOutputWithMinus>;
static_assert(cuda::std::same_as<LegacyOutputWithMinusTraits::iterator_category, cuda::std::output_iterator_tag>);
static_assert(cuda::std::same_as<LegacyOutputWithMinusTraits::value_type, void>);
static_assert(cuda::std::same_as<LegacyOutputWithMinusTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyOutputWithMinusTraits::reference, void>);
static_assert(cuda::std::same_as<LegacyOutputWithMinusTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyOutputWithMinusTraits>);

struct LegacyOutputWithMemberTypes
{
  struct value_type
  {}; // ignored
  struct reference
  {}; // ignored
  using difference_type = long;

  TEST_FUNC friend bool operator==(LegacyOutputWithMemberTypes, LegacyOutputWithMemberTypes);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(LegacyOutputWithMemberTypes, LegacyOutputWithMemberTypes);
#endif
  TEST_FUNC reference operator*() const;
  TEST_FUNC LegacyOutputWithMemberTypes& operator++();
  TEST_FUNC LegacyOutputWithMemberTypes operator++(int);
  TEST_FUNC friend short operator-(LegacyOutputWithMemberTypes, LegacyOutputWithMemberTypes); // ignored
  // Since (*it) is not convertible to value_type, this is not a LegacyInputIterator.
};
using LegacyOutputWithMemberTypesTraits = cuda::std::iterator_traits<LegacyOutputWithMemberTypes>;
static_assert(cuda::std::same_as<LegacyOutputWithMemberTypesTraits::iterator_category, cuda::std::output_iterator_tag>,
              "");
static_assert(cuda::std::same_as<LegacyOutputWithMemberTypesTraits::value_type, void>);
static_assert(cuda::std::same_as<LegacyOutputWithMemberTypesTraits::difference_type, long>);
static_assert(cuda::std::same_as<LegacyOutputWithMemberTypesTraits::reference, void>);
static_assert(cuda::std::same_as<LegacyOutputWithMemberTypesTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyOutputWithMemberTypesTraits>);

struct LegacyRandomAccessSpecialized
{
  struct not_value_type
  {};

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  auto operator<=>(const LegacyRandomAccessSpecialized&) const = default; // nvbug3908399
#else
  TEST_FUNC friend bool operator==(const LegacyRandomAccessSpecialized&, const LegacyRandomAccessSpecialized&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(const LegacyRandomAccessSpecialized&, const LegacyRandomAccessSpecialized&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<(const LegacyRandomAccessSpecialized&, const LegacyRandomAccessSpecialized&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator<=(const LegacyRandomAccessSpecialized&, const LegacyRandomAccessSpecialized&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>(const LegacyRandomAccessSpecialized&, const LegacyRandomAccessSpecialized&) noexcept
  {
    return true;
  }
  TEST_FUNC friend bool operator>=(const LegacyRandomAccessSpecialized&, const LegacyRandomAccessSpecialized&) noexcept
  {
    return true;
  }
#endif

  TEST_FUNC not_value_type& operator*() const;
  TEST_FUNC not_value_type& operator[](long) const;
  TEST_FUNC LegacyRandomAccessSpecialized& operator++();
  TEST_FUNC LegacyRandomAccessSpecialized operator++(int);
  TEST_FUNC LegacyRandomAccessSpecialized& operator--();
  TEST_FUNC LegacyRandomAccessSpecialized operator--(int);
  TEST_FUNC LegacyRandomAccessSpecialized& operator+=(long);
  TEST_FUNC LegacyRandomAccessSpecialized& operator-=(long);
  TEST_FUNC friend long operator-(LegacyRandomAccessSpecialized, LegacyRandomAccessSpecialized);
  TEST_FUNC friend LegacyRandomAccessSpecialized operator-(LegacyRandomAccessSpecialized, int);
  TEST_FUNC friend LegacyRandomAccessSpecialized operator+(LegacyRandomAccessSpecialized, int);
  TEST_FUNC friend LegacyRandomAccessSpecialized operator+(int, LegacyRandomAccessSpecialized);
};
template <>
struct cuda::std::iterator_traits<LegacyRandomAccessSpecialized>
{
  using iterator_category = cuda::std::output_iterator_tag;
  using value_type        = short;
  using difference_type   = short;
  using reference         = short&;
  using pointer           = short*;
};
using LegacyRandomAccessSpecializedTraits = cuda::std::iterator_traits<LegacyRandomAccessSpecialized>;
static_assert(
  cuda::std::same_as<LegacyRandomAccessSpecializedTraits::iterator_category, cuda::std::output_iterator_tag>, "");
static_assert(cuda::std::same_as<LegacyRandomAccessSpecializedTraits::value_type, short>);
static_assert(cuda::std::same_as<LegacyRandomAccessSpecializedTraits::difference_type, short>);
static_assert(cuda::std::same_as<LegacyRandomAccessSpecializedTraits::reference, short&>);
static_assert(cuda::std::same_as<LegacyRandomAccessSpecializedTraits::pointer, short*>);
static_assert(!has_iterator_concept_v<LegacyRandomAccessSpecializedTraits>);

// Other test iterators.

using InputTestIteratorTraits = cuda::std::iterator_traits<cpp17_input_iterator<int*>>;
static_assert(cuda::std::same_as<InputTestIteratorTraits::iterator_category, cuda::std::input_iterator_tag>);
static_assert(cuda::std::same_as<InputTestIteratorTraits::value_type, int>);
static_assert(cuda::std::same_as<InputTestIteratorTraits::difference_type, cuda::std::ptrdiff_t>);
static_assert(cuda::std::same_as<InputTestIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<InputTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<InputTestIteratorTraits>);

using OutputTestIteratorTraits = cuda::std::iterator_traits<cpp17_output_iterator<int*>>;
static_assert(cuda::std::same_as<OutputTestIteratorTraits::iterator_category, cuda::std::output_iterator_tag>);
static_assert(cuda::std::same_as<OutputTestIteratorTraits::value_type, void>);
static_assert(cuda::std::same_as<OutputTestIteratorTraits::difference_type, cuda::std::ptrdiff_t>);
static_assert(cuda::std::same_as<OutputTestIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<OutputTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<OutputTestIteratorTraits>);

using ForwardTestIteratorTraits = cuda::std::iterator_traits<forward_iterator<int*>>;
static_assert(cuda::std::same_as<ForwardTestIteratorTraits::iterator_category, cuda::std::forward_iterator_tag>);
static_assert(cuda::std::same_as<ForwardTestIteratorTraits::value_type, int>);
static_assert(cuda::std::same_as<ForwardTestIteratorTraits::difference_type, cuda::std::ptrdiff_t>);
static_assert(cuda::std::same_as<ForwardTestIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<ForwardTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<ForwardTestIteratorTraits>);

using BidirectionalTestIteratorTraits = cuda::std::iterator_traits<bidirectional_iterator<int*>>;
static_assert(
  cuda::std::same_as<BidirectionalTestIteratorTraits::iterator_category, cuda::std::bidirectional_iterator_tag>, "");
static_assert(cuda::std::same_as<BidirectionalTestIteratorTraits::value_type, int>);
static_assert(cuda::std::same_as<BidirectionalTestIteratorTraits::difference_type, cuda::std::ptrdiff_t>);
static_assert(cuda::std::same_as<BidirectionalTestIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<BidirectionalTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<BidirectionalTestIteratorTraits>);

using RandomAccessTestIteratorTraits = cuda::std::iterator_traits<random_access_iterator<int*>>;
static_assert(
  cuda::std::same_as<RandomAccessTestIteratorTraits::iterator_category, cuda::std::random_access_iterator_tag>, "");
static_assert(cuda::std::same_as<RandomAccessTestIteratorTraits::value_type, int>);
static_assert(cuda::std::same_as<RandomAccessTestIteratorTraits::difference_type, cuda::std::ptrdiff_t>);
static_assert(cuda::std::same_as<RandomAccessTestIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<RandomAccessTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<RandomAccessTestIteratorTraits>);

using ContiguousTestIteratorTraits = cuda::std::iterator_traits<contiguous_iterator<int*>>;
static_assert(cuda::std::same_as<ContiguousTestIteratorTraits::iterator_category, cuda::std::contiguous_iterator_tag>,
              "");
static_assert(cuda::std::same_as<ContiguousTestIteratorTraits::value_type, int>);
static_assert(cuda::std::same_as<ContiguousTestIteratorTraits::difference_type, cuda::std::ptrdiff_t>);
static_assert(cuda::std::same_as<ContiguousTestIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<ContiguousTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<ContiguousTestIteratorTraits>);

using Cpp17BasicIteratorTraits = cuda::std::iterator_traits<iterator_traits_cpp17_iterator>;
static_assert(cuda::std::same_as<Cpp17BasicIteratorTraits::iterator_category, cuda::std::output_iterator_tag>);
static_assert(cuda::std::same_as<Cpp17BasicIteratorTraits::value_type, void>);
static_assert(cuda::std::same_as<Cpp17BasicIteratorTraits::difference_type, void>);
static_assert(cuda::std::same_as<Cpp17BasicIteratorTraits::reference, void>);
static_assert(cuda::std::same_as<Cpp17BasicIteratorTraits::pointer, void>);
static_assert(!has_iterator_concept_v<Cpp17BasicIteratorTraits>);

using Cpp17InputIteratorTraits = cuda::std::iterator_traits<iterator_traits_cpp17_input_iterator>;
static_assert(cuda::std::same_as<Cpp17InputIteratorTraits::iterator_category, cuda::std::input_iterator_tag>);
static_assert(cuda::std::same_as<Cpp17InputIteratorTraits::value_type, long>);
static_assert(cuda::std::same_as<Cpp17InputIteratorTraits::difference_type, int>);
static_assert(cuda::std::same_as<Cpp17InputIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<Cpp17InputIteratorTraits::pointer, void>);
static_assert(!has_iterator_concept_v<Cpp17InputIteratorTraits>);

using Cpp17ForwardIteratorTraits = cuda::std::iterator_traits<iterator_traits_cpp17_forward_iterator>;
static_assert(cuda::std::same_as<Cpp17ForwardIteratorTraits::iterator_category, cuda::std::forward_iterator_tag>);
static_assert(cuda::std::same_as<Cpp17ForwardIteratorTraits::value_type, int>);
static_assert(cuda::std::same_as<Cpp17ForwardIteratorTraits::difference_type, int>);
static_assert(cuda::std::same_as<Cpp17ForwardIteratorTraits::reference, int&>);
static_assert(cuda::std::same_as<Cpp17ForwardIteratorTraits::pointer, void>);
static_assert(!has_iterator_concept_v<Cpp17ForwardIteratorTraits>);

int main(int, char**)
{
  return 0;
}
