//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCUDACXX_TEST_STD_ITERATORS_ITERATOR_REQUIREMENTS_ITERATOR_CONCEPTS_INCREMENTABLE_H
#define LIBCUDACXX_TEST_STD_ITERATORS_ITERATOR_REQUIREMENTS_ITERATOR_CONCEPTS_INCREMENTABLE_H

#include "test_macros.h"

struct postfix_increment_returns_void
{
  using difference_type = int;
  TEST_FUNC postfix_increment_returns_void& operator++();
  TEST_FUNC void operator++(int);
};

struct postfix_increment_returns_copy
{
  using difference_type = int;
  TEST_FUNC postfix_increment_returns_copy& operator++();
  TEST_FUNC postfix_increment_returns_copy operator++(int);
};

struct has_integral_minus
{
  TEST_FUNC has_integral_minus& operator++();
  TEST_FUNC has_integral_minus operator++(int);

  TEST_FUNC long operator-(has_integral_minus) const;
};

struct has_distinct_difference_type_and_minus
{
  using difference_type = short;

  TEST_FUNC has_distinct_difference_type_and_minus& operator++();
  TEST_FUNC has_distinct_difference_type_and_minus operator++(int);

  TEST_FUNC long operator-(has_distinct_difference_type_and_minus) const;
};

struct missing_difference_type
{
  TEST_FUNC missing_difference_type& operator++();
  TEST_FUNC void operator++(int);
};

struct floating_difference_type
{
  using difference_type = float;

  TEST_FUNC floating_difference_type& operator++();
  TEST_FUNC void operator++(int);
};

struct non_const_minus
{
  TEST_FUNC non_const_minus& operator++();
  TEST_FUNC non_const_minus operator++(int);

  TEST_FUNC long operator-(non_const_minus);
};

struct non_integral_minus
{
  TEST_FUNC non_integral_minus& operator++();
  TEST_FUNC non_integral_minus operator++(int);

  TEST_FUNC void operator-(non_integral_minus);
};

struct bad_difference_type_good_minus
{
  using difference_type = float;

  TEST_FUNC bad_difference_type_good_minus& operator++();
  TEST_FUNC void operator++(int);

  TEST_FUNC int operator-(bad_difference_type_good_minus) const;
};

struct not_default_initializable
{
  using difference_type                 = int;
  TEST_FUNC not_default_initializable() = delete;

  TEST_FUNC not_default_initializable& operator++();
  TEST_FUNC void operator++(int);
};

struct not_movable
{
  using difference_type = int;

  not_movable()              = default;
  not_movable(not_movable&&) = delete;

  TEST_FUNC not_movable& operator++();
  TEST_FUNC void operator++(int);
};

struct preinc_not_declared
{
  using difference_type = int;

  TEST_FUNC void operator++(int);
};

struct postinc_not_declared
{
  using difference_type = int;

  TEST_FUNC postinc_not_declared& operator++();
#if TEST_COMPILER(MSVC) // MSVC complains about "single-argument function used for postfix "++" (anachronism)""
  TEST_FUNC postinc_not_declared& operator++(int) = delete;
#endif // TEST_COMPILER(MSVC)
};

struct incrementable_with_difference_type
{
  using difference_type = int;

  TEST_FUNC incrementable_with_difference_type& operator++();
  TEST_FUNC incrementable_with_difference_type operator++(int);

  TEST_FUNC bool operator==(incrementable_with_difference_type const&) const;
#if TEST_STD_VER < 2020
  TEST_FUNC bool operator!=(incrementable_with_difference_type const&) const;
#endif
};

struct incrementable_without_difference_type
{
  TEST_FUNC incrementable_without_difference_type& operator++();
  TEST_FUNC incrementable_without_difference_type operator++(int);

  TEST_FUNC bool operator==(incrementable_without_difference_type const&) const;
#if TEST_STD_VER < 2020
  TEST_FUNC bool operator!=(incrementable_without_difference_type const&) const;
#endif

  TEST_FUNC int operator-(incrementable_without_difference_type) const;
};

struct difference_type_and_void_minus
{
  using difference_type = int;

  TEST_FUNC difference_type_and_void_minus& operator++();
  TEST_FUNC difference_type_and_void_minus operator++(int);

  TEST_FUNC bool operator==(difference_type_and_void_minus const&) const;
#if TEST_STD_VER < 2020
  TEST_FUNC bool operator!=(difference_type_and_void_minus const&) const;
#endif

  TEST_FUNC void operator-(difference_type_and_void_minus) const;
};

struct noncopyable_with_difference_type
{
  using difference_type = int;

  noncopyable_with_difference_type()                                        = default;
  noncopyable_with_difference_type(noncopyable_with_difference_type&&)      = default;
  noncopyable_with_difference_type(noncopyable_with_difference_type const&) = delete;

  noncopyable_with_difference_type& operator=(noncopyable_with_difference_type&&)      = default;
  noncopyable_with_difference_type& operator=(noncopyable_with_difference_type const&) = delete;

  TEST_FUNC noncopyable_with_difference_type& operator++();
  TEST_FUNC noncopyable_with_difference_type operator++(int);

  TEST_FUNC bool operator==(noncopyable_with_difference_type const&) const;
#if TEST_STD_VER < 2020
  TEST_FUNC bool operator!=(noncopyable_with_difference_type const&) const;
#endif
};

struct noncopyable_without_difference_type
{
  noncopyable_without_difference_type()                                           = default;
  noncopyable_without_difference_type(noncopyable_without_difference_type&&)      = default;
  noncopyable_without_difference_type(noncopyable_without_difference_type const&) = delete;

  noncopyable_without_difference_type& operator=(noncopyable_without_difference_type&&)      = default;
  noncopyable_without_difference_type& operator=(noncopyable_without_difference_type const&) = delete;

  TEST_FUNC noncopyable_without_difference_type& operator++();
  TEST_FUNC noncopyable_without_difference_type operator++(int);

  TEST_FUNC int operator-(noncopyable_without_difference_type const&) const;

  TEST_FUNC bool operator==(noncopyable_without_difference_type const&) const;
#if TEST_STD_VER < 2020
  TEST_FUNC bool operator!=(noncopyable_without_difference_type const&) const;
#endif
};

struct noncopyable_with_difference_type_and_minus
{
  using difference_type = int;

  noncopyable_with_difference_type_and_minus()                                                  = default;
  noncopyable_with_difference_type_and_minus(noncopyable_with_difference_type_and_minus&&)      = default;
  noncopyable_with_difference_type_and_minus(noncopyable_with_difference_type_and_minus const&) = delete;

  noncopyable_with_difference_type_and_minus& operator=(noncopyable_with_difference_type_and_minus&&)      = default;
  noncopyable_with_difference_type_and_minus& operator=(noncopyable_with_difference_type_and_minus const&) = delete;

  TEST_FUNC noncopyable_with_difference_type_and_minus& operator++();
  TEST_FUNC noncopyable_with_difference_type_and_minus operator++(int);

  TEST_FUNC int operator-(noncopyable_with_difference_type_and_minus const&) const;

  TEST_FUNC bool operator==(noncopyable_with_difference_type_and_minus const&) const;
#if TEST_STD_VER < 2020
  TEST_FUNC bool operator!=(noncopyable_with_difference_type_and_minus const&) const;
#endif
};

#endif // #define LIBCUDACXX_TEST_STD_ITERATORS_ITERATOR_REQUIREMENTS_ITERATOR_CONCEPTS_INCREMENTABLE_H
