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
  __host__ __device__ postfix_increment_returns_void& operator++();
  __host__ __device__ void operator++(int);
};

struct postfix_increment_returns_copy
{
  using difference_type = int;
  __host__ __device__ postfix_increment_returns_copy& operator++();
  __host__ __device__ postfix_increment_returns_copy operator++(int);
};

struct has_integral_minus
{
  __host__ __device__ has_integral_minus& operator++();
  __host__ __device__ has_integral_minus operator++(int);

  __host__ __device__ long operator-(has_integral_minus) const;
};

struct has_distinct_difference_type_and_minus
{
  using difference_type = short;

  __host__ __device__ has_distinct_difference_type_and_minus& operator++();
  __host__ __device__ has_distinct_difference_type_and_minus operator++(int);

  __host__ __device__ long operator-(has_distinct_difference_type_and_minus) const;
};

struct missing_difference_type
{
  __host__ __device__ missing_difference_type& operator++();
  __host__ __device__ void operator++(int);
};

struct floating_difference_type
{
  using difference_type = float;

  __host__ __device__ floating_difference_type& operator++();
  __host__ __device__ void operator++(int);
};

struct non_const_minus
{
  __host__ __device__ non_const_minus& operator++();
  __host__ __device__ non_const_minus operator++(int);

  __host__ __device__ long operator-(non_const_minus);
};

struct non_integral_minus
{
  __host__ __device__ non_integral_minus& operator++();
  __host__ __device__ non_integral_minus operator++(int);

  __host__ __device__ void operator-(non_integral_minus);
};

struct bad_difference_type_good_minus
{
  using difference_type = float;

  __host__ __device__ bad_difference_type_good_minus& operator++();
  __host__ __device__ void operator++(int);

  __host__ __device__ int operator-(bad_difference_type_good_minus) const;
};

struct not_default_initializable
{
  using difference_type                           = int;
  __host__ __device__ not_default_initializable() = delete;

  __host__ __device__ not_default_initializable& operator++();
  __host__ __device__ void operator++(int);
};

struct not_movable
{
  using difference_type = int;

  not_movable()              = default;
  not_movable(not_movable&&) = delete;

  __host__ __device__ not_movable& operator++();
  __host__ __device__ void operator++(int);
};

struct preinc_not_declared
{
  using difference_type = int;

  __host__ __device__ void operator++(int);
};

struct postinc_not_declared
{
  using difference_type = int;

  __host__ __device__ postinc_not_declared& operator++();
#if defined(TEST_COMPILER_MSVC) // MSVC complains about "single-argument function used for postfix "++" (anachronism)""
  __host__ __device__ postinc_not_declared& operator++(int) = delete;
#endif // TEST_COMPILER_MSVC
};

struct incrementable_with_difference_type
{
  using difference_type = int;

  __host__ __device__ incrementable_with_difference_type& operator++();
  __host__ __device__ incrementable_with_difference_type operator++(int);

  __host__ __device__ bool operator==(incrementable_with_difference_type const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(incrementable_with_difference_type const&) const;
#endif
};

struct incrementable_without_difference_type
{
  __host__ __device__ incrementable_without_difference_type& operator++();
  __host__ __device__ incrementable_without_difference_type operator++(int);

  __host__ __device__ bool operator==(incrementable_without_difference_type const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(incrementable_without_difference_type const&) const;
#endif

  __host__ __device__ int operator-(incrementable_without_difference_type) const;
};

struct difference_type_and_void_minus
{
  using difference_type = int;

  __host__ __device__ difference_type_and_void_minus& operator++();
  __host__ __device__ difference_type_and_void_minus operator++(int);

  __host__ __device__ bool operator==(difference_type_and_void_minus const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(difference_type_and_void_minus const&) const;
#endif

  __host__ __device__ void operator-(difference_type_and_void_minus) const;
};

struct noncopyable_with_difference_type
{
  using difference_type = int;

  noncopyable_with_difference_type()                                        = default;
  noncopyable_with_difference_type(noncopyable_with_difference_type&&)      = default;
  noncopyable_with_difference_type(noncopyable_with_difference_type const&) = delete;

  noncopyable_with_difference_type& operator=(noncopyable_with_difference_type&&)      = default;
  noncopyable_with_difference_type& operator=(noncopyable_with_difference_type const&) = delete;

  __host__ __device__ noncopyable_with_difference_type& operator++();
  __host__ __device__ noncopyable_with_difference_type operator++(int);

  __host__ __device__ bool operator==(noncopyable_with_difference_type const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(noncopyable_with_difference_type const&) const;
#endif
};

struct noncopyable_without_difference_type
{
  noncopyable_without_difference_type()                                           = default;
  noncopyable_without_difference_type(noncopyable_without_difference_type&&)      = default;
  noncopyable_without_difference_type(noncopyable_without_difference_type const&) = delete;

  noncopyable_without_difference_type& operator=(noncopyable_without_difference_type&&)      = default;
  noncopyable_without_difference_type& operator=(noncopyable_without_difference_type const&) = delete;

  __host__ __device__ noncopyable_without_difference_type& operator++();
  __host__ __device__ noncopyable_without_difference_type operator++(int);

  __host__ __device__ int operator-(noncopyable_without_difference_type const&) const;

  __host__ __device__ bool operator==(noncopyable_without_difference_type const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(noncopyable_without_difference_type const&) const;
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

  __host__ __device__ noncopyable_with_difference_type_and_minus& operator++();
  __host__ __device__ noncopyable_with_difference_type_and_minus operator++(int);

  __host__ __device__ int operator-(noncopyable_with_difference_type_and_minus const&) const;

  __host__ __device__ bool operator==(noncopyable_with_difference_type_and_minus const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(noncopyable_with_difference_type_and_minus const&) const;
#endif
};

#endif // #define LIBCUDACXX_TEST_STD_ITERATORS_ITERATOR_REQUIREMENTS_ITERATOR_CONCEPTS_INCREMENTABLE_H
