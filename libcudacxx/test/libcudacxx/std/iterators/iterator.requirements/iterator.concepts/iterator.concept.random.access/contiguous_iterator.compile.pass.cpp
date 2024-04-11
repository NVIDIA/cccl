//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
// concept contiguous_iterator;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

static_assert(!cuda::std::contiguous_iterator<cpp17_input_iterator<int*>>);
static_assert(!cuda::std::contiguous_iterator<cpp20_input_iterator<int*>>);
static_assert(!cuda::std::contiguous_iterator<forward_iterator<int*>>);
static_assert(!cuda::std::contiguous_iterator<bidirectional_iterator<int*>>);
static_assert(!cuda::std::contiguous_iterator<random_access_iterator<int*>>);
static_assert(cuda::std::contiguous_iterator<contiguous_iterator<int*>>);

#ifndef TEST_COMPILER_MSVC_2017
static_assert(cuda::std::contiguous_iterator<int*>);
static_assert(cuda::std::contiguous_iterator<int const*>);
static_assert(cuda::std::contiguous_iterator<int volatile*>);
static_assert(cuda::std::contiguous_iterator<int const volatile*>);
#endif // TEST_COMPILER_MSVC_2017

struct simple_contiguous_iterator
{
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef int value_type;
  typedef int element_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef simple_contiguous_iterator self;

  __host__ __device__ simple_contiguous_iterator();

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const self&) const = default;
#else
  __host__ __device__ friend bool operator==(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const self&, const self&)
  {
    return false;
  };
  __host__ __device__ friend bool operator<(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator<=(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>=(const self&, const self&)
  {
    return true;
  };
#endif

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type n, self x);

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self& n) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

static_assert(cuda::std::random_access_iterator<simple_contiguous_iterator>);
static_assert(cuda::std::contiguous_iterator<simple_contiguous_iterator>);

struct mismatch_value_iter_ref_t
{
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef short value_type;
#if TEST_STD_VER < 2020
  typedef short element_type;
#endif
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef mismatch_value_iter_ref_t self;

  __host__ __device__ mismatch_value_iter_ref_t();

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const self&) const = default;
#else
  __host__ __device__ friend bool operator==(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const self&, const self&)
  {
    return false;
  };
  __host__ __device__ friend bool operator<(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator<=(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>=(const self&, const self&)
  {
    return true;
  };
#endif

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type n, self x);

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self& n) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

static_assert(cuda::std::random_access_iterator<mismatch_value_iter_ref_t>);
static_assert(!cuda::std::contiguous_iterator<mismatch_value_iter_ref_t>);

struct wrong_iter_reference_t
{
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef short value_type;
  typedef short element_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef short* pointer;
  typedef int& reference;
  typedef wrong_iter_reference_t self;

  __host__ __device__ wrong_iter_reference_t();

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const self&) const = default;
#else
  __host__ __device__ friend bool operator==(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const self&, const self&)
  {
    return false;
  };
  __host__ __device__ friend bool operator<(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator<=(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>=(const self&, const self&)
  {
    return true;
  };
#endif

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type n, self x);

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self& n) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

static_assert(cuda::std::random_access_iterator<wrong_iter_reference_t>);
static_assert(!cuda::std::contiguous_iterator<wrong_iter_reference_t>);

struct to_address_wrong_return_type
{
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef int value_type;
  typedef int element_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef to_address_wrong_return_type self;

  __host__ __device__ to_address_wrong_return_type();

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const self&) const = default;
#else
  __host__ __device__ friend bool operator==(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const self&, const self&)
  {
    return false;
  };
  __host__ __device__ friend bool operator<(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator<=(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>=(const self&, const self&)
  {
    return true;
  };
#endif

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type n, self x);

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self& n) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

template <>
struct cuda::std::pointer_traits<to_address_wrong_return_type>
{
  typedef void element_type;
  __host__ __device__ static void* to_address(to_address_wrong_return_type const&);
};

static_assert(cuda::std::random_access_iterator<to_address_wrong_return_type>);
static_assert(!cuda::std::contiguous_iterator<to_address_wrong_return_type>);

template <class>
struct template_and_no_element_type
{
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef template_and_no_element_type self;

  __host__ __device__ template_and_no_element_type();

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const self&) const = default;
#else
  __host__ __device__ friend bool operator==(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const self&, const self&)
  {
    return false;
  };
  __host__ __device__ friend bool operator<(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator<=(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>=(const self&, const self&)
  {
    return true;
  };
#endif

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type, self)
  {
    return self{};
  }

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self& n) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

// Template param is used instead of element_type.
static_assert(cuda::std::random_access_iterator<template_and_no_element_type<int>>);
static_assert(cuda::std::contiguous_iterator<template_and_no_element_type<int>>);

template <bool DisableArrow, bool DisableToAddress>
struct no_operator_arrow
{
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef int value_type;
  typedef int element_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef no_operator_arrow self;

  __host__ __device__ no_operator_arrow();

  __host__ __device__ reference operator*() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  __host__ __device__ pointer operator->() const
    requires(!DisableArrow);
  auto operator<=>(const self&) const = default;
#else
  template <bool B = DisableArrow, cuda::std::enable_if_t<!B, int> = 0>
  __host__ __device__ pointer operator->() const;
  __host__ __device__ friend bool operator==(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const self&, const self&)
  {
    return false;
  };
  __host__ __device__ friend bool operator<(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator<=(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>(const self&, const self&)
  {
    return true;
  };
  __host__ __device__ friend bool operator>=(const self&, const self&)
  {
    return true;
  };
#endif

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type, self)
  {
    return self{};
  }

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self& n) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

template <>
struct cuda::std::pointer_traits<no_operator_arrow</*DisableArrow=*/true, /*DisableToAddress=*/false>>
{
  __host__ __device__ static constexpr int* to_address(const no_operator_arrow<true, false>&);
};

static_assert(cuda::std::contiguous_iterator<no_operator_arrow</*DisableArrow=*/false, /*DisableToAddress=*/true>>);
static_assert(!cuda::std::contiguous_iterator<no_operator_arrow</*DisableArrow=*/true, /*DisableToAddress=*/true>>);
static_assert(cuda::std::contiguous_iterator<no_operator_arrow</*DisableArrow=*/true, /*DisableToAddress=*/false>>);

int main(int, char**)
{
  return 0;
}
