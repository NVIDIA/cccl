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
// concept random_access_iterator;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

static_assert(!cuda::std::random_access_iterator<cpp17_input_iterator<int*>>);
static_assert(!cuda::std::random_access_iterator<cpp20_input_iterator<int*>>);
static_assert(!cuda::std::random_access_iterator<forward_iterator<int*>>);
static_assert(!cuda::std::random_access_iterator<bidirectional_iterator<int*>>);
static_assert( cuda::std::random_access_iterator<random_access_iterator<int*>>);
static_assert( cuda::std::random_access_iterator<contiguous_iterator<int*>>);

#ifndef TEST_COMPILER_MSVC_2017
static_assert(cuda::std::random_access_iterator<int*>);
static_assert(cuda::std::random_access_iterator<int const*>);
static_assert(cuda::std::random_access_iterator<int volatile*>);
static_assert(cuda::std::random_access_iterator<int const volatile*>);
#endif // TEST_COMPILER_MSVC_2017

struct wrong_iterator_category {
    typedef cuda::std::bidirectional_iterator_tag iterator_category;
    typedef int                             value_type;
    typedef cuda::std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef wrong_iterator_category         self;

    TEST_HOST_DEVICE reference operator*() const;
    TEST_HOST_DEVICE pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif

    TEST_HOST_DEVICE self& operator++();
    TEST_HOST_DEVICE self operator++(int);

    TEST_HOST_DEVICE self& operator--();
    TEST_HOST_DEVICE self operator--(int);

    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_HOST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);

    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;

    TEST_HOST_DEVICE reference operator[](difference_type n) const;
};
static_assert(cuda::std::bidirectional_iterator<wrong_iterator_category>);
static_assert(!cuda::std::random_access_iterator<wrong_iterator_category>);

template<class Child>
struct common_base {
    typedef cuda::std::random_access_iterator_tag iterator_category;
    typedef int                             value_type;
    typedef cuda::std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef Child                           self;

    TEST_HOST_DEVICE reference operator*() const;
    TEST_HOST_DEVICE pointer operator->() const;
    TEST_HOST_DEVICE self& operator++();
    TEST_HOST_DEVICE self operator++(int);
    TEST_HOST_DEVICE self& operator--();
    TEST_HOST_DEVICE self operator--(int);
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const common_base&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const common_base&, const common_base&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const common_base&, const common_base&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const common_base&, const common_base&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const common_base&, const common_base&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const common_base&, const common_base&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const common_base&, const common_base&)  { return true; };
#endif
};

struct simple_random_access_iterator : common_base<simple_random_access_iterator> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert(cuda::std::bidirectional_iterator<simple_random_access_iterator>);
static_assert(cuda::std::random_access_iterator<simple_random_access_iterator>);

struct no_plus_equals : common_base<no_plus_equals> {
/*  TEST_HOST_DEVICE self& operator+=(difference_type n); */
    TEST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<no_plus_equals>);
static_assert(!cuda::std::random_access_iterator<no_plus_equals>);

struct no_plus_difference_type : common_base<no_plus_difference_type> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
/*  TEST_DEVICE self operator+(difference_type n) const; */
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<no_plus_difference_type>);
static_assert(!cuda::std::random_access_iterator<no_plus_difference_type>);

struct difference_type_no_plus : common_base<difference_type_no_plus> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_HOST_DEVICE self operator+(difference_type n) const;
/*  TEST_HOST_DEVICE friend self operator+(difference_type n, self x); */
    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<difference_type_no_plus>);
static_assert(!cuda::std::random_access_iterator<difference_type_no_plus>);

struct no_minus_equals : common_base<no_minus_equals> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_HOST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
/*  TEST_HOST_DEVICE self& operator-=(difference_type n); */
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<no_minus_equals>);
static_assert(!cuda::std::random_access_iterator<no_minus_equals>);

struct no_minus : common_base<no_minus> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_HOST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
    TEST_HOST_DEVICE self& operator-=(difference_type n);
/*  TEST_HOST_DEVICE self operator-(difference_type n) const; */
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<no_minus>);
static_assert(!cuda::std::random_access_iterator<no_minus>);

struct not_sized_sentinel : common_base<not_sized_sentinel> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_HOST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
/*  TEST_HOST_DEVICE difference_type operator-(const self&) const; */
    TEST_HOST_DEVICE reference operator[](difference_type n) const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<not_sized_sentinel>);
static_assert(!cuda::std::random_access_iterator<not_sized_sentinel>);

struct no_subscript : common_base<no_subscript> {
    TEST_HOST_DEVICE self& operator+=(difference_type n);
    TEST_HOST_DEVICE self operator+(difference_type n) const;
    TEST_HOST_DEVICE friend self operator+(difference_type n, self x);
    TEST_HOST_DEVICE self& operator-=(difference_type n);
    TEST_HOST_DEVICE self operator-(difference_type n) const;
    TEST_HOST_DEVICE difference_type operator-(const self&) const;
 /* TEST_HOST_DEVICE reference operator[](difference_type n) const; */
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    auto operator<=>(const self&) const = default;
#else
    TEST_HOST_DEVICE friend bool operator==(const self&, const self&) { return true; };
    TEST_HOST_DEVICE friend bool operator!=(const self&, const self&) { return false; };
    TEST_HOST_DEVICE friend bool operator<(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator<=(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>(const self&, const self&)  { return true; };
    TEST_HOST_DEVICE friend bool operator>=(const self&, const self&)  { return true; };
#endif
};
static_assert( cuda::std::bidirectional_iterator<no_subscript>);
static_assert(!cuda::std::random_access_iterator<no_subscript>);

int main(int, char**)
{
  return 0;
}
