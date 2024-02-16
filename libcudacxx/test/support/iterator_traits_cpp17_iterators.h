//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_ITERATOR_TRAITS_ITERATOR_TRAITS_CPP17_ITERATORS
#define TEST_SUPPORT_ITERATOR_TRAITS_ITERATOR_TRAITS_CPP17_ITERATORS

struct iterator_traits_cpp17_iterator {
  TEST_HOST_DEVICE int& operator*();
  TEST_HOST_DEVICE iterator_traits_cpp17_iterator& operator++();
  TEST_HOST_DEVICE iterator_traits_cpp17_iterator operator++(int);
};

struct iterator_traits_cpp17_proxy_iterator {
  TEST_HOST_DEVICE int operator*();
  TEST_HOST_DEVICE iterator_traits_cpp17_proxy_iterator& operator++();

  // this returns legcay_iterator, not iterator_traits_cpp17_proxy_iterator
  TEST_HOST_DEVICE iterator_traits_cpp17_iterator operator++(int);
};

struct iterator_traits_cpp17_input_iterator {
  using difference_type = int;
  using value_type = long;

  TEST_HOST_DEVICE int& operator*();
  TEST_HOST_DEVICE iterator_traits_cpp17_input_iterator& operator++();
  TEST_HOST_DEVICE iterator_traits_cpp17_input_iterator operator++(int);

  TEST_HOST_DEVICE bool operator==(iterator_traits_cpp17_input_iterator const&) const;
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE bool operator!=(iterator_traits_cpp17_input_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_proxy_input_iterator {
  using difference_type = int;
  using value_type = long;

  TEST_HOST_DEVICE int operator*();
  TEST_HOST_DEVICE iterator_traits_cpp17_proxy_input_iterator& operator++();

  // this returns legcay_input_iterator, not iterator_traits_cpp17_proxy_input_iterator
  TEST_HOST_DEVICE iterator_traits_cpp17_input_iterator operator++(int);

  TEST_HOST_DEVICE bool operator==(iterator_traits_cpp17_proxy_input_iterator const&) const;
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE bool operator!=(iterator_traits_cpp17_proxy_input_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_forward_iterator {
  using difference_type = int;
  using value_type = int;

  TEST_HOST_DEVICE int& operator*();
  TEST_HOST_DEVICE iterator_traits_cpp17_forward_iterator& operator++();
  TEST_HOST_DEVICE iterator_traits_cpp17_forward_iterator operator++(int);

  TEST_HOST_DEVICE bool operator==(iterator_traits_cpp17_forward_iterator const&) const;
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE bool operator!=(iterator_traits_cpp17_forward_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_bidirectional_iterator {
  using difference_type = int;
  using value_type = int;

  TEST_HOST_DEVICE int& operator*();
  TEST_HOST_DEVICE iterator_traits_cpp17_bidirectional_iterator& operator++();
  TEST_HOST_DEVICE iterator_traits_cpp17_bidirectional_iterator operator++(int);
  TEST_HOST_DEVICE iterator_traits_cpp17_bidirectional_iterator& operator--();
  TEST_HOST_DEVICE iterator_traits_cpp17_bidirectional_iterator operator--(int);

  TEST_HOST_DEVICE bool operator==(iterator_traits_cpp17_bidirectional_iterator const&) const;
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE bool operator!=(iterator_traits_cpp17_bidirectional_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_random_access_iterator {
  using difference_type = int;
  using value_type = int;

  TEST_HOST_DEVICE int& operator*();
  TEST_HOST_DEVICE int& operator[](difference_type);
  TEST_HOST_DEVICE iterator_traits_cpp17_random_access_iterator& operator++();
  TEST_HOST_DEVICE iterator_traits_cpp17_random_access_iterator operator++(int);
  TEST_HOST_DEVICE iterator_traits_cpp17_random_access_iterator& operator--();
  TEST_HOST_DEVICE iterator_traits_cpp17_random_access_iterator operator--(int);

  TEST_HOST_DEVICE bool operator==(iterator_traits_cpp17_random_access_iterator const&) const;
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE bool operator!=(iterator_traits_cpp17_random_access_iterator const&) const;
#endif
  TEST_HOST_DEVICE bool operator<(iterator_traits_cpp17_random_access_iterator const&) const;
  TEST_HOST_DEVICE bool operator>(iterator_traits_cpp17_random_access_iterator const&) const;
  TEST_HOST_DEVICE bool operator<=(iterator_traits_cpp17_random_access_iterator const&) const;
  TEST_HOST_DEVICE bool operator>=(iterator_traits_cpp17_random_access_iterator const&) const;

  TEST_HOST_DEVICE iterator_traits_cpp17_random_access_iterator& operator+=(difference_type);
  TEST_HOST_DEVICE iterator_traits_cpp17_random_access_iterator& operator-=(difference_type);

  TEST_HOST_DEVICE friend iterator_traits_cpp17_random_access_iterator operator+(iterator_traits_cpp17_random_access_iterator,
                                                                difference_type);
  TEST_HOST_DEVICE friend iterator_traits_cpp17_random_access_iterator operator+(difference_type,
                                                                iterator_traits_cpp17_random_access_iterator);
  TEST_HOST_DEVICE friend iterator_traits_cpp17_random_access_iterator operator-(iterator_traits_cpp17_random_access_iterator,
                                                                difference_type);
  TEST_HOST_DEVICE friend difference_type operator-(iterator_traits_cpp17_random_access_iterator,
                                   iterator_traits_cpp17_random_access_iterator);
};

#endif // TEST_SUPPORT_ITERATOR_TRAITS_ITERATOR_TRAITS_CPP17_ITERATORS
