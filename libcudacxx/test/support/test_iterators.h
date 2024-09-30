//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_ITERATORS_H
#define SUPPORT_TEST_ITERATORS_H

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_STDEXCEPT)
#  include <cuda/std/stdexcept>
#endif
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>

#include "test_macros.h"
#include "type_algorithms.h"

// This iterator meets C++20's Cpp17OutputIterator requirements, as described
// in Table 90 ([output.iterators]).
template <class It>
class cpp17_output_iterator
{
  It it_;

  template <class U>
  friend class cpp17_output_iterator;

public:
  typedef cuda::std::output_iterator_tag iterator_category;
  typedef void value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;

  __host__ __device__ TEST_CONSTEXPR explicit cpp17_output_iterator(It it)
      : it_(cuda::std::move(it))
  {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR cpp17_output_iterator(const cpp17_output_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX14 cpp17_output_iterator(cpp17_output_iterator<U>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ TEST_CONSTEXPR reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 cpp17_output_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 cpp17_output_iterator operator++(int)
  {
    return cpp17_output_iterator(it_++);
  }

  __host__ __device__ friend TEST_CONSTEXPR It base(const cpp17_output_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
#if TEST_STD_VER > 2014
static_assert(cuda::std::output_iterator<cpp17_output_iterator<int*>, int>, "");
#endif

// This iterator meets C++20's Cpp17InputIterator requirements, as described
// in Table 89 ([input.iterators]).
template <class It, class ItTraits = It>
class cpp17_input_iterator
{
  typedef cuda::std::iterator_traits<ItTraits> Traits;
  It it_;

  template <class U, class T>
  friend class cpp17_input_iterator;

public:
  typedef cuda::std::input_iterator_tag iterator_category;
  typedef typename Traits::value_type value_type;
  typedef typename Traits::difference_type difference_type;
  typedef It pointer;
  typedef typename Traits::reference reference;

  __host__ __device__ TEST_CONSTEXPR explicit cpp17_input_iterator(It it)
      : it_(it)
  {}

  template <class U, class T>
  __host__ __device__ TEST_CONSTEXPR cpp17_input_iterator(const cpp17_input_iterator<U, T>& u)
      : it_(u.it_)
  {}

  template <class U, class T, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX14 cpp17_input_iterator(cpp17_input_iterator<U, T>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ TEST_CONSTEXPR reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 cpp17_input_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 cpp17_input_iterator operator++(int)
  {
    return cpp17_input_iterator(it_++);
  }

  __host__ __device__ friend TEST_CONSTEXPR bool operator==(const cpp17_input_iterator& x, const cpp17_input_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator!=(const cpp17_input_iterator& x, const cpp17_input_iterator& y)
  {
    return x.it_ != y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR It base(const cpp17_input_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
#if TEST_STD_VER > 2014
static_assert(cuda::std::input_or_output_iterator<cpp17_input_iterator<int*>>, "");
static_assert(cuda::std::indirectly_readable<cpp17_input_iterator<int*>>, "");
static_assert(cuda::std::input_iterator<cpp17_input_iterator<int*>>, "");
#endif

template <class It>
class forward_iterator
{
  It it_;

  template <class U>
  friend class forward_iterator;

public:
  typedef cuda::std::forward_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;

  __host__ __device__ TEST_CONSTEXPR forward_iterator()
      : it_()
  {}
  __host__ __device__ TEST_CONSTEXPR explicit forward_iterator(It it)
      : it_(it)
  {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR forward_iterator(const forward_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX14 forward_iterator(forward_iterator<U>&& other)
      : it_(other.it_)
  {
    other.it_ = U();
  }

  __host__ __device__ TEST_CONSTEXPR reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 forward_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 forward_iterator operator++(int)
  {
    return forward_iterator(it_++);
  }

  __host__ __device__ friend TEST_CONSTEXPR bool operator==(const forward_iterator& x, const forward_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator!=(const forward_iterator& x, const forward_iterator& y)
  {
    return x.it_ != y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR It base(const forward_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
#if TEST_STD_VER > 2014
static_assert(cuda::std::forward_iterator<forward_iterator<int*>>, "");
#endif

template <class It>
class bidirectional_iterator
{
  It it_;

  template <class U>
  friend class bidirectional_iterator;

public:
  typedef cuda::std::bidirectional_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;

  __host__ __device__ TEST_CONSTEXPR bidirectional_iterator()
      : it_()
  {}
  __host__ __device__ TEST_CONSTEXPR explicit bidirectional_iterator(It it)
      : it_(it)
  {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR bidirectional_iterator(const bidirectional_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX14 bidirectional_iterator(bidirectional_iterator<U>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ TEST_CONSTEXPR reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 bidirectional_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 bidirectional_iterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 bidirectional_iterator operator++(int)
  {
    return bidirectional_iterator(it_++);
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 bidirectional_iterator operator--(int)
  {
    return bidirectional_iterator(it_--);
  }

  __host__ __device__ friend TEST_CONSTEXPR bool
  operator==(const bidirectional_iterator& x, const bidirectional_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool
  operator!=(const bidirectional_iterator& x, const bidirectional_iterator& y)
  {
    return x.it_ != y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR It base(const bidirectional_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
#if TEST_STD_VER > 2014
static_assert(cuda::std::bidirectional_iterator<bidirectional_iterator<int*>>, "");
#endif

template <class It>
class random_access_iterator
{
  It it_;

  template <class U>
  friend class random_access_iterator;

public:
  typedef cuda::std::random_access_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;

  __host__ __device__ TEST_CONSTEXPR random_access_iterator()
      : it_()
  {}
  __host__ __device__ TEST_CONSTEXPR explicit random_access_iterator(It it)
      : it_(it)
  {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR random_access_iterator(const random_access_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator(random_access_iterator<U>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 reference operator*() const
  {
    return *it_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 reference operator[](difference_type n) const
  {
    return it_[n];
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator operator++(int)
  {
    return random_access_iterator(it_++);
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator operator--(int)
  {
    return random_access_iterator(it_--);
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator& operator+=(difference_type n)
  {
    it_ += n;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 random_access_iterator& operator-=(difference_type n)
  {
    it_ -= n;
    return *this;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX14 random_access_iterator
  operator+(random_access_iterator x, difference_type n)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX14 random_access_iterator
  operator+(difference_type n, random_access_iterator x)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX14 random_access_iterator
  operator-(random_access_iterator x, difference_type n)
  {
    x -= n;
    return x;
  }
  __host__ __device__ friend TEST_CONSTEXPR difference_type operator-(random_access_iterator x, random_access_iterator y)
  {
    return x.it_ - y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR bool
  operator==(const random_access_iterator& x, const random_access_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool
  operator!=(const random_access_iterator& x, const random_access_iterator& y)
  {
    return x.it_ != y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool
  operator<(const random_access_iterator& x, const random_access_iterator& y)
  {
    return x.it_ < y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool
  operator<=(const random_access_iterator& x, const random_access_iterator& y)
  {
    return x.it_ <= y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool
  operator>(const random_access_iterator& x, const random_access_iterator& y)
  {
    return x.it_ > y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool
  operator>=(const random_access_iterator& x, const random_access_iterator& y)
  {
    return x.it_ >= y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR It base(const random_access_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
#if TEST_STD_VER > 2014
static_assert(cuda::std::random_access_iterator<random_access_iterator<int*>>, "");

template <class It>
class cpp20_random_access_iterator
{
  It it_;

  template <class U>
  friend class cpp20_random_access_iterator;

public:
  using iterator_category = cuda::std::input_iterator_tag;
  using iterator_concept  = cuda::std::random_access_iterator_tag;
  using value_type        = typename cuda::std::iterator_traits<It>::value_type;
  using difference_type   = typename cuda::std::iterator_traits<It>::difference_type;

  __host__ __device__ constexpr cpp20_random_access_iterator()
      : it_()
  {}
  __host__ __device__ constexpr explicit cpp20_random_access_iterator(It it)
      : it_(it)
  {}

  template <class U>
  __host__ __device__ constexpr cpp20_random_access_iterator(const cpp20_random_access_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U>
  __host__ __device__ constexpr cpp20_random_access_iterator(cpp20_random_access_iterator<U>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ constexpr decltype(auto) operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr decltype(auto) operator[](difference_type n) const
  {
    return it_[n];
  }

  __host__ __device__ constexpr cpp20_random_access_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr cpp20_random_access_iterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ constexpr cpp20_random_access_iterator operator++(int)
  {
    return cpp20_random_access_iterator(it_++);
  }
  __host__ __device__ constexpr cpp20_random_access_iterator operator--(int)
  {
    return cpp20_random_access_iterator(it_--);
  }

  __host__ __device__ constexpr cpp20_random_access_iterator& operator+=(difference_type n)
  {
    it_ += n;
    return *this;
  }
  __host__ __device__ constexpr cpp20_random_access_iterator& operator-=(difference_type n)
  {
    it_ -= n;
    return *this;
  }
  __host__ __device__ friend constexpr cpp20_random_access_iterator
  operator+(cpp20_random_access_iterator x, difference_type n)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend constexpr cpp20_random_access_iterator
  operator+(difference_type n, cpp20_random_access_iterator x)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend constexpr cpp20_random_access_iterator
  operator-(cpp20_random_access_iterator x, difference_type n)
  {
    x -= n;
    return x;
  }
  __host__ __device__ friend constexpr difference_type
  operator-(cpp20_random_access_iterator x, cpp20_random_access_iterator y)
  {
    return x.it_ - y.it_;
  }

  __host__ __device__ friend constexpr bool
  operator==(const cpp20_random_access_iterator& x, const cpp20_random_access_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool
  operator!=(const cpp20_random_access_iterator& x, const cpp20_random_access_iterator& y)
  {
    return x.it_ != y.it_;
  }
  __host__ __device__ friend constexpr bool
  operator<(const cpp20_random_access_iterator& x, const cpp20_random_access_iterator& y)
  {
    return x.it_ < y.it_;
  }
  __host__ __device__ friend constexpr bool
  operator<=(const cpp20_random_access_iterator& x, const cpp20_random_access_iterator& y)
  {
    return x.it_ <= y.it_;
  }
  __host__ __device__ friend constexpr bool
  operator>(const cpp20_random_access_iterator& x, const cpp20_random_access_iterator& y)
  {
    return x.it_ > y.it_;
  }
  __host__ __device__ friend constexpr bool
  operator>=(const cpp20_random_access_iterator& x, const cpp20_random_access_iterator& y)
  {
    return x.it_ >= y.it_;
  }

  __host__ __device__ friend constexpr It base(const cpp20_random_access_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};

static_assert(cuda::std::random_access_iterator<cpp20_random_access_iterator<int*>>, "");

template <class It>
class contiguous_iterator
{
  static_assert(cuda::std::is_pointer_v<It>, "Things probably break in this case");

  It it_;

  template <class U>
  friend class contiguous_iterator;

public:
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;
  typedef typename cuda::std::remove_pointer<It>::type element_type;

  __host__ __device__ TEST_CONSTEXPR_CXX14 It base() const
  {
    return it_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator()
      : it_()
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 explicit contiguous_iterator(It it)
      : it_(it)
  {}

  template <class U, class = cuda::std::enable_if_t<cuda::std::is_constructible<It, const U&>::value>>
  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator(const contiguous_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U,
            class = cuda::std::enable_if_t<cuda::std::is_constructible<It, U>::value
                                           && cuda::std::is_default_constructible<U>::value>>
  __host__ __device__ constexpr contiguous_iterator(contiguous_iterator<U>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ TEST_CONSTEXPR reference operator*() const
  {
    return *it_;
  }
  __host__ __device__ TEST_CONSTEXPR pointer operator->() const
  {
    return it_;
  }
  __host__ __device__ TEST_CONSTEXPR reference operator[](difference_type n) const
  {
    return it_[n];
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator operator++(int)
  {
    return contiguous_iterator(it_++);
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator operator--(int)
  {
    return contiguous_iterator(it_--);
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator& operator+=(difference_type n)
  {
    it_ += n;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 contiguous_iterator& operator-=(difference_type n)
  {
    it_ -= n;
    return *this;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX14 contiguous_iterator operator+(contiguous_iterator x, difference_type n)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX14 contiguous_iterator operator+(difference_type n, contiguous_iterator x)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX14 contiguous_iterator operator-(contiguous_iterator x, difference_type n)
  {
    x -= n;
    return x;
  }
  __host__ __device__ friend TEST_CONSTEXPR difference_type operator-(contiguous_iterator x, contiguous_iterator y)
  {
    return x.it_ - y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR bool operator==(const contiguous_iterator& x, const contiguous_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator!=(const contiguous_iterator& x, const contiguous_iterator& y)
  {
    return x.it_ != y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator<(const contiguous_iterator& x, const contiguous_iterator& y)
  {
    return x.it_ < y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator<=(const contiguous_iterator& x, const contiguous_iterator& y)
  {
    return x.it_ <= y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator>(const contiguous_iterator& x, const contiguous_iterator& y)
  {
    return x.it_ > y.it_;
  }
  __host__ __device__ friend TEST_CONSTEXPR bool operator>=(const contiguous_iterator& x, const contiguous_iterator& y)
  {
    return x.it_ >= y.it_;
  }

  __host__ __device__ friend TEST_CONSTEXPR It base(const contiguous_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
static_assert(cuda::std::random_access_iterator<contiguous_iterator<int*>>, "");

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR

template <class It>
class three_way_contiguous_iterator
{
  static_assert(cuda::std::is_pointer_v<It>, "Things probably break in this case");

  It it_;

  template <class U>
  friend class three_way_contiguous_iterator;

public:
  typedef cuda::std::contiguous_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;
  typedef typename cuda::std::remove_pointer<It>::type element_type;

  __host__ __device__ constexpr It base() const
  {
    return it_;
  }

  __host__ __device__ constexpr three_way_contiguous_iterator()
      : it_()
  {}
  __host__ __device__ constexpr explicit three_way_contiguous_iterator(It it)
      : it_(it)
  {}

  template <class U>
  __host__ __device__ constexpr three_way_contiguous_iterator(const three_way_contiguous_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ constexpr three_way_contiguous_iterator(three_way_contiguous_iterator<U>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr pointer operator->() const
  {
    return it_;
  }
  __host__ __device__ constexpr reference operator[](difference_type n) const
  {
    return it_[n];
  }

  __host__ __device__ constexpr three_way_contiguous_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr three_way_contiguous_iterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ constexpr three_way_contiguous_iterator operator++(int)
  {
    return three_way_contiguous_iterator(it_++);
  }
  __host__ __device__ constexpr three_way_contiguous_iterator operator--(int)
  {
    return three_way_contiguous_iterator(it_--);
  }

  __host__ __device__ constexpr three_way_contiguous_iterator& operator+=(difference_type n)
  {
    it_ += n;
    return *this;
  }
  __host__ __device__ constexpr three_way_contiguous_iterator& operator-=(difference_type n)
  {
    it_ -= n;
    return *this;
  }
  __host__ __device__ friend constexpr three_way_contiguous_iterator
  operator+(three_way_contiguous_iterator x, difference_type n)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend constexpr three_way_contiguous_iterator
  operator+(difference_type n, three_way_contiguous_iterator x)
  {
    x += n;
    return x;
  }
  __host__ __device__ friend constexpr three_way_contiguous_iterator
  operator-(three_way_contiguous_iterator x, difference_type n)
  {
    x -= n;
    return x;
  }
  __host__ __device__ friend constexpr difference_type
  operator-(three_way_contiguous_iterator x, three_way_contiguous_iterator y)
  {
    return x.it_ - y.it_;
  }

  __host__ __device__ friend constexpr auto
  operator<=>(const three_way_contiguous_iterator& x, const three_way_contiguous_iterator& y)
  {
    return x.it_ <=> y.it_;
  }
  __host__ __device__ friend constexpr bool
  operator==(const three_way_contiguous_iterator& x, const three_way_contiguous_iterator& y)
  {
    return x.it_ == y.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 2011

template <class Iter> // ADL base() for everything else (including pointers)
__host__ __device__ TEST_CONSTEXPR Iter base(Iter i)
{
  return i;
}

template <typename T>
struct ThrowingIterator
{
  typedef cuda::std::bidirectional_iterator_tag iterator_category;
  typedef ptrdiff_t difference_type;
  typedef const T value_type;
  typedef const T* pointer;
  typedef const T& reference;

  enum ThrowingAction
  {
    TAIncrement,
    TADecrement,
    TADereference,
    TAAssignment,
    TAComparison
  };

  __host__ __device__ TEST_CONSTEXPR ThrowingIterator()
      : begin_(nullptr)
      , end_(nullptr)
      , current_(nullptr)
      , action_(TADereference)
      , index_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR explicit ThrowingIterator(
    const T* first, const T* last, int index = 0, ThrowingAction action = TADereference)
      : begin_(first)
      , end_(last)
      , current_(first)
      , action_(action)
      , index_(index)
  {}
  __host__ __device__ TEST_CONSTEXPR ThrowingIterator(const ThrowingIterator& rhs)
      : begin_(rhs.begin_)
      , end_(rhs.end_)
      , current_(rhs.current_)
      , action_(rhs.action_)
      , index_(rhs.index_)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX14 ThrowingIterator& operator=(const ThrowingIterator& rhs)
  {
    if (action_ == TAAssignment && --index_ < 0)
    {
      assert(false);
    }
    begin_   = rhs.begin_;
    end_     = rhs.end_;
    current_ = rhs.current_;
    action_  = rhs.action_;
    index_   = rhs.index_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 reference operator*() const
  {
    if (action_ == TADereference && --index_ < 0)
    {
      assert(false);
    }
    return *current_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 ThrowingIterator& operator++()
  {
    if (action_ == TAIncrement && --index_ < 0)
    {
      assert(false);
    }
    ++current_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 ThrowingIterator operator++(int)
  {
    ThrowingIterator temp = *this;
    ++(*this);
    return temp;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 ThrowingIterator& operator--()
  {
    if (action_ == TADecrement && --index_ < 0)
    {
      assert(false);
    }
    --current_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 ThrowingIterator operator--(int)
  {
    ThrowingIterator temp = *this;
    --(*this);
    return temp;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX14 friend bool operator==(const ThrowingIterator& a, const ThrowingIterator& b)
  {
    if (a.action_ == TAComparison && --a.index_ < 0)
    {
      assert(false);
    }
    bool atEndL = a.current_ == a.end_;
    bool atEndR = b.current_ == b.end_;
    if (atEndL != atEndR)
    {
      return false; // one is at the end (or empty), the other is not.
    }
    if (atEndL)
    {
      return true; // both are at the end (or empty)
    }
    return a.current_ == b.current_;
  }

  __host__ __device__ TEST_CONSTEXPR friend bool operator!=(const ThrowingIterator& a, const ThrowingIterator& b)
  {
    return !(a == b);
  }

  template <class T2>
  void operator,(T2 const&) = delete;

private:
  const T* begin_;
  const T* end_;
  const T* current_;
  ThrowingAction action_;
  mutable int index_;
};

template <typename T>
struct NonThrowingIterator
{
  typedef cuda::std::bidirectional_iterator_tag iterator_category;
  typedef ptrdiff_t difference_type;
  typedef const T value_type;
  typedef const T* pointer;
  typedef const T& reference;

  __host__ __device__ NonThrowingIterator()
      : begin_(nullptr)
      , end_(nullptr)
      , current_(nullptr)
  {}
  __host__ __device__ explicit NonThrowingIterator(const T* first, const T* last)
      : begin_(first)
      , end_(last)
      , current_(first)
  {}
  __host__ __device__ NonThrowingIterator(const NonThrowingIterator& rhs)
      : begin_(rhs.begin_)
      , end_(rhs.end_)
      , current_(rhs.current_)
  {}

  __host__ __device__ NonThrowingIterator& operator=(const NonThrowingIterator& rhs) TEST_NOEXCEPT
  {
    begin_   = rhs.begin_;
    end_     = rhs.end_;
    current_ = rhs.current_;
    return *this;
  }

  __host__ __device__ reference operator*() const TEST_NOEXCEPT
  {
    return *current_;
  }

  __host__ __device__ NonThrowingIterator& operator++() TEST_NOEXCEPT
  {
    ++current_;
    return *this;
  }

  __host__ __device__ NonThrowingIterator operator++(int) TEST_NOEXCEPT
  {
    NonThrowingIterator temp = *this;
    ++(*this);
    return temp;
  }

  __host__ __device__ NonThrowingIterator& operator--() TEST_NOEXCEPT
  {
    --current_;
    return *this;
  }

  __host__ __device__ NonThrowingIterator operator--(int) TEST_NOEXCEPT
  {
    NonThrowingIterator temp = *this;
    --(*this);
    return temp;
  }

  __host__ __device__ friend bool operator==(const NonThrowingIterator& a, const NonThrowingIterator& b) TEST_NOEXCEPT
  {
    bool atEndL = a.current_ == a.end_;
    bool atEndR = b.current_ == b.end_;
    if (atEndL != atEndR)
    {
      return false; // one is at the end (or empty), the other is not.
    }
    if (atEndL)
    {
      return true; // both are at the end (or empty)
    }
    return a.current_ == b.current_;
  }

  __host__ __device__ friend bool operator!=(const NonThrowingIterator& a, const NonThrowingIterator& b) TEST_NOEXCEPT
  {
    return !(a == b);
  }

  template <class T2>
  void operator,(T2 const&) = delete;

private:
  const T* begin_;
  const T* end_;
  const T* current_;
};

#if TEST_STD_VER > 2014

template <class It>
class cpp20_input_iterator
{
  It it_;

public:
  using value_type       = cuda::std::iter_value_t<It>;
  using difference_type  = cuda::std::iter_difference_t<It>;
  using iterator_concept = cuda::std::input_iterator_tag;

  __host__ __device__ constexpr explicit cpp20_input_iterator(It it)
      : it_(it)
  {}

#  ifndef TEST_COMPILER_MSVC_2017 // MSVC2017 has issues determining common_reference
  cpp20_input_iterator(cpp20_input_iterator&&)            = default;
  cpp20_input_iterator& operator=(cpp20_input_iterator&&) = default;
#  endif // !TEST_COMPILER_MSVC_2017

  __host__ __device__ constexpr decltype(auto) operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr cpp20_input_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++it_;
  }

  __host__ __device__ friend constexpr It base(const cpp20_input_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
static_assert(cuda::std::input_iterator<cpp20_input_iterator<int*>>, "");

template <class, class = void>
struct iter_value_or_void;

template <class It>
struct iter_value_or_void<
  It,
  cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It> && !cuda::std::input_iterator<It>>>
{
  using type = void;
};

template <class It>
struct iter_value_or_void<It, cuda::std::enable_if_t<cuda::std::input_iterator<It>>>
{
  using type = cuda::std::iter_value_t<It>;
};

template <class It>
class cpp20_output_iterator
{
  It it_;

public:
  using difference_type = cuda::std::iter_difference_t<It>;

  __host__ __device__ constexpr explicit cpp20_output_iterator(It it)
      : it_(it)
  {}

#  ifndef TEST_COMPILER_MSVC_2017 // MSVC2017 has issues determining common_reference
  cpp20_output_iterator(cpp20_output_iterator&&)            = default;
  cpp20_output_iterator& operator=(cpp20_output_iterator&&) = default;
#  endif // !TEST_COMPILER_MSVC_2017

  __host__ __device__ constexpr decltype(auto) operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr cpp20_output_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr cpp20_output_iterator operator++(int)
  {
    return cpp20_output_iterator(it_++);
  }

  __host__ __device__ friend constexpr It base(const cpp20_output_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};

static_assert(cuda::std::output_iterator<cpp20_output_iterator<int*>, int>, "");

// Iterator adaptor that counts the number of times the iterator has had a successor/predecessor
// operation called. Has two recorders:
// * `stride_count`, which records the total number of calls to an op++, op--, op+=, or op-=.
// * `stride_displacement`, which records the displacement of the calls. This means that both
//   op++/op+= will increase the displacement counter by 1, and op--/op-= will decrease the
//   displacement counter by 1.
template <class It>
class stride_counting_iterator
{
public:
  using value_type       = typename iter_value_or_void<It>::type;
  using difference_type  = cuda::std::iter_difference_t<It>;
  using iterator_concept = cuda::std::conditional_t<
    cuda::std::contiguous_iterator<It>,
    cuda::std::contiguous_iterator_tag,
    cuda::std::conditional_t<
      cuda::std::random_access_iterator<It>,
      cuda::std::random_access_iterator_tag,
      cuda::std::conditional_t<
        cuda::std::bidirectional_iterator<It>,
        cuda::std::bidirectional_iterator_tag,
        cuda::std::conditional_t<cuda::std::forward_iterator<It>,
                                 cuda::std::forward_iterator_tag,
                                 cuda::std::conditional_t<cuda::std::input_iterator<It>,
                                                          cuda::std::input_iterator_tag,
                                                          /* else */ cuda::std::output_iterator_tag>>>>>;

#  if TEST_STD_VER > 2017
  stride_counting_iterator()
    requires cuda::std::default_initializable<It>
  = default;
#  else
  template <class It2 = It, cuda::std::enable_if_t<cuda::std::default_initializable<It2>, int> = 0>
  __host__ __device__ constexpr stride_counting_iterator() noexcept {};
#  endif

  __host__ __device__ constexpr explicit stride_counting_iterator(It const& it)
      : base_(base(it))
  {}

  __host__ __device__ friend constexpr It base(stride_counting_iterator const& it)
  {
    return It(it.base_);
  }

  __host__ __device__ constexpr difference_type stride_count() const
  {
    return stride_count_;
  }

  __host__ __device__ constexpr difference_type stride_displacement() const
  {
    return stride_displacement_;
  }

  __host__ __device__ constexpr decltype(auto) operator*() const
  {
    return *It(base_);
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ constexpr decltype(auto) operator[](difference_type n) const
  {
    return It(base_)[n];
  }

  __host__ __device__ constexpr stride_counting_iterator& operator++()
  {
    It tmp(base_);
    base_ = base(++tmp);
    ++stride_count_;
    ++stride_displacement_;
    return *this;
  }

  template <class It2 = It, cuda::std::enable_if_t<!cuda::std::forward_iterator<It2>, int> = 0>
  __host__ __device__ constexpr void operator++(int)
  {
    ++*this;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::forward_iterator<It2>, int> = 0>
  __host__ __device__ constexpr stride_counting_iterator operator++(int)
  {
    auto temp = *this;
    ++*this;
    return temp;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::bidirectional_iterator<It2>, int> = 0>
  __host__ __device__ constexpr stride_counting_iterator& operator--()
  {
    It tmp(base_);
    base_ = base(--tmp);
    ++stride_count_;
    --stride_displacement_;
    return *this;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::bidirectional_iterator<It2>, int> = 0>
  __host__ __device__ constexpr stride_counting_iterator operator--(int)
  {
    auto temp = *this;
    --*this;
    return temp;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ constexpr stride_counting_iterator& operator+=(difference_type const n)
  {
    It tmp(base_);
    base_ = base(tmp += n);
    ++stride_count_;
    ++stride_displacement_;
    return *this;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ constexpr stride_counting_iterator& operator-=(difference_type const n)
  {
    It tmp(base_);
    base_ = base(tmp -= n);
    ++stride_count_;
    --stride_displacement_;
    return *this;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr stride_counting_iterator operator+(stride_counting_iterator it, difference_type n)
  {
    return it += n;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr stride_counting_iterator operator+(difference_type n, stride_counting_iterator it)
  {
    return it += n;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr stride_counting_iterator operator-(stride_counting_iterator it, difference_type n)
  {
    return it -= n;
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::sized_sentinel_for<It2, It2>, int> = 0>
  __host__ __device__ friend constexpr difference_type
  operator-(stride_counting_iterator const& x, stride_counting_iterator const& y)
  {
    return base(x) - base(y);
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::sentinel_for<It2, It2>, int> = 0>
  __host__ __device__ constexpr bool operator==(stride_counting_iterator const& other) const
  {
    return It(base_) == It(other.base_);
  }

#  if TEST_STD_VER < 2020
  template <class It2 = It, cuda::std::enable_if_t<cuda::std::sentinel_for<It2, It2>, int> = 0>
  __host__ __device__ constexpr bool operator!=(stride_counting_iterator const& other) const
  {
    return It(base_) != It(other.base_);
  }
#  endif

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr bool
  operator<(stride_counting_iterator const& x, stride_counting_iterator const& y)
  {
    return It(x.base_) < It(y.base_);
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr bool
  operator>(stride_counting_iterator const& x, stride_counting_iterator const& y)
  {
    return It(x.base_) > It(y.base_);
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr bool
  operator<=(stride_counting_iterator const& x, stride_counting_iterator const& y)
  {
    return It(x.base_) <= It(y.base_);
  }

  template <class It2 = It, cuda::std::enable_if_t<cuda::std::random_access_iterator<It2>, int> = 0>
  __host__ __device__ friend constexpr bool
  operator>=(stride_counting_iterator const& x, stride_counting_iterator const& y)
  {
    return It(x.base_) >= It(y.base_);
  }

  template <class T>
  void operator,(T const&) = delete;

private:
  decltype(base(cuda::std::declval<It>())) base_;
  difference_type stride_count_        = 0;
  difference_type stride_displacement_ = 0;
};

template <class It>
class sentinel_wrapper
{
public:
  explicit sentinel_wrapper() = default;
  __host__ __device__ constexpr explicit sentinel_wrapper(const It& it)
      : base_(base(it))
  {}
  __host__ __device__ friend constexpr bool operator==(const sentinel_wrapper& s, const It& i)
  {
    return s.base_ == base(i);
  }
#  if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const It& i, const sentinel_wrapper& s)
  {
    return s.base_ == base(i);
  }
  __host__ __device__ friend constexpr bool operator!=(const sentinel_wrapper& s, const It& i)
  {
    return s.base_ != base(i);
  }
  __host__ __device__ friend constexpr bool operator!=(const It& i, const sentinel_wrapper& s)
  {
    return s.base_ != base(i);
  }
#  endif
  __host__ __device__ friend constexpr It base(const sentinel_wrapper& s)
  {
    return It(s.base_);
  }

private:
  decltype(base(cuda::std::declval<It>())) base_;
};

template <class It>
class sized_sentinel
{
public:
  explicit sized_sentinel() = default;
  __host__ __device__ constexpr explicit sized_sentinel(const It& it)
      : base_(base(it))
  {}
  __host__ __device__ friend constexpr bool operator==(const sized_sentinel& s, const It& i)
  {
    return s.base_ == base(i);
  }
#  if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const It& i, const sized_sentinel& s)
  {
    return s.base_ == base(i);
  }
  __host__ __device__ friend constexpr bool operator!=(const sized_sentinel& s, const It& i)
  {
    return s.base_ != base(i);
  }
  __host__ __device__ friend constexpr bool operator!=(const It& i, const sized_sentinel& s)
  {
    return s.base_ != base(i);
  }
#  endif
  __host__ __device__ friend constexpr auto operator-(const sized_sentinel& s, const It& i)
  {
    return s.base_ - base(i);
  }
  __host__ __device__ friend constexpr auto operator-(const It& i, const sized_sentinel& s)
  {
    return base(i) - s.base_;
  }
  __host__ __device__ friend constexpr It base(const sized_sentinel& s)
  {
    return It(s.base_);
  }

private:
  decltype(base(cuda::std::declval<It>())) base_;
};

namespace adl
{

class Iterator
{
public:
  using value_type      = int;
  using reference       = int&;
  using difference_type = ptrdiff_t;

private:
  value_type* ptr_ = nullptr;
  int* iter_moves_ = nullptr;
  int* iter_swaps_ = nullptr;

  __host__ __device__ constexpr Iterator(int* p, int* iter_moves, int* iter_swaps)
      : ptr_(p)
      , iter_moves_(iter_moves)
      , iter_swaps_(iter_swaps)
  {}

public:
  constexpr Iterator() = default;
  __host__ __device__ static constexpr Iterator TrackMoves(int* p, int& iter_moves)
  {
    return Iterator(p, &iter_moves, /*iter_swaps=*/nullptr);
  }
  __host__ __device__ static constexpr Iterator TrackSwaps(int& iter_swaps)
  {
    return Iterator(/*p=*/nullptr, /*iter_moves=*/nullptr, &iter_swaps);
  }
  __host__ __device__ static constexpr Iterator TrackSwaps(int* p, int& iter_swaps)
  {
    return Iterator(p, /*iter_moves=*/nullptr, &iter_swaps);
  }

  __host__ __device__ constexpr int iter_moves() const
  {
    assert(iter_moves_);
    return *iter_moves_;
  }
  __host__ __device__ constexpr int iter_swaps() const
  {
    assert(iter_swaps_);
    return *iter_swaps_;
  }

  __host__ __device__ constexpr value_type& operator*() const
  {
    return *ptr_;
  }
  __host__ __device__ constexpr reference operator[](difference_type n) const
  {
    return ptr_[n];
  }

  __host__ __device__ friend constexpr Iterator operator+(Iterator i, difference_type n)
  {
    return Iterator(i.ptr_ + n, i.iter_moves_, i.iter_swaps_);
  }
  __host__ __device__ friend constexpr Iterator operator+(difference_type n, Iterator i)
  {
    return i + n;
  }
  __host__ __device__ constexpr Iterator operator-(difference_type n) const
  {
    return Iterator(ptr_ - n, iter_moves_, iter_swaps_);
  }
  __host__ __device__ constexpr difference_type operator-(Iterator rhs) const
  {
    return ptr_ - rhs.ptr_;
  }
  __host__ __device__ constexpr Iterator& operator+=(difference_type n)
  {
    ptr_ += n;
    return *this;
  }
  __host__ __device__ constexpr Iterator& operator-=(difference_type n)
  {
    ptr_ -= n;
    return *this;
  }

  __host__ __device__ constexpr Iterator& operator++()
  {
    ++ptr_;
    return *this;
  }
  __host__ __device__ constexpr Iterator operator++(int)
  {
    Iterator prev = *this;
    ++ptr_;
    return prev;
  }

  __host__ __device__ constexpr Iterator& operator--()
  {
    --ptr_;
    return *this;
  }
  __host__ __device__ constexpr Iterator operator--(int)
  {
    Iterator prev = *this;
    --ptr_;
    return prev;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 friend void iter_swap(Iterator a, Iterator b)
  {
    cuda::std::swap(a.ptr_, b.ptr_);
    if (a.iter_swaps_)
    {
      ++(*a.iter_swaps_);
    }
  }

  __host__ __device__ constexpr friend value_type&& iter_move(Iterator iter)
  {
    if (iter.iter_moves_)
    {
      ++(*iter.iter_moves_);
    }
    return cuda::std::move(*iter);
  }

  __host__ __device__ constexpr friend bool operator==(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ == rhs.ptr_;
  }
#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  __host__ __device__ constexpr friend auto operator<=>(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ <=> rhs.ptr_;
  }
#  else
  __host__ __device__ constexpr friend bool operator!=(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ != rhs.ptr_;
  }
  __host__ __device__ constexpr friend bool operator<(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ < rhs.ptr_;
  }
  __host__ __device__ constexpr friend bool operator<=(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ <= rhs.ptr_;
  }
  __host__ __device__ constexpr friend bool operator>(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ > rhs.ptr_;
  }
  __host__ __device__ constexpr friend bool operator>=(const Iterator& lhs, const Iterator& rhs)
  {
    return lhs.ptr_ >= rhs.ptr_;
  }
#  endif // TEST_STD_VER < 2020
};

} // namespace adl

// Proxy
// ======================================================================
// Proxy that can wrap a value or a reference. It simulates C++23's tuple
// but simplified to just hold one argument.
// Note that unlike tuple, this class deliberately doesn't have special handling
// of swap to cause a compilation error if it's used in an algorithm that relies
// on plain swap instead of ranges::iter_swap.
// This class is useful for testing that if algorithms support proxy iterator
// properly, i.e. calling ranges::iter_swap and ranges::iter_move instead of
// plain swap and cuda::std::move.
template <class T>
struct Proxy;

template <class T>
_LIBCUDACXX_INLINE_VAR constexpr bool IsProxy = false;

template <class T>
_LIBCUDACXX_INLINE_VAR constexpr bool IsProxy<Proxy<T>> = true;

template <class T>
struct Proxy
{
  T data;

  __host__ __device__ constexpr T& getData() &
  {
    return data;
  }

  __host__ __device__ constexpr const T& getData() const&
  {
    return data;
  }

  __host__ __device__ constexpr T&& getData() &&
  {
    return static_cast<T&&>(data);
  }

  __host__ __device__ constexpr const T&& getData() const&&
  {
    return static_cast<const T&&>(data);
  }

  _LIBCUDACXX_TEMPLATE(class U)
  _LIBCUDACXX_REQUIRES(cuda::std::constructible_from<T, U&&>)
  __host__ __device__ constexpr Proxy(U&& u)
      : data{cuda::std::forward<U>(u)}
  {}

  // This constructor covers conversion from cvref of Proxy<U>, including non-const/const versions of copy/move
  // constructor
  _LIBCUDACXX_TEMPLATE(class Other)
  _LIBCUDACXX_REQUIRES((IsProxy<cuda::std::decay_t<Other>>
                        && cuda::std::constructible_from<T, decltype(cuda::std::declval<Other>().getData())>) )
  __host__ __device__ constexpr Proxy(Other&& other)
      : data{cuda::std::forward<Other>(other).getData()}
  {}

  _LIBCUDACXX_TEMPLATE(class Other)
  _LIBCUDACXX_REQUIRES((IsProxy<cuda::std::decay_t<Other>>
                        && cuda::std::assignable_from<cuda::std::__add_lvalue_reference_t<T>,
                                                      decltype(cuda::std::declval<Other>().getData())>) )
  __host__ __device__ constexpr Proxy& operator=(Other&& other)
  {
    data = cuda::std::forward<Other>(other).getData();
    return *this;
  }

#  if defined(TEST_COMPILER_MSVC)
  TEST_NV_DIAG_SUPPRESS(1805) // MSVC complains that if we pass a pointer type, adding const is useless
#  endif // TEST_COMPILER_MSVC

  // const assignment required to make ProxyIterator model cuda::std::indirectly_writable
  _LIBCUDACXX_TEMPLATE(class Other)
  _LIBCUDACXX_REQUIRES((IsProxy<cuda::std::decay_t<Other>>
                        && cuda::std::assignable_from<const cuda::std::__add_lvalue_reference_t<T>,
                                                      decltype(cuda::std::declval<Other>().getData())>) )
  __host__ __device__ constexpr const Proxy& operator=(Other&& other) const
  {
    data = cuda::std::forward<Other>(other).getData();
    return *this;
  }

  // If `T` is a reference type, the implicitly-generated assignment operator will be deleted (and would take precedence
  // over the templated `operator=` above because it's a better match).
  __host__ __device__ constexpr Proxy& operator=(const Proxy& rhs)
  {
    data = rhs.data;
    return *this;
  }

  // no specialised swap function that takes const Proxy& and no specialised const member swap
  // Calling swap(Proxy<T>{}, Proxy<T>{}) would fail (pass prvalues)

  // Compare operators are defined for the convenience of the tests
#  if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const Proxy&, const Proxy&)
    requires(cuda::std::equality_comparable<T> && !cuda::std::is_reference_v<T>)
  = default;
#  else
  _LIBCUDACXX_TEMPLATE(class T2 = T)
  _LIBCUDACXX_REQUIRES((cuda::std::equality_comparable<T2> && !cuda::std::is_reference_v<T2>) )
  __host__ __device__ friend constexpr bool operator==(const Proxy& lhs, const Proxy& rhs)
  {
    return lhs.data == rhs.data;
  }
#  endif // TEST_STD_VER > 2017

  // Helps compare e.g. `Proxy<int>` and `Proxy<int&>`. Note that the default equality comparison operator is deleted
  // when `T` is a reference type.
  _LIBCUDACXX_TEMPLATE(class U)
  _LIBCUDACXX_REQUIRES((cuda::std::equality_comparable_with<cuda::std::decay_t<T>, cuda::std::decay_t<U>>) )
  __host__ __device__ friend constexpr bool operator==(const Proxy& lhs, const Proxy<U>& rhs)
  {
    return lhs.data == rhs.data;
  }

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  __host__ __device__ friend constexpr auto operator<=>(const Proxy&, const Proxy&)
    requires(cuda::std::three_way_comparable<T> && !cuda::std::is_reference_v<T>)
  = default;

  // Helps compare e.g. `Proxy<int>` and `Proxy<int&>`. Note that the default 3-way comparison operator is deleted when
  // `T` is a reference type.
  template <class U>
  __host__ __device__ friend constexpr auto operator<=>(const Proxy& lhs, const Proxy<U>& rhs)
    requires cuda::std::three_way_comparable_with<cuda::std::decay_t<T>, cuda::std::decay_t<U>>
  {
    return lhs.data <=> rhs.data;
  }
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
};

// This is to make ProxyIterator model `cuda::std::indirectly_readable`
template <class T, class U, template <class> class TQual, template <class> class UQual>
struct cuda::std::basic_common_reference<Proxy<T>, Proxy<U>, TQual, UQual>
    : public cuda::std::enable_if<cuda::std::__has_common_reference<TQual<T>, UQual<U>>,
                                  Proxy<cuda::std::common_reference_t<TQual<T>, UQual<U>>>>
{};

template <class T, class U>
struct cuda::std::common_type<Proxy<T>, Proxy<U>>
    : public cuda::std::enable_if<cuda::std::__has_common_type<T, U>, Proxy<cuda::std::common_type_t<T, U>>>
{};

// ProxyIterator
// ======================================================================
// It wraps `Base` iterator and when dereferenced it returns a Proxy<ref>
// It simulates C++23's zip_view::iterator but simplified to just wrap
// one base iterator.
// Note it forwards value_type, iter_move, iter_swap. e.g if the base
// iterator is int*,
// operator*    -> Proxy<int&>
// iter_value_t -> Proxy<int>
// iter_move    -> Proxy<int&&>
template <class Base, class = void>
struct ProxyIteratorBase
{};

template <class Base>
struct ProxyIteratorBase<
  Base,
  cuda::std::enable_if_t<cuda::std::derived_from<typename cuda::std::iterator_traits<Base>::iterator_category,
                                                 cuda::std::input_iterator_tag>>>
{
  using iterator_category = cuda::std::input_iterator_tag;
};

template <class Base, cuda::std::enable_if_t<cuda::std::input_iterator<Base>, int> = 0>
__host__ __device__ constexpr auto get_iterator_concept()
{
  if constexpr (cuda::std::random_access_iterator<Base>)
  {
    return cuda::std::random_access_iterator_tag{};
  }
  else if constexpr (cuda::std::bidirectional_iterator<Base>)
  {
    return cuda::std::bidirectional_iterator_tag{};
  }
  else if constexpr (cuda::std::forward_iterator<Base>)
  {
    return cuda::std::forward_iterator_tag{};
  }
  else
  {
    return cuda::std::input_iterator_tag{};
  }
  _CCCL_UNREACHABLE();
}

template <class Base, cuda::std::enable_if_t<cuda::std::input_iterator<Base>, int> = 0>
struct ProxyIterator : ProxyIteratorBase<Base>
{
  Base base_;

  using iterator_concept = decltype(get_iterator_concept<Base>());
  using value_type       = Proxy<cuda::std::iter_value_t<Base>>;
  using difference_type  = cuda::std::iter_difference_t<Base>;

#  if TEST_STD_VER > 2017
  ProxyIterator()
    requires cuda::std::default_initializable<Base>
  = default;
#  else
  template <class B2 = Base, cuda::std::enable_if_t<cuda::std::default_initializable<B2>, int> = 0>
  __host__ __device__ constexpr ProxyIterator() noexcept {};
#  endif // TEST_STD_VER > 2017

  __host__ __device__ constexpr ProxyIterator(Base base)
      : base_{cuda::std::move(base)}
  {}

  _LIBCUDACXX_TEMPLATE(class T)
  _LIBCUDACXX_REQUIRES(cuda::std::constructible_from<Base, T&&>)
  __host__ __device__ constexpr ProxyIterator(T&& t)
      : base_{cuda::std::forward<T>(t)}
  {}

  __host__ __device__ friend constexpr decltype(auto) base(const ProxyIterator& p)
  {
    return base(p.base_);
  }

  // Specialization of iter_move
  // If operator* returns Proxy<Foo&>, iter_move will return Proxy<Foo&&>
  // Note cuda::std::move(*it) returns Proxy<Foo&>&&, which is not what we want as
  // it will likely result in a copy rather than a move
  __host__ __device__ friend constexpr Proxy<cuda::std::iter_rvalue_reference_t<Base>>
  iter_move(const ProxyIterator& p) noexcept
  {
    return {cuda::std::ranges::iter_move(p.base_)};
  }

  // Specialization of iter_swap
  // Note cuda::std::swap(*x, *y) would fail to compile as operator* returns prvalues
  // and cuda::std::swap takes non-const lvalue references
  __host__ __device__ friend constexpr void iter_swap(const ProxyIterator& x, const ProxyIterator& y) noexcept
  {
    cuda::std::ranges::iter_swap(x.base_, y.base_);
  }

  // to satisfy input_iterator
  __host__ __device__ constexpr Proxy<cuda::std::iter_reference_t<Base>> operator*() const
  {
    return {*base_};
  }

  __host__ __device__ constexpr ProxyIterator& operator++()
  {
    ++base_;
    return *this;
  }

  __host__ __device__ constexpr void operator++(int)
  {
    ++*this;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::equality_comparable<B2>)
  __host__ __device__ friend constexpr bool operator==(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ == y.base_;
  }

  // to satisfy forward_iterator
  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::forward_iterator<B2>)
  __host__ __device__ constexpr ProxyIterator operator++(int)
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  // to satisfy bidirectional_iterator
  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::bidirectional_iterator<B2>)
  __host__ __device__ constexpr ProxyIterator& operator--()
  {
    --base_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::bidirectional_iterator<B2>)
  __host__ __device__ constexpr ProxyIterator operator--(int)
  {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  // to satisfy random_access_iterator
  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ constexpr ProxyIterator& operator+=(difference_type n)
  {
    base_ += n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ constexpr ProxyIterator& operator-=(difference_type n)
  {
    base_ -= n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ constexpr Proxy<cuda::std::iter_reference_t<Base>> operator[](difference_type n) const
  {
    return {base_[n]};
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr bool operator<(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ < y.base_;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr bool operator>(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ > y.base_;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr bool operator<=(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ <= y.base_;
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr bool operator>=(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ >= y.base_;
  }

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>&& cuda::std::three_way_comparable<B2>)
  __host__ __device__ friend constexpr auto operator<=>(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ <=> y.base_;
  }
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr ProxyIterator operator+(const ProxyIterator& x, difference_type n)
  {
    return ProxyIterator{x.base_ + n};
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr ProxyIterator operator+(difference_type n, const ProxyIterator& x)
  {
    return ProxyIterator{n + x.base_};
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr ProxyIterator operator-(const ProxyIterator& x, difference_type n)
  {
    return ProxyIterator{x.base_ - n};
  }

  _LIBCUDACXX_TEMPLATE(class B2 = Base)
  _LIBCUDACXX_REQUIRES(cuda::std::random_access_iterator<B2>)
  __host__ __device__ friend constexpr difference_type operator-(const ProxyIterator& x, const ProxyIterator& y)
  {
    return x.base_ - y.base_;
  }
};

static_assert(cuda::std::indirectly_readable<ProxyIterator<int*>>, "");
static_assert(cuda::std::indirectly_writable<ProxyIterator<int*>, Proxy<int>>, "");
static_assert(cuda::std::indirectly_writable<ProxyIterator<int*>, Proxy<int&>>, "");

template <class BaseSent>
struct ProxySentinel
{
  BaseSent base_;

  ProxySentinel() = default;
  __host__ __device__ constexpr ProxySentinel(BaseSent base)
      : base_{cuda::std::move(base)}
  {}

  _LIBCUDACXX_TEMPLATE(class Base)
  _LIBCUDACXX_REQUIRES(cuda::std::equality_comparable_with<Base, BaseSent>)
  __host__ __device__ friend constexpr bool operator==(const ProxyIterator<Base>& p, const ProxySentinel& sent)
  {
    return p.base_ == sent.base_;
  }
};

#  if !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES)
template <cuda::std::ranges::input_range Base>
  requires cuda::std::ranges::view<Base>
struct ProxyRange
{
  Base base_;

  __host__ __device__ constexpr auto begin()
  {
    return ProxyIterator{cuda::std::ranges::begin(base_)};
  }

  __host__ __device__ constexpr auto end()
  {
    return ProxySentinel{cuda::std::ranges::end(base_)};
  }

  __host__ __device__ constexpr auto begin() const
    requires cuda::std::ranges::input_range<const Base>
  {
    return ProxyIterator{cuda::std::ranges::begin(base_)};
  }

  __host__ __device__ constexpr auto end() const
    requires cuda::std::ranges::input_range<const Base>
  {
    return ProxySentinel{cuda::std::ranges::end(base_)};
  }
};

template <cuda::std::ranges::input_range R>
  requires cuda::std::ranges::viewable_range<R&&>
ProxyRange(R&&) -> ProxyRange<cuda::std::views::all_t<R&&>>;
#  endif // !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES)

#endif // TEST_STD_VER > 2014

namespace types
{
template <class Ptr>
using random_access_iterator_list =
  type_list<Ptr,
#if TEST_STD_VER >= 2017
            contiguous_iterator<Ptr>,
#endif
            random_access_iterator<Ptr>>;

template <class Ptr>
using bidirectional_iterator_list =
  concatenate_t<random_access_iterator_list<Ptr>, type_list<bidirectional_iterator<Ptr>>>;

template <class Ptr>
using forward_iterator_list = concatenate_t<bidirectional_iterator_list<Ptr>, type_list<forward_iterator<Ptr>>>;

template <class Ptr>
using cpp17_input_iterator_list = concatenate_t<forward_iterator_list<Ptr>, type_list<cpp17_input_iterator<Ptr>>>;

#if TEST_STD_VER >= 2020
template <class Ptr>
using cpp20_input_iterator_list =
  concatenate_t<forward_iterator_list<Ptr>, type_list<cpp20_input_iterator<Ptr>, cpp17_input_iterator<Ptr>>>;
#endif
} // namespace types

#endif // SUPPORT_TEST_ITERATORS_H
