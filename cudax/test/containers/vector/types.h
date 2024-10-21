//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_CONTAINER_VECTOR_TYPES_H
#define CUDAX_TEST_CONTAINER_VECTOR_TYPES_H

#include <thrust/equal.h>

#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

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

  __host__ __device__ constexpr explicit cpp17_input_iterator(It it)
      : it_(it)
  {}

  template <class U, class T>
  __host__ __device__ constexpr cpp17_input_iterator(const cpp17_input_iterator<U, T>& u)
      : it_(u.it_)
  {}

  template <class U, class T, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ constexpr cpp17_input_iterator(cpp17_input_iterator<U, T>&& u)
      : it_(u.it_)
  {
    u.it_ = U();
  }

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr cpp17_input_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr cpp17_input_iterator operator++(int)
  {
    return cpp17_input_iterator(it_++);
  }

  __host__ __device__ friend constexpr bool operator==(const cpp17_input_iterator& x, const cpp17_input_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool operator!=(const cpp17_input_iterator& x, const cpp17_input_iterator& y)
  {
    return x.it_ != y.it_;
  }

  __host__ __device__ friend constexpr It base(const cpp17_input_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
static_assert(cuda::std::input_or_output_iterator<cpp17_input_iterator<int*>>, "");
static_assert(cuda::std::indirectly_readable<cpp17_input_iterator<int*>>, "");
static_assert(cuda::std::input_iterator<cpp17_input_iterator<int*>>, "");
static_assert(!thrust::is_indirectly_trivially_relocatable_to<cpp17_input_iterator<int*>, int*>::value, "");

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

  __host__ __device__ constexpr forward_iterator()
      : it_()
  {}
  __host__ __device__ constexpr explicit forward_iterator(It it)
      : it_(it)
  {}

  template <class U>
  __host__ __device__ constexpr forward_iterator(const forward_iterator<U>& u)
      : it_(u.it_)
  {}

  template <class U, class = typename cuda::std::enable_if<cuda::std::is_default_constructible<U>::value>::type>
  __host__ __device__ constexpr forward_iterator(forward_iterator<U>&& other)
      : it_(other.it_)
  {
    other.it_ = U();
  }

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr forward_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr forward_iterator operator++(int)
  {
    return forward_iterator(it_++);
  }

  __host__ __device__ friend constexpr bool operator==(const forward_iterator& x, const forward_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool operator!=(const forward_iterator& x, const forward_iterator& y)
  {
    return x.it_ != y.it_;
  }

  __host__ __device__ friend constexpr It base(const forward_iterator& i)
  {
    return i.it_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
static_assert(cuda::std::forward_iterator<forward_iterator<int*>>, "");
static_assert(!thrust::is_indirectly_trivially_relocatable_to<cpp17_input_iterator<int*>, int*>::value, "");

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
#if TEST_STD_VER < 2020
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
#endif
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
#if TEST_STD_VER < 2020
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
#endif
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

template <class T, size_t Capacity>
struct input_range
{
  cuda::std::array<T, Capacity> data;
  cpp17_input_iterator<T*> end_{data.data() + Capacity};

  __host__ __device__ constexpr cpp17_input_iterator<T*> begin() noexcept
  {
    return cpp17_input_iterator<T*>{data.begin()};
  }

  __host__ __device__ constexpr sentinel_wrapper<cpp17_input_iterator<T*>> end() noexcept
  {
    return sentinel_wrapper<cpp17_input_iterator<T*>>{end_};
  }
};
static_assert(cuda::std::ranges::input_range<input_range<int, 4>>);
static_assert(!cuda::std::ranges::forward_range<input_range<int, 4>>);
static_assert(!cuda::std::ranges::common_range<input_range<int, 4>>);
static_assert(!cuda::std::ranges::sized_range<input_range<int, 4>>);

template <class T, size_t Capacity>
struct uncommon_range
{
  cuda::std::array<T, Capacity> data;
  forward_iterator<T*> end_{data.data() + Capacity};

  __host__ __device__ constexpr forward_iterator<T*> begin() noexcept
  {
    return forward_iterator<T*>{data.begin()};
  }

  __host__ __device__ constexpr sentinel_wrapper<forward_iterator<T*>> end() noexcept
  {
    return sentinel_wrapper<forward_iterator<T*>>{end_};
  }
};
static_assert(cuda::std::ranges::forward_range<uncommon_range<int, 4>>);
static_assert(!cuda::std::ranges::common_range<uncommon_range<int, 4>>);
static_assert(!cuda::std::ranges::sized_range<uncommon_range<int, 4>>);

template <class T, size_t Capacity>
struct sized_uncommon_range
{
  cuda::std::array<T, Capacity> data;
  forward_iterator<T*> end_{data.data() + Capacity};

  __host__ __device__ constexpr forward_iterator<T*> begin() noexcept
  {
    return forward_iterator<T*>{data.begin()};
  }

  __host__ __device__ constexpr sized_sentinel<forward_iterator<T*>> end() noexcept
  {
    return sized_sentinel<forward_iterator<T*>>{end_};
  }
};
static_assert(cuda::std::ranges::forward_range<sized_uncommon_range<int, 4>>);
static_assert(!cuda::std::ranges::common_range<sized_uncommon_range<int, 4>>);
static_assert(cuda::std::ranges::sized_range<sized_uncommon_range<int, 4>>);

#endif // CUDAX_TEST_CONTAINER_VECTOR_TYPES_H
