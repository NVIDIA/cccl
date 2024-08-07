//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_CONTAINER_VECTOR_TYPES_H
#define CUDAX_TEST_CONTAINER_VECTOR_TYPES_H

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <new>

struct Trivial
{
  int val_;

  Trivial() = default;
  __host__ __device__ constexpr Trivial(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ friend constexpr bool operator==(const Trivial& lhs, const Trivial& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend constexpr bool operator<(const Trivial& lhs, const Trivial& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};

struct NonTrivial
{
  int val_;

  __host__ __device__ constexpr NonTrivial() noexcept
      : val_(0)
  {}
  __host__ __device__ constexpr NonTrivial(const int val) noexcept
      : val_(val)
  {}
  __host__ __device__ friend constexpr bool operator==(const NonTrivial& lhs, const NonTrivial& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend constexpr bool operator<(const NonTrivial& lhs, const NonTrivial& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};

struct NonTrivialDestructor
{
  int val_;

  __host__ __device__ NonTrivialDestructor() noexcept
      : val_(0)
  {}
  __host__ __device__ NonTrivialDestructor(const int val) noexcept
      : val_(val)
  {}
  NonTrivialDestructor(const NonTrivialDestructor&)            = default;
  NonTrivialDestructor(NonTrivialDestructor&&)                 = default;
  NonTrivialDestructor& operator=(const NonTrivialDestructor&) = default;
  NonTrivialDestructor& operator=(NonTrivialDestructor&&)      = default;
  __host__ __device__ ~NonTrivialDestructor() noexcept {}
  __host__ __device__ friend bool operator==(const NonTrivialDestructor& lhs, const NonTrivialDestructor& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend bool operator<(const NonTrivialDestructor& lhs, const NonTrivialDestructor& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(!cuda::std::is_trivially_copy_constructible<NonTrivialDestructor>::value, "");
static_assert(!cuda::std::is_trivially_move_constructible<NonTrivialDestructor>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<NonTrivialDestructor>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<NonTrivialDestructor>::value, "");

struct ThrowingDefaultConstruct
{
  int val_;

  __host__ __device__ constexpr ThrowingDefaultConstruct() noexcept(false)
      : val_(0)
  {}
  __host__ __device__ constexpr ThrowingDefaultConstruct(const int val) noexcept
      : val_(val)
  {}
  __host__ __device__ friend constexpr bool
  operator==(const ThrowingDefaultConstruct& lhs, const ThrowingDefaultConstruct& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend constexpr bool
  operator<(const ThrowingDefaultConstruct& lhs, const ThrowingDefaultConstruct& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingDefaultConstruct>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<ThrowingDefaultConstruct>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingDefaultConstruct>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<ThrowingDefaultConstruct>::value, "");
#if !defined(TEST_COMPILER_GCC) || __GNUC__ >= 10
static_assert(!cuda::std::is_nothrow_default_constructible<ThrowingDefaultConstruct>::value, "");
#endif // !TEST_COMPILER_GCC < 10

struct ThrowingCopyConstructor
{
  int val_;

  __host__ __device__ ThrowingCopyConstructor() noexcept
      : val_(0)
  {}
  __host__ __device__ ThrowingCopyConstructor(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ ThrowingCopyConstructor(const ThrowingCopyConstructor& other) noexcept(false)
      : val_(other.val_)
  {}
  ThrowingCopyConstructor(ThrowingCopyConstructor&&)                 = default;
  ThrowingCopyConstructor& operator=(const ThrowingCopyConstructor&) = default;
  ThrowingCopyConstructor& operator=(ThrowingCopyConstructor&&)      = default;

  __host__ __device__ friend bool
  operator==(const ThrowingCopyConstructor& lhs, const ThrowingCopyConstructor& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend bool
  operator<(const ThrowingCopyConstructor& lhs, const ThrowingCopyConstructor& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(!cuda::std::is_trivially_copy_constructible<ThrowingCopyConstructor>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<ThrowingCopyConstructor>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingCopyConstructor>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<ThrowingCopyConstructor>::value, "");
static_assert(!cuda::std::is_nothrow_copy_constructible<ThrowingCopyConstructor>::value, "");

struct ThrowingMoveConstructor
{
  int val_;

  __host__ __device__ ThrowingMoveConstructor() noexcept
      : val_(0)
  {}
  __host__ __device__ ThrowingMoveConstructor(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ ThrowingMoveConstructor(ThrowingMoveConstructor&& other) noexcept(false)
      : val_(other.val_)
  {}
  ThrowingMoveConstructor(const ThrowingMoveConstructor&)            = default;
  ThrowingMoveConstructor& operator=(const ThrowingMoveConstructor&) = default;
  ThrowingMoveConstructor& operator=(ThrowingMoveConstructor&&)      = default;

  __host__ __device__ friend bool
  operator==(const ThrowingMoveConstructor& lhs, const ThrowingMoveConstructor& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend bool
  operator<(const ThrowingMoveConstructor& lhs, const ThrowingMoveConstructor& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingMoveConstructor>::value, "");
static_assert(!cuda::std::is_trivially_move_constructible<ThrowingMoveConstructor>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingMoveConstructor>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<ThrowingMoveConstructor>::value, "");
static_assert(!cuda::std::is_nothrow_move_constructible<ThrowingMoveConstructor>::value, "");

struct ThrowingCopyAssignment
{
  int val_;

  __host__ __device__ ThrowingCopyAssignment() noexcept
      : val_(0)
  {}
  __host__ __device__ ThrowingCopyAssignment(const int val) noexcept
      : val_(val)
  {}

  ThrowingCopyAssignment(const ThrowingCopyAssignment&) = default;
  ThrowingCopyAssignment(ThrowingCopyAssignment&&)      = default;
  __host__ __device__ ThrowingCopyAssignment& operator=(const ThrowingCopyAssignment& other) noexcept(false)
  {
    val_ = other.val_;
    return *this;
  }
  ThrowingCopyAssignment& operator=(ThrowingCopyAssignment&&) = default;

  __host__ __device__ friend bool
  operator==(const ThrowingCopyAssignment& lhs, const ThrowingCopyAssignment& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend bool
  operator<(const ThrowingCopyAssignment& lhs, const ThrowingCopyAssignment& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingCopyAssignment>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<ThrowingCopyAssignment>::value, "");
static_assert(!cuda::std::is_trivially_copy_assignable<ThrowingCopyAssignment>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<ThrowingCopyAssignment>::value, "");
static_assert(!cuda::std::is_nothrow_copy_assignable<ThrowingCopyAssignment>::value, "");

struct ThrowingMoveAssignment
{
  int val_;

  __host__ __device__ ThrowingMoveAssignment() noexcept
      : val_(0)
  {}
  __host__ __device__ ThrowingMoveAssignment(const int val) noexcept
      : val_(val)
  {}

  ThrowingMoveAssignment(ThrowingMoveAssignment&&)                 = default;
  ThrowingMoveAssignment(const ThrowingMoveAssignment&)            = default;
  ThrowingMoveAssignment& operator=(const ThrowingMoveAssignment&) = default;
  __host__ __device__ ThrowingMoveAssignment& operator=(ThrowingMoveAssignment&& other) noexcept(false)
  {
    val_ = other.val_;
    return *this;
  }

  __host__ __device__ friend bool
  operator==(const ThrowingMoveAssignment& lhs, const ThrowingMoveAssignment& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend bool
  operator<(const ThrowingMoveAssignment& lhs, const ThrowingMoveAssignment& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingMoveAssignment>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<ThrowingMoveAssignment>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingMoveAssignment>::value, "");
static_assert(!cuda::std::is_trivially_move_assignable<ThrowingMoveAssignment>::value, "");
static_assert(!cuda::std::is_nothrow_move_assignable<ThrowingMoveAssignment>::value, "");

struct ThrowingSwap
{
  int val_;

  __host__ __device__ ThrowingSwap() noexcept
      : val_(0)
  {}
  __host__ __device__ ThrowingSwap(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ friend bool operator==(const ThrowingSwap& lhs, const ThrowingSwap& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }

  __host__ __device__ void swap(ThrowingSwap& other) noexcept(false)
  {
    cuda::std::swap(val_, other.val_);
  }
};
static_assert(!cuda::std::is_nothrow_swappable<ThrowingMoveConstructor>::value, "");

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

// Helper function to compare two ranges
template <class Range1, class Range2>
__host__ __device__ constexpr bool equal_range(const Range1& range1, const Range2& range2)
{
  return cuda::std::equal(range1.begin(), range1.end(), range2.begin(), range2.end());
}

namespace cudax = cuda::experimental;

struct user_defined_property
{};

template <class T>
struct host_memory_resource
{
  void* allocate(std::size_t size, std::size_t)
  {
    return new T[size];
  }
  void deallocate(void* ptr, std::size_t, std::size_t)
  {
    delete[] reinterpret_cast<T*>(ptr);
  }

  bool operator==(const host_memory_resource&) const
  {
    return true;
  }
  bool operator!=(const host_memory_resource&) const
  {
    return false;
  }

  friend void get_property(const host_memory_resource&, cuda::mr::host_accessible) {}
};
static_assert(cuda::mr::resource<host_memory_resource<int>>, "");
static_assert(cuda::mr::resource_with<host_memory_resource<int>, cuda::mr::host_accessible>, "");

// helper class as we need to pass the properties in a tuple to the catch tests
template <class>
struct extract_properties;

template <class... Properties>
struct extract_properties<cuda::std::tuple<Properties...>>
{
  using vector   = cudax::vector<int, Properties...>;
  using resource = cuda::std::conditional_t<
    cudax::__select_execution_space<Properties...> == cudax::_ExecutionSpace::__host_device,
    cuda::mr::cuda_managed_memory_resource,
    cuda::std::conditional_t<cudax::__select_execution_space<Properties...> == cudax::_ExecutionSpace::__device,
                             cuda::mr::cuda_memory_resource,
                             host_memory_resource<int>>>;

  using resource_ref = cuda::mr::resource_ref<Properties...>;
};

#endif // CUDAX_TEST_CONTAINER_VECTOR_TYPES_H
