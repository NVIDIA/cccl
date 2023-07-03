//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_TYPES_H

#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

#include "test_macros.h"
#include "test_iterators.h"

STATIC_TEST_GLOBAL_VAR int globalBuffer[4][4] = {
    {1111, 2222, 3333, 4444},
    {555, 666, 777, 888},
    {99, 1010, 1111, 1212},
    {13, 14, 15, 16},
};

struct ChildView : cuda::std::ranges::view_base {
  int* ptr_;

  using iterator = cpp20_input_iterator<int*>;
  using const_iterator = cpp20_input_iterator<const int*>;
  using sentinel = sentinel_wrapper<iterator>;
  using const_sentinel = sentinel_wrapper<const_iterator>;

  __host__ __device__ constexpr ChildView(int* ptr = globalBuffer[0]) : ptr_(ptr) {}
  ChildView(const ChildView&) = delete;
  ChildView(ChildView&&) = default;
  ChildView& operator=(const ChildView&) = delete;
  ChildView& operator=(ChildView&&) = default;

  __host__ __device__ constexpr iterator begin() { return iterator(ptr_); }
  __host__ __device__ constexpr const_iterator begin() const { return const_iterator(ptr_); }
  __host__ __device__ constexpr sentinel end() { return sentinel(iterator(ptr_ + 4)); }
  __host__ __device__ constexpr const_sentinel end() const { return const_sentinel(const_iterator(ptr_ + 4)); }
};

STATIC_TEST_GLOBAL_VAR ChildView globalChildren[4] = {
    ChildView(globalBuffer[0]),
    ChildView(globalBuffer[1]),
    ChildView(globalBuffer[2]),
    ChildView(globalBuffer[3]),
};

template <class T>
struct ParentView : cuda::std::ranges::view_base {
  T* ptr_;
  unsigned size_;

  using iterator = cpp20_input_iterator<T*>;
  using const_iterator = cpp20_input_iterator<const T*>;
  using sentinel = sentinel_wrapper<iterator>;
  using const_sentinel = sentinel_wrapper<const_iterator>;

  __host__ __device__ constexpr ParentView(T* ptr, unsigned size = 4) : ptr_(ptr), size_(size) {}
  template<class T2 = T, cuda::std::enable_if_t<cuda::std::same_as<ChildView, T2>, int> = 0>
  __host__ __device__ constexpr ParentView(ChildView* ptr = globalChildren, unsigned size = 4)
  : ptr_(ptr), size_(size) {}
  ParentView(const ParentView&) = delete;
  ParentView(ParentView&&) = default;
  ParentView& operator=(const ParentView&) = delete;
  ParentView& operator=(ParentView&&) = default;

  __host__ __device__ constexpr iterator begin() { return iterator(ptr_); }
  __host__ __device__ constexpr const_iterator begin() const { return const_iterator(ptr_); }
  __host__ __device__ constexpr sentinel end() { return sentinel(iterator(ptr_ + size_)); }
  __host__ __device__ constexpr const_sentinel end() const { return const_sentinel(const_iterator(ptr_ + size_)); }
};

template <class T>
ParentView(T*) -> ParentView<T>;

struct CopyableChild : cuda::std::ranges::view_base {
  int* ptr_;
  unsigned size_;

  using iterator = cpp17_input_iterator<int*>;
  using const_iterator = cpp17_input_iterator<const int*>;
  using sentinel = sentinel_wrapper<iterator>;
  using const_sentinel = sentinel_wrapper<const_iterator>;

  __host__ __device__ constexpr CopyableChild(int* ptr = globalBuffer[0], unsigned size = 4) : ptr_(ptr), size_(size) {}

  __host__ __device__ constexpr iterator begin() { return iterator(ptr_); }
  __host__ __device__ constexpr const_iterator begin() const { return const_iterator(ptr_); }
  __host__ __device__ constexpr sentinel end() { return sentinel(iterator(ptr_ + size_)); }
  __host__ __device__ constexpr const_sentinel end() const { return const_sentinel(const_iterator(ptr_ + size_)); }
};

struct CopyableParent : cuda::std::ranges::view_base {
  CopyableChild* ptr_;

  using iterator = cpp17_input_iterator<CopyableChild*>;
  using const_iterator = cpp17_input_iterator<const CopyableChild*>;
  using sentinel = sentinel_wrapper<iterator>;
  using const_sentinel = sentinel_wrapper<const_iterator>;

  __host__ __device__ constexpr CopyableParent(CopyableChild* ptr) : ptr_(ptr) {}

  __host__ __device__ constexpr iterator begin() { return iterator(ptr_); }
  __host__ __device__ constexpr const_iterator begin() const { return const_iterator(ptr_); }
  __host__ __device__ constexpr sentinel end() { return sentinel(iterator(ptr_ + 4)); }
  __host__ __device__ constexpr const_sentinel end() const { return const_sentinel(const_iterator(ptr_ + 4)); }
};

struct Box {
  int x;
};

template <class T>
struct InputValueIter {
  typedef cuda::std::input_iterator_tag iterator_category;
  typedef T value_type;
  typedef int difference_type;
  typedef T reference;

  T* ptr_ = nullptr;
  constexpr InputValueIter() = default;
  __host__ __device__ constexpr InputValueIter(T* ptr) : ptr_(ptr) {}

  __host__ __device__ constexpr T operator*() const { return cuda::std::move(*ptr_); }
  __host__ __device__ constexpr void operator++(int) { ++ptr_; }
  __host__ __device__ constexpr InputValueIter& operator++() {
    ++ptr_;
    return *this;
  }

  __host__ __device__ constexpr T* operator->() { return ptr_; }

#if TEST_STD_VER > 17
  __host__ __device__  constexpr friend bool operator==(const InputValueIter&, const InputValueIter&) = default;
#else
  __host__ __device__ constexpr friend bool operator==(const InputValueIter& lhs, const InputValueIter& rhs) { return lhs.ptr_ == rhs.ptr_;}
  __host__ __device__ constexpr friend bool operator!=(const InputValueIter& lhs, const InputValueIter& rhs) { return lhs.ptr_ != rhs.ptr_;}
#endif
};

template <class T>
struct ValueView : cuda::std::ranges::view_base {
  InputValueIter<T> ptr_;

  using sentinel = sentinel_wrapper<InputValueIter<T>>;

  __host__ __device__ constexpr ValueView(T* ptr) : ptr_(ptr) {}

  __host__ __device__ constexpr ValueView(ValueView&& other) : ptr_(other.ptr_) { other.ptr_.ptr_ = nullptr; }

  __host__ __device__ constexpr ValueView& operator=(ValueView&& other) {
    ptr_ = other.ptr_;
    other.ptr_ = InputValueIter<T>(nullptr);
    return *this;
  }

  ValueView(const ValueView&) = delete;
  ValueView& operator=(const ValueView&) = delete;

  __host__ __device__ constexpr InputValueIter<T> begin() const { return ptr_; }
  __host__ __device__ constexpr sentinel end() const { return sentinel(InputValueIter<T>(ptr_.ptr_ + 4)); }
};

template <class Iter, class Sent = Iter, class NonConstIter = Iter, class NonConstSent = Sent>
struct BufferView : cuda::std::ranges::view_base {

  using T = cuda::std::iter_value_t<Iter>;
  T* data_;
  cuda::std::size_t size_;

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferView(T (&b)[N]) : data_(b), size_(N) {}
  __host__ __device__ constexpr BufferView(T* p, cuda::std::size_t s) : data_(p), size_(s) {}

  template<class Iter2 = Iter, cuda::std::enable_if_t<!cuda::std::is_same_v<Iter2, NonConstIter>, int> = 0>
  __host__ __device__ constexpr NonConstIter begin() {
    return NonConstIter(this->data_);
  }
  __host__ __device__ constexpr Iter begin() const { return Iter(this->data_); }

  template<class Sent2 = Sent, cuda::std::enable_if_t<!cuda::std::is_same_v<Sent2, NonConstSent>, int> = 0>
  __host__ __device__ constexpr NonConstSent end() {
    if constexpr (cuda::std::is_same_v<NonConstIter, NonConstSent>) {
      return NonConstIter(this->data_ + this->size_);
    } else {
      return NonConstSent(NonConstIter(this->data_ + this->size_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  __host__ __device__ constexpr Sent end() const {
    if constexpr (cuda::std::is_same_v<Iter, Sent>) {
      return Iter(this->data_ + this->size_);
    } else {
      return Sent(Iter(this->data_ + this->size_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }
};

// an `input_iterator` that can be used in a `common_range`
template <class Base>
struct common_input_iterator {
  Base it_;

  using value_type = cuda::std::iter_value_t<Base>;
  using difference_type = cuda::std::intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  constexpr common_input_iterator() = default;
  __host__ __device__ constexpr explicit common_input_iterator(Base it) : it_(it) {}

  __host__ __device__ constexpr common_input_iterator& operator++() {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) { ++it_; }

  __host__ __device__ constexpr cuda::std::iter_reference_t<Base> operator*() const { return *it_; }

#if TEST_STD_VER > 17
  __host__ __device__ friend constexpr bool operator==(common_input_iterator const&, common_input_iterator const&) = default;
#else
  __host__ __device__ constexpr friend bool operator==(const common_input_iterator& lhs, const common_input_iterator& rhs) { return lhs.it_ == rhs.it_;}
  __host__ __device__ constexpr friend bool operator!=(const common_input_iterator& lhs, const common_input_iterator& rhs) { return lhs.it_ != rhs.it_;}
#endif
};

using InputCommonInner = BufferView<common_input_iterator<int*>>;
static_assert(cuda::std::ranges::input_range<InputCommonInner>);
static_assert(!cuda::std::ranges::forward_range<InputCommonInner>);
static_assert(cuda::std::ranges::common_range<InputCommonInner>);

using InputNonCommonInner = BufferView<common_input_iterator<int*>, sentinel_wrapper<common_input_iterator<int*>>>;
static_assert(cuda::std::ranges::input_range<InputNonCommonInner>);
static_assert(!cuda::std::ranges::forward_range<InputNonCommonInner>);
static_assert(!cuda::std::ranges::common_range<InputNonCommonInner>);

using ForwardCommonInner = BufferView<forward_iterator<int*>>;
static_assert(cuda::std::ranges::forward_range<ForwardCommonInner>);
static_assert(!cuda::std::ranges::bidirectional_range<ForwardCommonInner>);
static_assert(cuda::std::ranges::common_range<ForwardCommonInner>);

using ForwardNonCommonInner = BufferView<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>;
static_assert(cuda::std::ranges::forward_range<ForwardNonCommonInner>);
static_assert(!cuda::std::ranges::bidirectional_range<ForwardNonCommonInner>);
static_assert(!cuda::std::ranges::common_range<ForwardNonCommonInner>);

using BidiCommonInner = BufferView<bidirectional_iterator<int*>>;
static_assert(cuda::std::ranges::bidirectional_range<BidiCommonInner>);
static_assert(cuda::std::ranges::common_range<BidiCommonInner>);

using BidiNonCommonInner = BufferView<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>;
static_assert(cuda::std::ranges::bidirectional_range<BidiNonCommonInner>);
static_assert(!cuda::std::ranges::common_range<BidiNonCommonInner>);

template <class Inner = BufferView<int*>>
using SimpleInputCommonOuter = BufferView<common_input_iterator<Inner*>>;
static_assert(!cuda::std::ranges::forward_range<SimpleInputCommonOuter<>>);
static_assert(!cuda::std::ranges::bidirectional_range<SimpleInputCommonOuter<>>);
static_assert(cuda::std::ranges::common_range<SimpleInputCommonOuter<>>);
LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<SimpleInputCommonOuter<>>);

template <class Inner = BufferView<int*>>
using NonSimpleInputCommonOuter = BufferView<common_input_iterator<const Inner*>, common_input_iterator<const Inner*>,
                                             common_input_iterator< Inner*>, common_input_iterator< Inner*>>;
static_assert(!cuda::std::ranges::forward_range<NonSimpleInputCommonOuter<>>);
static_assert(!cuda::std::ranges::bidirectional_range<NonSimpleInputCommonOuter<>>);
static_assert(cuda::std::ranges::common_range<NonSimpleInputCommonOuter<>>);
LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<NonSimpleInputCommonOuter<>>);

template <class Inner = BufferView<int*>>
using SimpleForwardCommonOuter = BufferView<forward_iterator<Inner*>>;
static_assert(cuda::std::ranges::forward_range<SimpleForwardCommonOuter<>>);
static_assert(!cuda::std::ranges::bidirectional_range<SimpleForwardCommonOuter<>>);
static_assert(cuda::std::ranges::common_range<SimpleForwardCommonOuter<>>);
LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<SimpleForwardCommonOuter<>>);

template <class Inner = BufferView<int*>>
using NonSimpleForwardCommonOuter = BufferView<forward_iterator<const Inner*>, forward_iterator<const Inner*>,
                                               forward_iterator<Inner*>, forward_iterator<Inner*>>;
static_assert(cuda::std::ranges::forward_range<NonSimpleForwardCommonOuter<>>);
static_assert(!cuda::std::ranges::bidirectional_range<NonSimpleForwardCommonOuter<>>);
static_assert(cuda::std::ranges::common_range<NonSimpleForwardCommonOuter<>>);
LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<NonSimpleForwardCommonOuter<>>);

template <class Inner = BufferView<int*>>
using SimpleForwardNonCommonOuter = BufferView<forward_iterator<Inner*>, sentinel_wrapper<forward_iterator<Inner*>>>;
static_assert(cuda::std::ranges::forward_range<SimpleForwardNonCommonOuter<>>);
static_assert(!cuda::std::ranges::bidirectional_range<SimpleForwardNonCommonOuter<>>);
static_assert(!cuda::std::ranges::common_range<SimpleForwardNonCommonOuter<>>);
LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<SimpleForwardNonCommonOuter<>>);

template <class Inner = BufferView<int*>>
using NonSimpleForwardNonCommonOuter =
    BufferView<forward_iterator<const Inner*>, sentinel_wrapper<forward_iterator<const Inner*>>,
               forward_iterator<Inner*>, sentinel_wrapper<forward_iterator<Inner*>>>;
static_assert(cuda::std::ranges::forward_range<NonSimpleForwardNonCommonOuter<>>);
static_assert(!cuda::std::ranges::bidirectional_range<NonSimpleForwardNonCommonOuter<>>);
static_assert(!cuda::std::ranges::common_range<NonSimpleForwardNonCommonOuter<>>);
LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<NonSimpleForwardNonCommonOuter<>>);

template <class Inner = BufferView<int*>>
using BidiCommonOuter = BufferView<bidirectional_iterator<Inner*>>;
static_assert(cuda::std::ranges::bidirectional_range<BidiCommonOuter<>>);
static_assert(cuda::std::ranges::common_range<BidiCommonOuter<>>);
LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<BidiCommonOuter<>>);

// an iterator where its operator* makes a copy of underlying operator*
template <class It>
struct copying_iterator {
  It it_ = It();

  using value_type = typename cuda::std::iterator_traits<It>::value_type;
  using difference_type = typename cuda::std::iterator_traits<It>::difference_type;
  using pointer = typename cuda::std::iterator_traits<It>::pointer;

  template<class It2 = It, cuda::std::enable_if_t<cuda::std::default_initializable<It2>, int> = 0>
  __host__ __device__ constexpr copying_iterator() noexcept(cuda::std::is_nothrow_default_constructible_v<It2>) {}
  __host__ __device__ constexpr copying_iterator(It it) : it_(cuda::std::move(it)) {}

  // makes a copy of underlying operator* to create a PRValue
  __host__ __device__ constexpr value_type operator*() const { return *it_; }

  __host__ __device__ constexpr copying_iterator& operator++() {
    ++it_;
    return *this;
  }

  template<class It2 = It, cuda::std::enable_if_t<!cuda::std::forward_iterator<It2>, int> = 0>
  __host__ __device__ constexpr void operator++(int) { return it_++; }

  template<class It2 = It, cuda::std::enable_if_t<cuda::std::forward_iterator<It2>, int> = 0>
  __host__ __device__ constexpr copying_iterator operator++(int) {
    return copying_iterator(it_++);
  }

  template<class It2 = It, cuda::std::enable_if_t<cuda::std::bidirectional_iterator<It2>, int> = 0>
  __host__ __device__ constexpr copying_iterator& operator--() {
    --it_;
    return *this;
  }
  template<class It2 = It, cuda::std::enable_if_t<cuda::std::bidirectional_iterator<It2>, int> = 0>
  __host__ __device__ constexpr copying_iterator operator--(int) {
    return copying_iterator(it_--);
  }

#if TEST_STD_VER > 17
  __host__ __device__ friend constexpr bool operator==(copying_iterator const&, copying_iterator const&) = default;
#else
  __host__ __device__ constexpr friend bool operator==(const copying_iterator& lhs, const copying_iterator& rhs) { return lhs.it_ == rhs.it_;}
  __host__ __device__ constexpr friend bool operator!=(const copying_iterator& lhs, const copying_iterator& rhs) { return lhs.it_ != rhs.it_;}
#endif
};

template <class Outer>
struct InnerRValue : Outer {

  using iterator = copying_iterator<cuda::std::ranges::iterator_t<Outer>>;
  using const_iterator = copying_iterator<cuda::std::ranges::iterator_t<const Outer>>;
  using sentinel = copying_iterator<cuda::std::ranges::sentinel_t<Outer>>;
  using const_sentinel = copying_iterator<cuda::std::ranges::sentinel_t<const Outer>>;

#if defined(_LIBCUDACXX_COMPILER_NVRTC)
    constexpr InnerRValue() noexcept = default;

    template<class... _Args>
    __host__ __device__ constexpr
    InnerRValue(_Args&&... __args)
      noexcept(noexcept(Outer(cuda::std::declval<_Args>()...)))
      : Outer(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
#else
    using Outer::Outer;
#endif

  static_assert(cuda::std::ranges::common_range<Outer>, "non-common range is not supported yet");

  __host__ __device__ constexpr iterator begin() { return Outer::begin(); }
  template<class Outer2 = Outer, cuda::std::enable_if_t<cuda::std::ranges::range<const Outer2>, int> = 0>
  __host__ __device__ constexpr const_iterator begin() const {
    return Outer::begin();
  }

  __host__ __device__ constexpr auto end() { return iterator{Outer::end()}; }
  template<class Outer2 = Outer, cuda::std::enable_if_t<cuda::std::ranges::range<const Outer2>, int> = 0>
  __host__ __device__ constexpr auto end() const {
    return const_iterator{Outer::end()};
  }
};
static_assert(cuda::std::ranges::forward_range<InnerRValue<SimpleForwardCommonOuter<>>>);
static_assert(!cuda::std::ranges::bidirectional_range<InnerRValue<SimpleForwardCommonOuter<>>>);
static_assert(cuda::std::ranges::common_range<InnerRValue<SimpleForwardCommonOuter<>>>);
LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<InnerRValue<SimpleForwardCommonOuter<>>>);
static_assert(!cuda::std::is_lvalue_reference_v<cuda::std::ranges::range_reference_t<InnerRValue<SimpleForwardCommonOuter<>>>>);

struct move_swap_aware_iter {

  // This is a proxy-like iterator where `reference` is a prvalue, and
  // `reference` and `value_type` are distinct types (similar to `zip_view::iterator`).
  using value_type = cuda::std::pair<int, int>;
  using reference = cuda::std::pair<int&, int&>;
  using rvalue_reference = cuda::std::pair<int&&, int&&>;

  using difference_type = cuda::std::intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  int* iter_move_called = nullptr;
  int* iter_swap_called = nullptr;
  int* i_ = nullptr;

  __host__ __device__ constexpr move_swap_aware_iter& operator++() {
    ++i_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) { ++i_; }

  __host__ __device__ constexpr reference operator*() const { return reference(*i_, *i_); }
  __host__ __device__ constexpr friend bool operator==(const move_swap_aware_iter& x, const move_swap_aware_iter& y) {
    return x.i_ == y.i_;
  }
#if TEST_STD_VER < 20
  __host__ __device__ constexpr friend bool operator!=(const move_swap_aware_iter& x, const move_swap_aware_iter& y) {
    return x.i_ != y.i_;
  }
#endif

  __host__ __device__ constexpr friend rvalue_reference iter_move(const move_swap_aware_iter& x) noexcept {
    ++(*x.iter_move_called);
    return rvalue_reference{cuda::std::move(*x.i_), cuda::std::move(*x.i_)};
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 friend void iter_swap(const move_swap_aware_iter& x, const move_swap_aware_iter& y) noexcept {
    ++(*x.iter_swap_called);
    cuda::std::swap(*x.i_, *y.i_);
  }
};

struct IterMoveSwapAwareView : BufferView<int*> {
#if defined(_LIBCUDACXX_COMPILER_NVRTC)
  IterMoveSwapAwareView() noexcept = default;

  template<class T>
  __host__ __device__ constexpr
  IterMoveSwapAwareView(T&& arr)
    noexcept(noexcept(BufferView<int*>(cuda::std::declval<T>())))
    : BufferView<int*>(_CUDA_VSTD::forward<T>(arr))
  {}
#else
  using BufferView<int*>::BufferView;
#endif

  int iter_move_called = 0;
  int iter_swap_called = 0;

  __host__ __device__ constexpr auto begin() { return move_swap_aware_iter{&iter_move_called, &iter_swap_called, data_}; }

  __host__ __device__ constexpr auto end() { return move_swap_aware_iter{&iter_move_called, &iter_swap_called, data_ + size_}; }
};
static_assert(cuda::std::ranges::input_range<IterMoveSwapAwareView>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_TYPES_H
