//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TYPES_H

#include <cuda/std/functional>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

template <class T>
struct BufferView : cuda::std::ranges::view_base
{
  T* buffer_;
  cuda::std::size_t size_;

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferView(T (&b)[N])
      : buffer_(b)
      , size_(N)
  {}
};
using IntBufferView = BufferView<int>;

#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
#  define DELEGATE_INTBUFFERVIEW(Derived, Base)       \
    template <cuda::std::size_t N>                    \
    __host__ __device__ constexpr Derived(int(&b)[N]) \
        : Base(b)                                     \
    {}
#else // ^^^ TEST_COMPILER_NVRTC ^^^ / vvv !TEST_COMPILER_NVRTC vvv
#  define DELEGATE_INTBUFFERVIEW(Derived, Base) using Base::Base;
#endif // !TEST_COMPILER_NVRTC

template <bool Simple>
struct Common : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(Common, IntBufferView)

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr int* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr const int* begin() const
  {
    return buffer_;
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr int* end()
  {
    return buffer_ + size_;
  }
  __host__ __device__ constexpr const int* end() const
  {
    return buffer_ + size_;
  }
};
using SimpleCommon    = Common<true>;
using NonSimpleCommon = Common<false>;

using SimpleCommonRandomAccessSized    = SimpleCommon;
using NonSimpleCommonRandomAccessSized = NonSimpleCommon;

static_assert(cuda::std::ranges::common_range<Common<true>>);
static_assert(cuda::std::ranges::random_access_range<SimpleCommon>);
static_assert(cuda::std::ranges::sized_range<SimpleCommon>);
static_assert(simple_view<SimpleCommon>);
static_assert(!simple_view<NonSimpleCommon>);

template <bool Simple>
struct CommonNonRandom : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(CommonNonRandom, IntBufferView)

  using const_iterator = forward_iterator<const int*>;
  using iterator       = forward_iterator<int*>;
  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr iterator begin()
  {
    return iterator(buffer_);
  }
  __host__ __device__ constexpr const_iterator begin() const
  {
    return const_iterator(buffer_);
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr iterator end()
  {
    return iterator(buffer_ + size_);
  }
  __host__ __device__ constexpr const_iterator end() const
  {
    return const_iterator(buffer_ + size_);
  }
};

using SimpleCommonNonRandom    = CommonNonRandom<true>;
using NonSimpleCommonNonRandom = CommonNonRandom<false>;

static_assert(cuda::std::ranges::common_range<SimpleCommonNonRandom>);
static_assert(!cuda::std::ranges::random_access_range<SimpleCommonNonRandom>);
static_assert(!cuda::std::ranges::sized_range<SimpleCommonNonRandom>);
static_assert(simple_view<SimpleCommonNonRandom>);
static_assert(!simple_view<NonSimpleCommonNonRandom>);

template <bool Simple>
struct NonCommon : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(NonCommon, IntBufferView)

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr int* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr const int* begin() const
  {
    return buffer_;
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr sentinel_wrapper<int*> end()
  {
    return sentinel_wrapper<int*>(buffer_ + size_);
  }
  __host__ __device__ constexpr sentinel_wrapper<const int*> end() const
  {
    return sentinel_wrapper<const int*>(buffer_ + size_);
  }
};

using SimpleNonCommon    = NonCommon<true>;
using NonSimpleNonCommon = NonCommon<false>;

static_assert(!cuda::std::ranges::common_range<SimpleNonCommon>);
static_assert(cuda::std::ranges::random_access_range<SimpleNonCommon>);
static_assert(!cuda::std::ranges::sized_range<SimpleNonCommon>);
static_assert(simple_view<SimpleNonCommon>);
static_assert(!simple_view<NonSimpleNonCommon>);

template <bool Simple>
struct NonCommonSized : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(NonCommonSized, IntBufferView)

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr int* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr const int* begin() const
  {
    return buffer_;
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr sentinel_wrapper<int*> end()
  {
    return sentinel_wrapper<int*>(buffer_ + size_);
  }
  __host__ __device__ constexpr sentinel_wrapper<const int*> end() const
  {
    return sentinel_wrapper<const int*>(buffer_ + size_);
  }
  __host__ __device__ constexpr cuda::std::size_t size() const
  {
    return size_;
  }
};

using SimpleNonCommonSized               = NonCommonSized<true>;
using SimpleNonCommonRandomAcessSized    = SimpleNonCommonSized;
using NonSimpleNonCommonSized            = NonCommonSized<false>;
using NonSimpleNonCommonRandomAcessSized = NonSimpleNonCommonSized;

static_assert(!cuda::std::ranges::common_range<SimpleNonCommonSized>);
static_assert(cuda::std::ranges::random_access_range<SimpleNonCommonSized>);
static_assert(cuda::std::ranges::sized_range<SimpleNonCommonSized>);
static_assert(simple_view<SimpleNonCommonSized>);
static_assert(!simple_view<NonSimpleNonCommonSized>);

template <bool Simple>
struct NonCommonNonRandom : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(NonCommonNonRandom, IntBufferView)

  using const_iterator = forward_iterator<const int*>;
  using iterator       = forward_iterator<int*>;

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr iterator begin()
  {
    return iterator(buffer_);
  }
  __host__ __device__ constexpr const_iterator begin() const
  {
    return const_iterator(buffer_);
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr sentinel_wrapper<iterator> end()
  {
    return sentinel_wrapper<iterator>(iterator(buffer_ + size_));
  }
  __host__ __device__ constexpr sentinel_wrapper<const_iterator> end() const
  {
    return sentinel_wrapper<const_iterator>(const_iterator(buffer_ + size_));
  }
};

using SimpleNonCommonNonRandom    = NonCommonNonRandom<true>;
using NonSimpleNonCommonNonRandom = NonCommonNonRandom<false>;

static_assert(!cuda::std::ranges::common_range<SimpleNonCommonNonRandom>);
static_assert(!cuda::std::ranges::random_access_range<SimpleNonCommonNonRandom>);
static_assert(!cuda::std::ranges::sized_range<SimpleNonCommonNonRandom>);
static_assert(simple_view<SimpleNonCommonNonRandom>);
static_assert(!simple_view<NonSimpleNonCommonNonRandom>);

template <class Iter, class Sent = Iter, class NonConstIter = Iter, class NonConstSent = Sent>
struct BasicView : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(BasicView, IntBufferView)

  template <class Iter2 = Iter, cuda::std::enable_if_t<!cuda::std::is_same_v<Iter2, NonConstIter>, int> = 0>
  __host__ __device__ constexpr NonConstIter begin()
  {
    return NonConstIter(buffer_);
  }
  __host__ __device__ constexpr Iter begin() const
  {
    return Iter(buffer_);
  }

  template <class Sent2 = Sent, cuda::std::enable_if_t<!cuda::std::is_same_v<Sent2, NonConstSent>, int> = 0>
  __host__ __device__ constexpr NonConstSent end()
  {
    if constexpr (cuda::std::is_same_v<NonConstIter, NonConstSent>)
    {
      return NonConstIter(buffer_ + size_);
    }
    else
    {
      return NonConstSent(NonConstIter(buffer_ + size_));
    }
    _CCCL_UNREACHABLE();
  }

  __host__ __device__ constexpr Sent end() const
  {
    if constexpr (cuda::std::is_same_v<Iter, Sent>)
    {
      return Iter(buffer_ + size_);
    }
    else
    {
      return Sent(Iter(buffer_ + size_));
    }
    _CCCL_UNREACHABLE();
  }
};

template <class Base = int*>
struct forward_sized_iterator
{
  Base it_ = nullptr;

  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;
  using pointer           = Base;
  using reference         = decltype(*Base{});

  forward_sized_iterator() = default;
  __host__ __device__ constexpr forward_sized_iterator(Base it)
      : it_(it)
  {}

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr forward_sized_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr forward_sized_iterator operator++(int)
  {
    return forward_sized_iterator(it_++);
  }

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool
  operator==(const forward_sized_iterator&, const forward_sized_iterator&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool operator!=(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ != y.it_;
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ friend constexpr difference_type
  operator-(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ - y.it_;
  }
};
static_assert(cuda::std::forward_iterator<forward_sized_iterator<>>);
static_assert(cuda::std::sized_sentinel_for<forward_sized_iterator<>, forward_sized_iterator<>>);

using ForwardSizedView = BasicView<forward_sized_iterator<>>;
static_assert(cuda::std::ranges::forward_range<ForwardSizedView>);
static_assert(cuda::std::ranges::sized_range<ForwardSizedView>);
static_assert(cuda::std::ranges::common_range<ForwardSizedView>);
static_assert(!cuda::std::ranges::random_access_range<ForwardSizedView>);
static_assert(simple_view<ForwardSizedView>);

using NonSimpleForwardSizedView =
  BasicView<forward_sized_iterator<const int*>,
            forward_sized_iterator<const int*>,
            forward_sized_iterator<int*>,
            forward_sized_iterator<int*>>;
static_assert(cuda::std::ranges::forward_range<NonSimpleForwardSizedView>);
static_assert(cuda::std::ranges::sized_range<NonSimpleForwardSizedView>);
static_assert(cuda::std::ranges::common_range<NonSimpleForwardSizedView>);
static_assert(!cuda::std::ranges::random_access_range<NonSimpleForwardSizedView>);
static_assert(!simple_view<NonSimpleForwardSizedView>);

using ForwardSizedNonCommon = BasicView<forward_sized_iterator<>, sized_sentinel<forward_sized_iterator<>>>;
static_assert(cuda::std::ranges::forward_range<ForwardSizedNonCommon>);
static_assert(cuda::std::ranges::sized_range<ForwardSizedNonCommon>);
static_assert(!cuda::std::ranges::common_range<ForwardSizedNonCommon>);
static_assert(!cuda::std::ranges::random_access_range<ForwardSizedNonCommon>);
static_assert(simple_view<ForwardSizedNonCommon>);

using NonSimpleForwardSizedNonCommon =
  BasicView<forward_sized_iterator<const int*>,
            sized_sentinel<forward_sized_iterator<const int*>>,
            forward_sized_iterator<int*>,
            sized_sentinel<forward_sized_iterator<int*>>>;
static_assert(cuda::std::ranges::forward_range<NonSimpleForwardSizedNonCommon>);
static_assert(cuda::std::ranges::sized_range<NonSimpleForwardSizedNonCommon>);
static_assert(!cuda::std::ranges::common_range<NonSimpleForwardSizedNonCommon>);
static_assert(!cuda::std::ranges::random_access_range<NonSimpleForwardSizedNonCommon>);
static_assert(!simple_view<NonSimpleForwardSizedNonCommon>);

struct SizedRandomAccessView : IntBufferView
{
  DELEGATE_INTBUFFERVIEW(SizedRandomAccessView, IntBufferView)

  using iterator = random_access_iterator<int*>;

  __host__ __device__ constexpr auto begin() const
  {
    return iterator(buffer_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<iterator>(iterator(buffer_ + size_));
  }

  __host__ __device__ constexpr decltype(auto) operator[](cuda::std::size_t n) const
  {
    return *(begin() + n);
  }
};
static_assert(cuda::std::ranges::view<SizedRandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<SizedRandomAccessView>);
static_assert(cuda::std::ranges::sized_range<SizedRandomAccessView>);

using NonSizedRandomAccessView =
  BasicView<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>;
static_assert(!cuda::std::ranges::contiguous_range<NonSizedRandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<SizedRandomAccessView>);
static_assert(!cuda::std::ranges::common_range<NonSizedRandomAccessView>);
static_assert(!cuda::std::ranges::sized_range<NonSizedRandomAccessView>);
static_assert(simple_view<NonSizedRandomAccessView>);

using NonSimpleNonSizedRandomAccessView =
  BasicView<random_access_iterator<const int*>,
            sentinel_wrapper<random_access_iterator<const int*>>,
            random_access_iterator<int*>,
            sentinel_wrapper<random_access_iterator<int*>>>;
static_assert(!cuda::std::ranges::contiguous_range<NonSimpleNonSizedRandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<NonSimpleNonSizedRandomAccessView>);
static_assert(!cuda::std::ranges::common_range<NonSimpleNonSizedRandomAccessView>);
static_assert(!cuda::std::ranges::sized_range<NonSimpleNonSizedRandomAccessView>);
static_assert(!simple_view<NonSimpleNonSizedRandomAccessView>);

using ContiguousCommonView = BasicView<int*>;
static_assert(cuda::std::ranges::contiguous_range<ContiguousCommonView>);
static_assert(cuda::std::ranges::common_range<ContiguousCommonView>);
static_assert(cuda::std::ranges::sized_range<ContiguousCommonView>);

using ContiguousNonCommonView = BasicView<int*, sentinel_wrapper<int*>>;
static_assert(cuda::std::ranges::contiguous_range<ContiguousNonCommonView>);
static_assert(!cuda::std::ranges::common_range<ContiguousNonCommonView>);
static_assert(!cuda::std::ranges::sized_range<ContiguousNonCommonView>);

using ContiguousNonCommonSized = BasicView<int*, sized_sentinel<int*>>;

static_assert(cuda::std::ranges::contiguous_range<ContiguousNonCommonSized>);
static_assert(!cuda::std::ranges::common_range<ContiguousNonCommonSized>);
static_assert(cuda::std::ranges::sized_range<ContiguousNonCommonSized>);

using InputCommonView = BasicView<common_input_iterator<int*>>;
static_assert(cuda::std::ranges::input_range<InputCommonView>);
static_assert(!cuda::std::ranges::forward_range<InputCommonView>);
static_assert(cuda::std::ranges::common_range<InputCommonView>);
static_assert(simple_view<InputCommonView>);

using NonSimpleInputCommonView =
  BasicView<common_input_iterator<const int*>,
            common_input_iterator<const int*>,
            common_input_iterator<int*>,
            common_input_iterator<int*>>;
static_assert(cuda::std::ranges::input_range<NonSimpleInputCommonView>);
static_assert(!cuda::std::ranges::forward_range<NonSimpleInputCommonView>);
static_assert(cuda::std::ranges::common_range<NonSimpleInputCommonView>);
static_assert(!simple_view<NonSimpleInputCommonView>);

using InputNonCommonView = BasicView<common_input_iterator<int*>, sentinel_wrapper<common_input_iterator<int*>>>;
static_assert(cuda::std::ranges::input_range<InputNonCommonView>);
static_assert(!cuda::std::ranges::forward_range<InputNonCommonView>);
static_assert(!cuda::std::ranges::common_range<InputNonCommonView>);
static_assert(simple_view<InputNonCommonView>);

using NonSimpleInputNonCommonView =
  BasicView<common_input_iterator<const int*>,
            sentinel_wrapper<common_input_iterator<const int*>>,
            common_input_iterator<int*>,
            sentinel_wrapper<common_input_iterator<int*>>>;
static_assert(cuda::std::ranges::input_range<InputNonCommonView>);
static_assert(!cuda::std::ranges::forward_range<InputNonCommonView>);
static_assert(!cuda::std::ranges::common_range<InputNonCommonView>);
static_assert(!simple_view<NonSimpleInputNonCommonView>);

using BidiCommonView = BasicView<bidirectional_iterator<int*>>;
static_assert(!cuda::std::ranges::sized_range<BidiCommonView>);
static_assert(cuda::std::ranges::bidirectional_range<BidiCommonView>);
static_assert(!cuda::std::ranges::random_access_range<BidiCommonView>);
static_assert(cuda::std::ranges::common_range<BidiCommonView>);
static_assert(simple_view<BidiCommonView>);

using NonSimpleBidiCommonView =
  BasicView<bidirectional_iterator<const int*>,
            bidirectional_iterator<const int*>,
            bidirectional_iterator<int*>,
            bidirectional_iterator<int*>>;
static_assert(!cuda::std::ranges::sized_range<NonSimpleBidiCommonView>);
static_assert(cuda::std::ranges::bidirectional_range<NonSimpleBidiCommonView>);
static_assert(!cuda::std::ranges::random_access_range<NonSimpleBidiCommonView>);
static_assert(cuda::std::ranges::common_range<NonSimpleBidiCommonView>);
static_assert(!simple_view<NonSimpleBidiCommonView>);

struct SizedBidiCommon : BidiCommonView
{
  DELEGATE_INTBUFFERVIEW(SizedBidiCommon, BidiCommonView)

  __host__ __device__ cuda::std::size_t size() const
  {
    return base(end()) - base(begin());
  }
};
static_assert(cuda::std::ranges::sized_range<SizedBidiCommon>);
static_assert(cuda::std::ranges::bidirectional_range<SizedBidiCommon>);
static_assert(!cuda::std::ranges::random_access_range<SizedBidiCommon>);
static_assert(cuda::std::ranges::common_range<SizedBidiCommon>);
static_assert(simple_view<SizedBidiCommon>);

struct NonSimpleSizedBidiCommon : NonSimpleBidiCommonView
{
  DELEGATE_INTBUFFERVIEW(NonSimpleSizedBidiCommon, NonSimpleBidiCommonView)
  __host__ __device__ cuda::std::size_t size() const
  {
    return base(end()) - base(begin());
  }
};
static_assert(cuda::std::ranges::sized_range<NonSimpleSizedBidiCommon>);
static_assert(cuda::std::ranges::bidirectional_range<NonSimpleSizedBidiCommon>);
static_assert(!cuda::std::ranges::random_access_range<NonSimpleSizedBidiCommon>);
static_assert(cuda::std::ranges::common_range<NonSimpleSizedBidiCommon>);
static_assert(!simple_view<NonSimpleSizedBidiCommon>);

using BidiNonCommonView = BasicView<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>;
static_assert(!cuda::std::ranges::sized_range<BidiNonCommonView>);
static_assert(cuda::std::ranges::bidirectional_range<BidiNonCommonView>);
static_assert(!cuda::std::ranges::random_access_range<BidiNonCommonView>);
static_assert(!cuda::std::ranges::common_range<BidiNonCommonView>);
static_assert(simple_view<BidiNonCommonView>);

using NonSimpleBidiNonCommonView =
  BasicView<bidirectional_iterator<const int*>,
            sentinel_wrapper<bidirectional_iterator<const int*>>,
            bidirectional_iterator<int*>,
            sentinel_wrapper<bidirectional_iterator<int*>>>;
static_assert(!cuda::std::ranges::sized_range<NonSimpleBidiNonCommonView>);
static_assert(cuda::std::ranges::bidirectional_range<NonSimpleBidiNonCommonView>);
static_assert(!cuda::std::ranges::random_access_range<NonSimpleBidiNonCommonView>);
static_assert(!cuda::std::ranges::common_range<NonSimpleBidiNonCommonView>);
static_assert(!simple_view<NonSimpleBidiNonCommonView>);

using SizedBidiNonCommonView = BasicView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
static_assert(cuda::std::ranges::sized_range<SizedBidiNonCommonView>);
static_assert(cuda::std::ranges::bidirectional_range<SizedBidiNonCommonView>);
static_assert(!cuda::std::ranges::random_access_range<SizedBidiNonCommonView>);
static_assert(!cuda::std::ranges::common_range<SizedBidiNonCommonView>);
static_assert(simple_view<SizedBidiNonCommonView>);

using NonSimpleSizedBidiNonCommonView =
  BasicView<bidirectional_iterator<const int*>,
            sized_sentinel<bidirectional_iterator<const int*>>,
            bidirectional_iterator<int*>,
            sized_sentinel<bidirectional_iterator<int*>>>;
static_assert(cuda::std::ranges::sized_range<NonSimpleSizedBidiNonCommonView>);
static_assert(cuda::std::ranges::bidirectional_range<NonSimpleSizedBidiNonCommonView>);
static_assert(!cuda::std::ranges::random_access_range<NonSimpleSizedBidiNonCommonView>);
static_assert(!cuda::std::ranges::common_range<NonSimpleSizedBidiNonCommonView>);
static_assert(!simple_view<NonSimpleSizedBidiNonCommonView>);

namespace adltest
{
struct iter_move_swap_iterator
{
  cuda::std::reference_wrapper<int> iter_move_called_times;
  cuda::std::reference_wrapper<int> iter_swap_called_times;
  int i = 0;

  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ TEST_CONSTEXPR_CXX20 int operator*() const
  {
    return i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 iter_move_swap_iterator& operator++()
  {
    ++i;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 void operator++(int)
  {
    ++i;
  }

#if TEST_STD_VER >= 2020
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator==(const iter_move_swap_iterator& x, cuda::std::default_sentinel_t)
  {
    return x.i == 5;
  }
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator==(const iter_move_swap_iterator& x, cuda::std::default_sentinel_t)
  {
    return x.i == 5;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator==(cuda::std::default_sentinel_t, const iter_move_swap_iterator& x)
  {
    return x.i == 5;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator!=(const iter_move_swap_iterator& x, cuda::std::default_sentinel_t)
  {
    return x.i != 5;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator!=(cuda::std::default_sentinel_t, const iter_move_swap_iterator& x)
  {
    return x.i != 5;
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ friend TEST_CONSTEXPR_CXX20 int iter_move(iter_move_swap_iterator const& it)
  {
    ++it.iter_move_called_times;
    return it.i;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 void
  iter_swap(iter_move_swap_iterator const& x, iter_move_swap_iterator const& y)
  {
    ++x.iter_swap_called_times;
    ++y.iter_swap_called_times;
  }
};

struct IterMoveSwapRange
{
  int iter_move_called_times = 0;
  int iter_swap_called_times = 0;
  __host__ __device__ TEST_CONSTEXPR_CXX20 auto begin()
  {
    return iter_move_swap_iterator{iter_move_called_times, iter_swap_called_times};
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 auto end() const
  {
    return cuda::std::default_sentinel;
  }
};
} // namespace adltest

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TYPES_H
