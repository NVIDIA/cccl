//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr InnerIter operator->() const
//   requires has-arrow<InnerIter> && copyable<InnerIter>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

template <class T>
inline constexpr bool HasArrow = cuda::std::__has_arrow<T>;

template <class Base>
struct move_only_input_iter_with_arrow
{
  Base it_;

  using value_type       = cuda::std::iter_value_t<Base>;
  using difference_type  = cuda::std::intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  __host__ __device__ constexpr move_only_input_iter_with_arrow(Base it)
      : it_(cuda::std::move(it))
  {}
  constexpr move_only_input_iter_with_arrow(move_only_input_iter_with_arrow&&)                 = default;
  constexpr move_only_input_iter_with_arrow(const move_only_input_iter_with_arrow&)            = delete;
  constexpr move_only_input_iter_with_arrow& operator=(move_only_input_iter_with_arrow&&)      = default;
  constexpr move_only_input_iter_with_arrow& operator=(const move_only_input_iter_with_arrow&) = delete;

  __host__ __device__ constexpr move_only_input_iter_with_arrow& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++it_;
  }

  __host__ __device__ constexpr cuda::std::iter_reference_t<Base> operator*() const
  {
    return *it_;
  }

  template <class Base2 = Base, cuda::std::enable_if_t<HasArrow<Base> && cuda::std::copyable<Base>, int> = 0>
  __host__ __device__ constexpr auto operator->() const
  {
    return it_;
  }
};
static_assert(!cuda::std::copyable<move_only_input_iter_with_arrow<int*>>);
static_assert(cuda::std::input_iterator<move_only_input_iter_with_arrow<int*>>);

template <class Base>
struct move_iter_sentinel
{
  Base it_;
  explicit move_iter_sentinel() = default;
  __host__ __device__ constexpr move_iter_sentinel(Base it)
      : it_(cuda::std::move(it))
  {}
  __host__ __device__ constexpr friend bool
  operator==(const move_iter_sentinel& s, const move_only_input_iter_with_arrow<Base>& other)
  {
    return s.it_ == other.it_;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool
  operator==(const move_only_input_iter_with_arrow<Base>& other, const move_iter_sentinel& s)
  {
    return s.it_ == other.it_;
  }
  __host__ __device__ constexpr friend bool
  operator!=(const move_iter_sentinel& s, const move_only_input_iter_with_arrow<Base>& other)
  {
    return s.it_ != other.it_;
  }
  __host__ __device__ constexpr friend bool
  operator!=(const move_only_input_iter_with_arrow<Base>& other, const move_iter_sentinel& s)
  {
    return s.it_ != other.it_;
  }
#endif // TEST_STD_VER <= 2017
};
static_assert(cuda::std::sentinel_for<move_iter_sentinel<int*>, move_only_input_iter_with_arrow<int*>>);

struct MoveOnlyIterInner : BufferView<move_only_input_iter_with_arrow<Box*>, move_iter_sentinel<Box*>>
{
#if defined(TEST_COMPILER_NVRTC)
  MoveOnlyIterInner() noexcept = default;

  template <class T>
  __host__ __device__ constexpr MoveOnlyIterInner(T&& arr) noexcept(
    noexcept(BufferView<move_only_input_iter_with_arrow<Box*>, move_iter_sentinel<Box*>>(cuda::std::declval<T>())))
      : BufferView<move_only_input_iter_with_arrow<Box*>, move_iter_sentinel<Box*>>(_CUDA_VSTD::forward<T>(arr))
  {}
#else // ^^^ TEST_COMPILER_NVRTC ^^^ / vvv !TEST_COMPILER_NVRTC vvv
  using BufferView::BufferView;
#endif // !TEST_COMPILER_NVRTC

  using iterator = move_only_input_iter_with_arrow<Box*>;
  using sentinel = move_iter_sentinel<Box*>;

  __host__ __device__ iterator begin() const
  {
    return data_;
  }
  __host__ __device__ sentinel end() const
  {
    return sentinel{data_ + size_};
  }
};
static_assert(cuda::std::ranges::input_range<MoveOnlyIterInner>);

template <class Base>
struct arrow_input_iter
{
  Base it_;

  using value_type       = cuda::std::iter_value_t<Base>;
  using difference_type  = cuda::std::intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  arrow_input_iter() = default;
  __host__ __device__ constexpr arrow_input_iter(Base it)
      : it_(cuda::std::move(it))
  {}

  __host__ __device__ constexpr arrow_input_iter& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++it_;
  }

  __host__ __device__ constexpr cuda::std::iter_reference_t<Base> operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr auto operator->() const
  {
    return it_;
  }

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(const arrow_input_iter& x, const arrow_input_iter& y) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const arrow_input_iter& x, const arrow_input_iter& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool operator!=(const arrow_input_iter& x, const arrow_input_iter& y)
  {
    return x.it_ != y.it_;
  }
#endif // TEST_STD_VER <= 2017
};

using ArrowInner = BufferView<arrow_input_iter<Box*>>;
static_assert(cuda::std::ranges::input_range<ArrowInner>);
static_assert(HasArrow<cuda::std::ranges::iterator_t<ArrowInner>>);

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  Box buffer[4][4] = {{{1111}, {2222}, {3333}, {4444}},
                      {{555}, {666}, {777}, {888}},
                      {{99}, {1010}, {1111}, {1212}},
                      {{13}, {14}, {15}, {16}}};

  {
    // Copyable input iterator with arrow.
    using BoxView              = ValueView<Box>;
    ValueView<Box> children[4] = {BoxView(buffer[0]), BoxView(buffer[1]), BoxView(buffer[2]), BoxView(buffer[3])};
    cuda::std::ranges::join_view jv(ValueView<ValueView<Box>>{children});
    assert(jv.begin()->x == 1111);
    static_assert(HasArrow<decltype(jv.begin())>);
  }

  {
    cuda::std::ranges::join_view jv(buffer);
    assert(jv.begin()->x == 1111);
    static_assert(HasArrow<decltype(jv.begin())>);
  }

  {
    const cuda::std::ranges::join_view jv(buffer);
    assert(jv.begin()->x == 1111);
    static_assert(HasArrow<decltype(jv.begin())>);
  }

  {
    // LWG3500 `join_view::iterator::operator->()` is bogus
    // `operator->` should not be defined if inner iterator is not copyable
    // has-arrow<InnerIter> && !copyable<InnerIter>
    static_assert(HasArrow<move_only_input_iter_with_arrow<int*>>);
    MoveOnlyIterInner inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv{inners};
    static_assert(HasArrow<decltype(cuda::std::ranges::begin(inners[0]))>);
    static_assert(!HasArrow<decltype(jv.begin())>);
  }

  {
    // LWG3500 `join_view::iterator::operator->()` is bogus
    // `operator->` should not be defined if inner iterator does not have `operator->`
    // !has-arrow<InnerIter> && copyable<InnerIter>
    using Inner     = BufferView<forward_iterator<Box*>>;
    Inner inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv{inners};
    static_assert(!HasArrow<decltype(cuda::std::ranges::begin(inners[0]))>);
    static_assert(!HasArrow<decltype(jv.begin())>);
  }

  {
    // arrow returns inner iterator
    ArrowInner inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv{inners};
    static_assert(HasArrow<decltype(cuda::std::ranges::begin(inners[0]))>);
    static_assert(HasArrow<decltype(jv.begin())>);

    auto jv_it              = jv.begin();
    decltype(auto) arrow_it = jv_it.operator->();
    static_assert(cuda::std::same_as<decltype(arrow_it), arrow_input_iter<Box*>>);
    assert(arrow_it->x == 1111);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  return 0;
}
