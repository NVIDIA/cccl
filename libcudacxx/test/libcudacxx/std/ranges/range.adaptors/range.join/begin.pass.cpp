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

// constexpr auto begin();
// constexpr auto begin() const
//    requires input_range<const V> &&
//             is_reference_v<range_reference_t<const V>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

struct NonSimpleParentView : cuda::std::ranges::view_base
{
  __host__ __device__ ChildView* begin()
  {
    return nullptr;
  }
  __host__ __device__ const ChildView* begin() const;
  __host__ __device__ const ChildView* end() const;
};

struct SimpleParentView : cuda::std::ranges::view_base
{
  __host__ __device__ const ChildView* begin() const;
  __host__ __device__ const ChildView* end() const;
};

struct ConstNotRange : cuda::std::ranges::view_base
{
  __host__ __device__ const ChildView* begin();
  __host__ __device__ const ChildView* end();
};
static_assert(cuda::std::ranges::range<ConstNotRange>);
static_assert(!cuda::std::ranges::range<const ConstNotRange>);

#if TEST_STD_VER >= 2020
template <class T>
concept HasConstBegin = requires(const T& t) { t.begin(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasConstBegin = false;

template <class T>
inline constexpr bool HasConstBegin<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>().begin())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv               = cuda::std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 4),
      CopyableChild(buffer[1], 0),
      CopyableChild(buffer[2], 1),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  // Parent is empty.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]), CopyableChild(buffer[3])};
    cuda::std::ranges::join_view jv(ParentView(children, 0));
    assert(jv.begin() == jv.end());
  }

  // Parent size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer[0])};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(*jv.begin() == 1111);
  }

  // Parent and child size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer[0], 1)};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(*jv.begin() == 1111);
  }

  // Parent size is one child is empty
  {
    CopyableChild children[1] = {CopyableChild(buffer[0], 0)};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.begin() == jv.end());
  }

  // Has all empty children.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 0),
      CopyableChild(buffer[1], 0),
      CopyableChild(buffer[2], 0),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView{children});
    assert(jv.begin() == jv.end());
  }

  // First child is empty, others are not.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 4),
      CopyableChild(buffer[1], 0),
      CopyableChild(buffer[2], 0),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  // Last child is empty, others are not.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 4),
      CopyableChild(buffer[1], 4),
      CopyableChild(buffer[2], 4),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

#if defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED) // nvcc believes we are accessing expired storage
#  if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
  if (!cuda::std::is_constant_evaluated())
#  endif // TEST_COMPILER_NVCC || TEST_COMPILER_NVRTC
  {
    cuda::std::ranges::join_view jv(buffer);
    assert(*jv.begin() == 1111);
  }

#  if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
  if (!cuda::std::is_constant_evaluated())
#  endif // TEST_COMPILER_NVCC || TEST_COMPILER_NVRTC
  {
    const cuda::std::ranges::join_view jv(buffer);
    assert(*jv.begin() == 1111);
    static_assert(HasConstBegin<decltype(jv)>);
  }
#endif // _LIBCUDACXX_IS_CONSTANT_EVALUATED

  // !forward_range<const V>
  {
    cuda::std::ranges::join_view jv{ConstNotRange{}};
    static_assert(!HasConstBegin<decltype(jv)>);
    unused(jv);
  }

  // !is_reference_v<range_reference_t<const V>>
  {
    auto innerRValueRange = cuda::std::views::iota(0, 5) | cuda::std::views::transform([](int) {
                              return ChildView{};
                            });
    static_assert(!cuda::std::is_reference_v<cuda::std::ranges::range_reference_t<const decltype(innerRValueRange)>>);
    cuda::std::ranges::join_view jv{innerRValueRange};
    static_assert(!HasConstBegin<decltype(jv)>);
    unused(jv);
  }

  // !simple-view<V>
  {
    cuda::std::ranges::join_view<NonSimpleParentView> jv;
    static_assert(!cuda::std::same_as<decltype(jv.begin()), decltype(cuda::std::as_const(jv).begin())>);
    unused(jv);
  }

  // simple-view<V> && is_reference_v<range_reference_t<V>>;
  {
    cuda::std::ranges::join_view<SimpleParentView> jv;
    static_assert(cuda::std::same_as<decltype(jv.begin()), decltype(cuda::std::as_const(jv).begin())>);
    unused(jv);
  }

#ifdef _LIBCUDACXX_HAS_STRING
  // Check stashing iterators (LWG3698: regex_iterator and join_view don't work together very well)
  {
    cuda::std::ranges::join_view<StashingRange> jv;
    assert(cuda::std::ranges::equal(cuda::std::views::counted(jv.begin(), 10), cuda::std::string_view{"aababcabcd"}));
  }

  // LWG3700: The `const begin` of the `join_view` family does not require `InnerRng` to be a range
  {
    cuda::std::ranges::join_view<ConstNonJoinableRange> jv;
    static_assert(!HasConstBegin<decltype(jv)>);
  }
#endif // _LIBCUDACXX_HAS_STRING

#if 0 // Not yet implemented split_view
  // Check example from LWG3700
  {
    auto r = cuda::std::views::iota(0, 5) | cuda::std::views::split(1);
    auto s = cuda::std::views::single(r);
    auto j = s | cuda::std::views::join;
    auto f = j.front();
    assert(cuda::std::ranges::equal(f, cuda::std::views::single(0)));
  }
#endif // Not yet implemented split_view

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
