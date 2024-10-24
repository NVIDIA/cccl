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

// cuda::std::views::join

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "types.h"

struct MoveOnlyOuter : SimpleForwardCommonOuter<ForwardCommonInner>
{
#if defined(TEST_COMPILER_NVRTC)
  MoveOnlyOuter() noexcept = default;

  template <class T>
  __host__ __device__ constexpr MoveOnlyOuter(T&& arr) noexcept(
    noexcept(SimpleForwardCommonOuter<ForwardCommonInner>(cuda::std::declval<T>())))
      : SimpleForwardCommonOuter<ForwardCommonInner>(_CUDA_VSTD::forward<T>(arr))
  {}
#else
  using SimpleForwardCommonOuter<ForwardCommonInner>::SimpleForwardCommonOuter;
#endif

  constexpr MoveOnlyOuter(MoveOnlyOuter&&)      = default;
  constexpr MoveOnlyOuter(const MoveOnlyOuter&) = delete;

  constexpr MoveOnlyOuter& operator=(MoveOnlyOuter&&)      = default;
  constexpr MoveOnlyOuter& operator=(const MoveOnlyOuter&) = delete;
};

struct Foo
{
  int i;
  __host__ __device__ constexpr Foo(int ii)
      : i(ii)
  {}
};

#if TEST_STD_VER >= 2020
template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { cuda::std::forward<View>(view) | cuda::std::forward<T>(t) };
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class View, class T, class = void>
inline constexpr bool CanBePiped = false;

template <class View, class T>
inline constexpr bool
  CanBePiped<View, T, cuda::std::void_t<decltype(cuda::std::declval<View>() | cuda::std::declval<T>())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer1[3]      = {1, 2, 3};
  int buffer2[2]      = {4, 5};
  int buffer3[4]      = {6, 7, 8, 9};
  Foo nested[2][3][3] = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}};

  {
    // Test `views::join(v)`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};
    using Result                 = cuda::std::ranges::join_view<cuda::std::ranges::ref_view<ForwardCommonInner[3]>>;
    decltype(auto) v             = cuda::std::views::join(inners);
    static_assert(cuda::std::same_as<decltype(v), Result>);
    assert(cuda::std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);
  }

  {
    // Test `views::join(move-only-view)`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};
    using Result                 = cuda::std::ranges::join_view<MoveOnlyOuter>;
    decltype(auto) v             = cuda::std::views::join(MoveOnlyOuter{inners});
    static_assert(cuda::std::same_as<decltype(v), Result>);
    assert(cuda::std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);

    static_assert(cuda::std::invocable<decltype(cuda::std::views::join), MoveOnlyOuter>);
#if !defined(TEST_COMPILER_NVRTC)
    static_assert(!cuda::std::invocable<decltype(cuda::std::views::join), MoveOnlyOuter&>);
#endif // !TEST_COMPILER_NVRTC
  }

  {
    // LWG3474 Nesting `join_views` is broken because of CTAD
    // views::join(join_view) should join the view instead of calling copy constructor
    auto jv = cuda::std::views::join(nested);
    ASSERT_SAME_TYPE(cuda::std::ranges::range_reference_t<decltype(jv)>, Foo(&)[3]);

    auto jv2 = cuda::std::views::join(jv);
    ASSERT_SAME_TYPE(cuda::std::ranges::range_reference_t<decltype(jv2)>, Foo&);

    assert(&(*jv2.begin()) == &nested[0][0][0]);
  }

  {
    // Test `v | views::join`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};

    using Result     = cuda::std::ranges::join_view<cuda::std::ranges::ref_view<ForwardCommonInner[3]>>;
    decltype(auto) v = inners | cuda::std::views::join;
    static_assert(cuda::std::same_as<decltype(v), Result>);
    assert(cuda::std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);
    static_assert(CanBePiped<decltype((inners)), decltype((cuda::std::views::join))>);
  }

  {
    // Test `move-only-view | views::join`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};
    using Result                 = cuda::std::ranges::join_view<MoveOnlyOuter>;
    decltype(auto) v             = MoveOnlyOuter{inners} | cuda::std::views::join;
    static_assert(cuda::std::same_as<decltype(v), Result>);
    assert(cuda::std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);

    static_assert(CanBePiped<MoveOnlyOuter, decltype((cuda::std::views::join))>);
#if !defined(TEST_COMPILER_NVRTC)
    static_assert(!CanBePiped<MoveOnlyOuter&, decltype((cuda::std::views::join))>);
#endif // !TEST_COMPILER_NVRTC
  }

  {
    // LWG3474 Nesting `join_views` is broken because of CTAD
    // join_view | views::join should join the view instead of calling copy constructor
    auto jv = nested | cuda::std::views::join | cuda::std::views::join;
    ASSERT_SAME_TYPE(cuda::std::ranges::range_reference_t<decltype(jv)>, Foo&);

    assert(&(*jv.begin()) == &nested[0][0][0]);
    static_assert(CanBePiped<decltype((nested)), decltype((cuda::std::views::join))>);
  }

  {
    // Test `adaptor | views::join`
    auto join_twice = cuda::std::views::join | cuda::std::views::join;
    auto jv         = nested | join_twice;
    ASSERT_SAME_TYPE(cuda::std::ranges::range_reference_t<decltype(jv)>, Foo&);

    assert(&(*jv.begin()) == &nested[0][0][0]);
    static_assert(CanBePiped<decltype((nested)), decltype((join_twice))>);
  }

  {
    static_assert(!CanBePiped<int, decltype((cuda::std::views::join))>);
    static_assert(!CanBePiped<Foo, decltype((cuda::std::views::join))>);
    static_assert(!CanBePiped<int(&)[2], decltype((cuda::std::views::join))>);
    static_assert(CanBePiped<int(&)[2][2], decltype((cuda::std::views::join))>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
#  if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC) // access to expired storage
  static_assert(test(), "");
#  endif // !TEST_COMPILER_NVCC && !TEST_COMPILER_NVRTC
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  return 0;
}
