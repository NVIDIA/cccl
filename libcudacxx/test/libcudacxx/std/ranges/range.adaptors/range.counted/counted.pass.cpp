//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::views::counted;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

struct RvalueConvertible
{
  RvalueConvertible(const RvalueConvertible&) = delete;
  __host__ __device__ operator int() &&;
};

struct LvalueConvertible
{
  LvalueConvertible(const LvalueConvertible&) = delete;
  __host__ __device__ operator int() &;
};

struct OnlyExplicitlyConvertible
{
  __host__ __device__ explicit operator int() const;
};

#if TEST_STD_VER >= 2020
template <class... Ts>
concept CountedInvocable = requires(Ts&&... ts) { cuda::std::views::counted(cuda::std::forward<Ts>(ts)...); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class... Ts>
inline constexpr bool CountedInvocable = cuda::std::invocable<cuda::std::ranges::views::__counted::__fn, Ts...>;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
#if defined(_LIBCUDACXX_ADDRESSOF)
    static_assert(
      cuda::std::addressof(cuda::std::views::counted) == cuda::std::addressof(cuda::std::ranges::views::counted));
#endif // _LIBCUDACXX_ADDRESSOF

    static_assert(CountedInvocable<int*, size_t>);
    static_assert(!CountedInvocable<int*, LvalueConvertible>);
    static_assert(CountedInvocable<int*, LvalueConvertible&>);
    static_assert(CountedInvocable<int*, RvalueConvertible>);
    static_assert(!CountedInvocable<int*, RvalueConvertible&>);
    static_assert(!CountedInvocable<int*, OnlyExplicitlyConvertible>);
    static_assert(!CountedInvocable<int*, int*>);
    static_assert(!CountedInvocable<int*>);
    static_assert(!CountedInvocable<size_t>);
    static_assert(!CountedInvocable<>);
  }

  {
    auto c1 = cuda::std::views::counted(buffer, 3);
    auto c2 = cuda::std::views::counted(cuda::std::as_const(buffer), 3);

    ASSERT_SAME_TYPE(decltype(c1), cuda::std::span<int>);
    ASSERT_SAME_TYPE(decltype(c2), cuda::std::span<const int>);

    assert(c1.data() == buffer && c1.size() == 3);
    assert(c2.data() == buffer && c2.size() == 3);
  }

  {
    auto it  = contiguous_iterator<int*>(buffer);
    auto cit = contiguous_iterator<const int*>(buffer);

    auto c1 = cuda::std::views::counted(it, 3);
    auto c2 = cuda::std::views::counted(cuda::std::as_const(it), 3);
    auto c3 = cuda::std::views::counted(cuda::std::move(it), 3);
    auto c4 = cuda::std::views::counted(contiguous_iterator<int*>(buffer), 3);
    auto c5 = cuda::std::views::counted(cit, 3);
    auto c6 = cuda::std::views::counted(cuda::std::as_const(cit), 3);
    auto c7 = cuda::std::views::counted(cuda::std::move(cit), 3);
    auto c8 = cuda::std::views::counted(contiguous_iterator<const int*>(buffer), 3);

    ASSERT_SAME_TYPE(decltype(c1), cuda::std::span<int>);
    ASSERT_SAME_TYPE(decltype(c2), cuda::std::span<int>);
    ASSERT_SAME_TYPE(decltype(c3), cuda::std::span<int>);
    ASSERT_SAME_TYPE(decltype(c4), cuda::std::span<int>);
    ASSERT_SAME_TYPE(decltype(c5), cuda::std::span<const int>);
    ASSERT_SAME_TYPE(decltype(c6), cuda::std::span<const int>);
    ASSERT_SAME_TYPE(decltype(c7), cuda::std::span<const int>);
    ASSERT_SAME_TYPE(decltype(c8), cuda::std::span<const int>);

    assert(c1.data() == buffer && c1.size() == 3);
    assert(c2.data() == buffer && c2.size() == 3);
    assert(c3.data() == buffer && c3.size() == 3);
    assert(c4.data() == buffer && c4.size() == 3);
    assert(c5.data() == buffer && c5.size() == 3);
    assert(c6.data() == buffer && c6.size() == 3);
    assert(c7.data() == buffer && c7.size() == 3);
    assert(c8.data() == buffer && c8.size() == 3);
  }

  {
    auto it  = random_access_iterator<int*>(buffer);
    auto cit = random_access_iterator<const int*>(buffer);

    auto c1 = cuda::std::views::counted(it, 3);
    auto c2 = cuda::std::views::counted(cuda::std::as_const(it), 3);
    auto c3 = cuda::std::views::counted(cuda::std::move(it), 3);
    auto c4 = cuda::std::views::counted(random_access_iterator<int*>(buffer), 3);
    auto c5 = cuda::std::views::counted(cit, 3);
    auto c6 = cuda::std::views::counted(cuda::std::as_const(cit), 3);
    auto c7 = cuda::std::views::counted(cuda::std::move(cit), 3);
    auto c8 = cuda::std::views::counted(random_access_iterator<const int*>(buffer), 3);

    ASSERT_SAME_TYPE(decltype(c1), cuda::std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c2), cuda::std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c3), cuda::std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c4), cuda::std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c5), cuda::std::ranges::subrange<random_access_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(c6), cuda::std::ranges::subrange<random_access_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(c7), cuda::std::ranges::subrange<random_access_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(c8), cuda::std::ranges::subrange<random_access_iterator<const int*>>);

    assert(c1.begin() == it && c1.end() == it + 3);
    assert(c2.begin() == it && c2.end() == it + 3);
    assert(c3.begin() == it && c3.end() == it + 3);
    assert(c4.begin() == it && c4.end() == it + 3);
    assert(c5.begin() == cit && c5.end() == cit + 3);
    assert(c6.begin() == cit && c6.end() == cit + 3);
    assert(c7.begin() == cit && c7.end() == cit + 3);
    assert(c8.begin() == cit && c8.end() == cit + 3);
  }

  {
    auto it  = bidirectional_iterator<int*>(buffer);
    auto cit = bidirectional_iterator<const int*>(buffer);

    auto c1 = cuda::std::views::counted(it, 3);
    auto c2 = cuda::std::views::counted(cuda::std::as_const(it), 3);
    auto c3 = cuda::std::views::counted(cuda::std::move(it), 3);
    auto c4 = cuda::std::views::counted(bidirectional_iterator<int*>(buffer), 3);
    auto c5 = cuda::std::views::counted(cit, 3);
    auto c6 = cuda::std::views::counted(cuda::std::as_const(cit), 3);
    auto c7 = cuda::std::views::counted(cuda::std::move(cit), 3);
    auto c8 = cuda::std::views::counted(bidirectional_iterator<const int*>(buffer), 3);

    using Expected =
      cuda::std::ranges::subrange<cuda::std::counted_iterator<decltype(it)>, cuda::std::default_sentinel_t>;
    using ConstExpected =
      cuda::std::ranges::subrange<cuda::std::counted_iterator<decltype(cit)>, cuda::std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c1), Expected);
    ASSERT_SAME_TYPE(decltype(c2), Expected);
    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);
    ASSERT_SAME_TYPE(decltype(c5), ConstExpected);
    ASSERT_SAME_TYPE(decltype(c6), ConstExpected);
    ASSERT_SAME_TYPE(decltype(c7), ConstExpected);
    ASSERT_SAME_TYPE(decltype(c8), ConstExpected);

    assert(c1.begin().base() == it && c1.size() == 3);
    assert(c2.begin().base() == it && c2.size() == 3);
    assert(c3.begin().base() == it && c3.size() == 3);
    assert(c4.begin().base() == it && c4.size() == 3);
    assert(c5.begin().base() == cit && c5.size() == 3);
    assert(c6.begin().base() == cit && c6.size() == 3);
    assert(c7.begin().base() == cit && c7.size() == 3);
    assert(c8.begin().base() == cit && c8.size() == 3);
  }

  {
    auto it = cpp17_output_iterator<int*>(buffer);

    auto c1 = cuda::std::views::counted(it, 3);
    auto c2 = cuda::std::views::counted(cuda::std::as_const(it), 3);
    auto c3 = cuda::std::views::counted(cuda::std::move(it), 3);
    auto c4 = cuda::std::views::counted(cpp17_output_iterator<int*>(buffer), 3);

    using Expected =
      cuda::std::ranges::subrange<cuda::std::counted_iterator<decltype(it)>, cuda::std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c1), Expected);
    ASSERT_SAME_TYPE(decltype(c2), Expected);
    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);

    assert(base(c1.begin().base()) == buffer && c1.size() == 3);
    assert(base(c2.begin().base()) == buffer && c2.size() == 3);
    assert(base(c3.begin().base()) == buffer && c3.size() == 3);
    assert(base(c4.begin().base()) == buffer && c4.size() == 3);
  }

  {
    auto it = cpp17_input_iterator<int*>(buffer);

    auto c1 = cuda::std::views::counted(it, 3);
    auto c2 = cuda::std::views::counted(cuda::std::as_const(it), 3);
    auto c3 = cuda::std::views::counted(cuda::std::move(it), 3);
    auto c4 = cuda::std::views::counted(cpp17_input_iterator<int*>(buffer), 3);

    using Expected =
      cuda::std::ranges::subrange<cuda::std::counted_iterator<decltype(it)>, cuda::std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c1), Expected);
    ASSERT_SAME_TYPE(decltype(c2), Expected);
    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);

    assert(base(c1.begin().base()) == buffer && c1.size() == 3);
    assert(base(c2.begin().base()) == buffer && c2.size() == 3);
    assert(base(c3.begin().base()) == buffer && c3.size() == 3);
    assert(base(c4.begin().base()) == buffer && c4.size() == 3);
  }

  {
    auto it = cpp20_input_iterator<int*>(buffer);

    static_assert(!cuda::std::copyable<cpp20_input_iterator<int*>>);
    // C++17 tries really hard to use the deleted operator= of the move only cpp20_input_iterator
#if TEST_STD_VER >= 2020
    static_assert(!CountedInvocable<cpp20_input_iterator<int*>&, int>);
    static_assert(!CountedInvocable<const cpp20_input_iterator<int*>&, int>);
#endif // TEST_STD_VER >= 2020
    auto c3 = cuda::std::views::counted(cuda::std::move(it), 3);
    auto c4 = cuda::std::views::counted(cpp20_input_iterator<int*>(buffer), 3);

    using Expected =
      cuda::std::ranges::subrange<cuda::std::counted_iterator<decltype(it)>, cuda::std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);

    assert(base(c3.begin().base()) == buffer && c3.size() == 3);
    assert(base(c4.begin().base()) == buffer && c4.size() == 3);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
