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
// constexpr auto begin() const requires range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "types.h"

struct MutableView : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin();
  __host__ __device__ sentinel_wrapper<int*> end();
};

#if TEST_STD_VER >= 2020
template <class View>
concept BeginEnabled = requires(View v) { v.begin(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class View, class = void>
inline constexpr bool BeginEnabled = false;

template <class View>
inline constexpr bool BeginEnabled<View, cuda::std::void_t<decltype(cuda::std::declval<View>().begin())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert(BeginEnabled<cuda::std::ranges::common_view<CopyableView> const&>);
    static_assert(BeginEnabled<cuda::std::ranges::common_view<MutableView>&>);
    static_assert(!BeginEnabled<cuda::std::ranges::common_view<MutableView> const&>);
  }

  {
    SizedRandomAccessView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedRandomAccessView> common(view);
    decltype(auto) begin = common.begin();
    static_assert(cuda::std::same_as<decltype(begin), RandomAccessIter>);
    assert(begin == cuda::std::ranges::begin(view));
  }

  {
    SizedRandomAccessView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedRandomAccessView> const common(view);
    decltype(auto) begin = common.begin();
    static_assert(cuda::std::same_as<decltype(begin), RandomAccessIter>);
    assert(begin == cuda::std::ranges::begin(view));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  // The non-constexpr tests:
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    SizedForwardView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedForwardView> common(view);
    using CommonIter     = cuda::std::common_iterator<ForwardIter, sized_sentinel<ForwardIter>>;
    decltype(auto) begin = common.begin();
    static_assert(cuda::std::same_as<decltype(begin), CommonIter>);
    assert(begin == cuda::std::ranges::begin(view));
    decltype(auto) cbegin = cuda::std::as_const(common).begin();
    static_assert(cuda::std::same_as<decltype(cbegin), CommonIter>);
    assert(cbegin == cuda::std::ranges::begin(view));
  }

  {
    MoveOnlyView view{buf, buf + 8};
    cuda::std::ranges::common_view<MoveOnlyView> common(cuda::std::move(view));
    using CommonIter     = cuda::std::common_iterator<int*, sentinel_wrapper<int*>>;
    decltype(auto) begin = common.begin();
    static_assert(cuda::std::same_as<decltype(begin), CommonIter>);
    assert(begin == cuda::std::ranges::begin(view));
  }

  {
    CopyableView view{buf, buf + 8};
    cuda::std::ranges::common_view<CopyableView> const common(view);
    using CommonIter     = cuda::std::common_iterator<int*, sentinel_wrapper<int*>>;
    decltype(auto) begin = common.begin();
    static_assert(cuda::std::same_as<decltype(begin), CommonIter>);
    assert(begin == cuda::std::ranges::begin(view));
  }

  return 0;
}
