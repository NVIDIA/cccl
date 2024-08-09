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

// constexpr explicit zip_view(Views...)

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_LIBCUDACXX_CONCEPT implicitly_constructible_from = cuda::std::invocable<decltype(&conversion_test<T>), Args...>;

// test constructor is explicit
static_assert(cuda::std::constructible_from<cuda::std::ranges::zip_view<SimpleCommon>, SimpleCommon>);
static_assert(!implicitly_constructible_from<cuda::std::ranges::zip_view<SimpleCommon>, SimpleCommon>);

static_assert(
  cuda::std::constructible_from<cuda::std::ranges::zip_view<SimpleCommon, SimpleCommon>, SimpleCommon, SimpleCommon>);
static_assert(
  !implicitly_constructible_from<cuda::std::ranges::zip_view<SimpleCommon, SimpleCommon>, SimpleCommon, SimpleCommon>);

struct MoveAwareView : cuda::std::ranges::view_base
{
  int moves                 = 0;
  constexpr MoveAwareView() = default;
  __host__ __device__ constexpr MoveAwareView(MoveAwareView&& other)
      : moves(other.moves + 1)
  {
    other.moves = 1;
  }
  __host__ __device__ constexpr MoveAwareView& operator=(MoveAwareView&& other)
  {
    moves       = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  __host__ __device__ constexpr const int* begin() const
  {
    return &moves;
  }
  __host__ __device__ constexpr const int* end() const
  {
    return &moves + 1;
  }
};

template <class View1, class View2, class Input1, class Input2>
__host__ __device__ constexpr void constructorTest(Input1&& buffer1, Input2&& buffer2)
{
  cuda::std::ranges::zip_view<View1, View2> v{View1{buffer1}, View2{buffer2}};
  auto [i, j] = *v.begin();
  assert(i == buffer1[0]);
  assert(j == buffer2[0]);
};

__host__ __device__ constexpr bool test()
{
  int buffer[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
  int buffer2[4] = {9, 8, 7, 6};

  {
    // constructor from views
    cuda::std::ranges::zip_view v(
      SizedRandomAccessView{buffer}, cuda::std::views::iota(0), cuda::std::ranges::single_view(2.));
    auto [i, j, k] = *v.begin();
    assert(i == 1);
    assert(j == 0);
    assert(k == 2.0);
  }

  {
    // arguments are moved once
    MoveAwareView mv;
    cuda::std::ranges::zip_view v{cuda::std::move(mv), MoveAwareView{}};
    auto [numMoves1, numMoves2] = *v.begin();
    assert(numMoves1 == 2); // one move from the local variable to parameter, one move from parameter to member
    assert(numMoves2 == 1);
  }

  // input and forward
  {
    constructorTest<InputCommonView, ForwardSizedView>(buffer, buffer2);
  }

  // bidi and random_access
  {
    constructorTest<BidiCommonView, SizedRandomAccessView>(buffer, buffer2);
  }

  // contiguous
  {
    constructorTest<ContiguousCommonView, ContiguousCommonView>(buffer, buffer2);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
