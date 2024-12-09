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

// transform_view::<iterator>::operator[]

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  cuda::std::ranges::transform_view transformView1(MoveOnlyView{buff}, PlusOneMutable{});
  auto iter1 = cuda::std::move(transformView1).begin() + 1;
  assert(iter1[0] == 2);
  assert(iter1[4] == 6);

#if !defined(TEST_COMPILER_ICC) // broken noexcept
  ASSERT_NOT_NOEXCEPT(
    cuda::std::declval<
      cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>>()[0]);
#endif // !TEST_COMPILER_ICC
  LIBCPP_ASSERT_NOEXCEPT(
    cuda::std::declval<
      cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOneNoexcept>>>()[0]);

  ASSERT_SAME_TYPE(
    int,
    decltype(cuda::std::declval<cuda::std::ranges::transform_view<RandomAccessView, PlusOneMutable>>().begin()[0]));
  ASSERT_SAME_TYPE(
    int&, decltype(cuda::std::declval<cuda::std::ranges::transform_view<RandomAccessView, Increment>>().begin()[0]));
  ASSERT_SAME_TYPE(
    int&&,
    decltype(cuda::std::declval<cuda::std::ranges::transform_view<RandomAccessView, IncrementRvalueRef>>().begin()[0]));

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
