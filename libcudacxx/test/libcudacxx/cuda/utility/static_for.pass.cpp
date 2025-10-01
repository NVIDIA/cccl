//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

template <typename ExpectedType>
struct Op
{
  int i         = 0;
  int step      = 1;
  int max_iters = 0;
  int count     = 0;

  template <typename Index>
  __host__ __device__ constexpr void operator()(Index index)
  {
    static_assert(cuda::std::is_same_v<ExpectedType, decltype(index())>);
    [[maybe_unused]] constexpr auto value = index(); // compile-time evaluation
    assert(value == i);
    i += step;
    assert(count < max_iters);
    ++count;
  }
};

struct OpArgs
{
  template <typename Index>
  __host__ __device__ constexpr void operator()(Index index, int, int, int)
  {
    [[maybe_unused]] constexpr auto value = index(); // compile-time evaluation
  }
};

struct Op2D
{
  template <typename Index>
  __host__ __device__ constexpr void operator()(Index index)
  {
    using index_t = typename Index::value_type;
    if constexpr (index > 0)
    {
      cuda::static_for<index()>(Op<index_t>{0, 1, static_cast<int>(index.value)});
      cuda::static_for<index_t, index()>(Op<index_t>{0, 1, static_cast<int>(index.value)});
    }
  }
};

struct OpThrowing
{
  template <class Idx>
  __host__ __device__ constexpr void operator()(Idx) noexcept(false)
  {}
};

template <class IdxType>
struct OpThrowingIdx1
{
  template <class Idx>
  __host__ __device__ constexpr void operator()(Idx) noexcept
  {}
  __host__ __device__ constexpr void operator()(cuda::std::integral_constant<IdxType, IdxType{1}>) noexcept(false) {}
};

template <typename T>
__host__ __device__ constexpr void test()
{
  cuda::static_for<T{10}>(Op<T>{0, 1, 10});
  cuda::static_for<T{10}>(OpArgs{}, 1, 2, 3);
  cuda::static_for<T{10}>(Op2D{});

  cuda::static_for<T{15}, 20>(Op<T>{15, 1, 5});
  cuda::static_for<T{15}, 137, 5>(Op<T>{15, 5, 137 - 15 / 5});

  if constexpr (cuda::std::is_signed_v<T>)
  {
    cuda::static_for<T{-15}, T{15}, T{5}>(Op<T>{-15, 5, 6});
    cuda::static_for<T{15}, T{-15}, T{-5}>(Op<T>{15, -5, 6});
  }

  // gcc < 9 and msvc < 19.42 in C++17 mode have problems determining noexcept for static_for,
  // see https://godbolt.org/z/7rT7bTxcK
#if !_CCCL_COMPILER(GCC, <, 9) && !(_CCCL_COMPILER(MSVC, <, 19, 42) && _CCCL_STD_VER == 2017)
  // noexcept test for an always throwing operator
  {
    // 1. The function should be noexcept if there are 0 iterations even if the operator is not noexcept
    static_assert(noexcept(cuda::static_for<T{0}>(OpThrowing{})));
    static_assert(noexcept(cuda::static_for<T{0}, T{0}>(OpThrowing{})));
    static_assert(noexcept(cuda::static_for<T, 0>(OpThrowing{})));
    static_assert(noexcept(cuda::static_for<T, 0, 0>(OpThrowing{})));

    // 2. The function should NOT be noexcept if there are iterations
    static_assert(!noexcept(cuda::static_for<T{1}>(OpThrowing{})));
    static_assert(!noexcept(cuda::static_for<T{0}, T{1}>(OpThrowing{})));
    static_assert(!noexcept(cuda::static_for<T, 1>(OpThrowing{})));
    static_assert(!noexcept(cuda::static_for<T, 0, 1>(OpThrowing{})));
  }

  // noexcept test for an operator that throws only if invked on index 1
  {
    // 1. The function should be noexcept in range [0, 1)]
    static_assert(noexcept(cuda::static_for<T{1}>(OpThrowingIdx1<T>{})));
    static_assert(noexcept(cuda::static_for<T{0}, T{1}>(OpThrowingIdx1<T>{})));
    static_assert(noexcept(cuda::static_for<T, 1>(OpThrowingIdx1<T>{})));
    static_assert(noexcept(cuda::static_for<T, 0, 1>(OpThrowingIdx1<T>{})));

    // 2. The function should NOT be noexcept in range [0, 2)
    static_assert(!noexcept(cuda::static_for<T{2}>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T{0}, T{2}>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T, 2>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T, 0, 2>(OpThrowingIdx1<T>{})));

    // 3. The function should NOT be noexcept in range [0, 3)
    static_assert(!noexcept(cuda::static_for<T{3}>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T{0}, T{3}>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T, 3>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T, 0, 3>(OpThrowingIdx1<T>{})));

    // 4. The function should be noexcept in range [2, 3)
    static_assert(noexcept(cuda::static_for<T{2}, T{3}>(OpThrowingIdx1<T>{})));
    static_assert(noexcept(cuda::static_for<T, 2, 3>(OpThrowingIdx1<T>{})));

    // 5. The function should be noexcept when the step is 2
    static_assert(noexcept(cuda::static_for<T{0}, T{2}, T{2}>(OpThrowingIdx1<T>{})));
    static_assert(noexcept(cuda::static_for<T, 0, 2, 2>(OpThrowingIdx1<T>{})));

    // 6. The function should NOT be noexcept when the step is 2 but we are starting from 1
    static_assert(!noexcept(cuda::static_for<T{1}, T{3}, T{2}>(OpThrowingIdx1<T>{})));
    static_assert(!noexcept(cuda::static_for<T, 1, 3, 2>(OpThrowingIdx1<T>{})));
  }
#endif // !_CCCL_COMPILER(GCC, <, 9) && !(_CCCL_COMPILER(MSVC, <, 19, 42) && _CCCL_STD_VER == 2017)
}

__host__ __device__ constexpr bool test()
{
  test<short>();
  test<int>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();
  return true;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
