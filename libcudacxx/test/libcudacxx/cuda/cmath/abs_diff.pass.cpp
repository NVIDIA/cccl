//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/cmath>

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC constexpr auto invoke_abs_diff(T x, T y)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(x);
    DoNotOptimize(y);
  }
  return cuda::abs_diff(x, y);
}

template <class T>
TEST_FUNC constexpr void test()
{
  using U = cuda::std::make_unsigned_t<T>;

  static_assert(cuda::std::is_same_v<U, decltype(cuda::abs_diff(T{}, T{}))>);
  static_assert(noexcept(cuda::abs_diff(T{}, T{})));

  assert(invoke_abs_diff(T{0}, T{0}) == 0);
  assert(invoke_abs_diff(T{1}, T{0}) == 1);
  assert(invoke_abs_diff(T{0}, T{1}) == 1);
  assert(invoke_abs_diff(T{1}, T{1}) == 0);

  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(invoke_abs_diff(T{0}, T{0}) == 0);
    assert(invoke_abs_diff(T{-1}, T{0}) == 1);
    assert(invoke_abs_diff(T{0}, T{-1}) == 1);
    assert(invoke_abs_diff(T{-1}, T{-1}) == 0);
  }

  constexpr auto max = cuda::std::numeric_limits<T>::max();
  constexpr auto min = cuda::std::numeric_limits<T>::min();

  assert(invoke_abs_diff(T{max}, T{0}) == max);
  assert(invoke_abs_diff(T{0}, T{min}) == cuda::uabs(min));
  assert(invoke_abs_diff(T{max}, T{min}) == (cuda::uabs(max) + cuda::uabs(min)));
}

TEST_FUNC constexpr bool test()
{
  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
