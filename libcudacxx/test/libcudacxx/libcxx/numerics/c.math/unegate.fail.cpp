//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>

template <class T>
__host__ __device__ constexpr void test_type()
{
  using U = cuda::std::make_unsigned_t<T>;

  cuda::__unegate(U(U(cuda::std::numeric_limits<T>::max()) + 2));
  cuda::__unegate(cuda::std::numeric_limits<U>::max());
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
