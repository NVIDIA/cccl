//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
// class complex
// {
// public:
//   using value_type = T;
//   ...
// };

#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC void test()
{
  using C = cuda::std::complex<T>;

  static_assert(cuda::std::is_default_constructible_v<C>);
  static_assert(!cuda::std::is_trivially_default_constructible_v<C>);
  static_assert(cuda::std::is_nothrow_default_constructible_v<C>);

  static_assert(cuda::std::is_copy_constructible_v<C>);
  static_assert(cuda::std::is_trivially_copy_constructible_v<C> == cuda::std::is_floating_point_v<T>);
  static_assert(cuda::std::is_nothrow_copy_constructible_v<C>);

  static_assert(cuda::std::is_move_constructible_v<C>);
  static_assert(cuda::std::is_trivially_move_constructible_v<C> == cuda::std::is_floating_point_v<T>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<C>);

  static_assert(cuda::std::is_copy_assignable_v<C>);
  static_assert(cuda::std::is_trivially_copy_assignable_v<C> == cuda::std::is_floating_point_v<T>);
  static_assert(cuda::std::is_nothrow_copy_assignable_v<C>);

  static_assert(cuda::std::is_move_assignable_v<C>);
  static_assert(cuda::std::is_trivially_move_assignable_v<C> == cuda::std::is_floating_point_v<T>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<C>);

  static_assert(cuda::std::is_trivially_destructible_v<C>);
  static_assert(cuda::std::is_trivially_copyable_v<C> == cuda::std::is_floating_point_v<T>);
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if !TEST_COMPILER(GCC, <, 10) // Old GCC considers the defaulted constructors as deleted
#  if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#  endif // _LIBCUDACXX_HAS_NVFP16()
#  if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#  endif // _LIBCUDACXX_HAS_NVBF16()
#endif // !TEST_COMPILER(GCC, <, 01)

  return 0;
}
