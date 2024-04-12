//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03 && !stdlib=libc++

// <utility>

// template <class T>
//     typename conditional
//     <
//         !is_nothrow_move_constructible<T>::value && is_copy_constructible<T>::value,
//         const T&,
//         T&&
//     >::type
//     move_if_noexcept(T& x);

#include <cuda/std/utility>

#include "test_macros.h"

class A
{
  __host__ __device__ A(const A&);
  __host__ __device__ A& operator=(const A&);

public:
  __host__ __device__ A() {}
  __host__ __device__ A(A&&) {}
};

struct legacy
{
  __host__ __device__ legacy() {}
  __host__ __device__ legacy(const legacy&);
};

int main(int, char**)
{
  int i        = 0;
  const int ci = 0;
  unused(i);
  unused(ci);

  legacy l;
  A a;
  const A ca;
  unused(l);
  unused(a);
  unused(ca);

  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(i)), int&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(ci)), const int&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(a)), A&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(ca)), const A&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(l)), const legacy&>::value), "");

#if TEST_STD_VER > 2011
  constexpr int i1 = 23;
  constexpr int i2 = cuda::std::move_if_noexcept(i1);
  static_assert(i2 == 23, "");
#endif

  return 0;
}
