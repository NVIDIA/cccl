//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept;
//
// Effects: Equivalent to: return p+i;

#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../MinimalElementType.h"
#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_offset()
{
  ElementPool<cuda::std::remove_const_t<T>, 10> data;
  T* ptr = data.get_ptr();
  cuda::std::default_accessor<T> acc;
  for (int i = 0; i < 10; i++)
  {
    static_assert(
      cuda::std::is_same<decltype(acc.offset(ptr, i)), typename cuda::std::default_accessor<T>::data_handle_type>::value,
      "");
    static_assert(noexcept(acc.offset(ptr, i)));
    assert(acc.offset(ptr, i) == ptr + i);
  }
}

__host__ __device__ constexpr bool test()
{
  test_offset<int>();
  test_offset<const int>();
  test_offset<MinimalElementType>();
  test_offset<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  NV_IF_TARGET(NV_IS_HOST,
               ( // This fails because we cannot allocate on device at compile time
                 static_assert(test(), "");))
#endif // TEST_STD_VER >= 2020

  return 0;
}
