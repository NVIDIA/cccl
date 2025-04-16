//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string/constexpr_c_functions.h>
#include <cuda/std/cassert>

template <class T>
__host__ __device__ constexpr void test_memset(T* ptr, T c, cuda::std::size_t n, const T* ref)
{
  assert(cuda::std::__cccl_memset(ptr, c, n) == ptr);
  assert(cuda::std::__cccl_memcmp(ptr, ref, n) == 0);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  {
    test_memset<T>(nullptr, 1, 0, nullptr);
  }
  {
    constexpr T value = 127;
    T obj{127};
    T ref{value};
    test_memset(&obj, value, 1, &ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{127, 46, 7};
    test_memset(arr, value, 0, ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{value, 46, 7};
    test_memset(arr, value, 1, ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{value, value, 7};
    test_memset(arr, value, 2, ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{value, value, value};
    test_memset(arr, value, 3, ref);
  }
}

__host__ __device__ constexpr bool test()
{
  test_type<char>();
#if _LIBCUDACXX_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _LIBCUDACXX_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
