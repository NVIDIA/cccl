//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/vector_types>

template <class VT>
__host__ __device__ constexpr bool test_is_vector_type(cuda::std::true_type)
{
  static_assert(cuda::is_vector_type<VT>::value);
  static_assert(cuda::is_vector_type<const VT>::value);
  static_assert(cuda::is_vector_type<volatile VT>::value);
  static_assert(cuda::is_vector_type<const volatile VT>::value);

  static_assert(cuda::is_vector_type_v<VT>);
  static_assert(cuda::is_vector_type_v<const VT>);
  static_assert(cuda::is_vector_type_v<volatile VT>);
  static_assert(cuda::is_vector_type_v<const volatile VT>);

  return true;
}

template <class VT>
__host__ __device__ constexpr bool test_is_vector_type(cuda::std::false_type)
{
  static_assert(!cuda::is_vector_type<VT>::value);
  static_assert(!cuda::is_vector_type<const VT>::value);
  static_assert(!cuda::is_vector_type<volatile VT>::value);
  static_assert(!cuda::is_vector_type<const volatile VT>::value);

  static_assert(!cuda::is_vector_type_v<VT>);
  static_assert(!cuda::is_vector_type_v<const VT>);
  static_assert(!cuda::is_vector_type_v<volatile VT>);
  static_assert(!cuda::is_vector_type_v<const volatile VT>);

  return true;
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_is_vector_type<cuda::vector_type_t<T, 1>>(cuda::std::true_type{});
  test_is_vector_type<cuda::vector_type_t<T, 2>>(cuda::std::true_type{});
  test_is_vector_type<cuda::vector_type_t<T, 3>>(cuda::std::true_type{});
  test_is_vector_type<cuda::vector_type_t<T, 4>>(cuda::std::true_type{});

  test_is_vector_type<cuda::vector_type_t<T, 1>&>(cuda::std::false_type{});
  test_is_vector_type<cuda::vector_type_t<T, 2>&>(cuda::std::false_type{});
  test_is_vector_type<cuda::vector_type_t<T, 3>&>(cuda::std::false_type{});
  test_is_vector_type<cuda::vector_type_t<T, 4>&>(cuda::std::false_type{});

  test_is_vector_type<cuda::vector_type_t<T, 1>&&>(cuda::std::false_type{});
  test_is_vector_type<cuda::vector_type_t<T, 2>&&>(cuda::std::false_type{});
  test_is_vector_type<cuda::vector_type_t<T, 3>&&>(cuda::std::false_type{});
  test_is_vector_type<cuda::vector_type_t<T, 4>&&>(cuda::std::false_type{});

  test_is_vector_type<T>(cuda::std::false_type{});
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();

  test_type<float>();
  test_type<double>();

  test_is_vector_type<void>(cuda::std::false_type{});
  test_is_vector_type<int*>(cuda::std::false_type{});
  test_is_vector_type<int&>(cuda::std::false_type{});
  test_is_vector_type<float&&>(cuda::std::false_type{});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
