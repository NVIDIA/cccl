//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/vector_types>

template <class VT>
__host__ __device__ constexpr VT test_make_vector()
{
  VT ret{};
  ret.x = static_cast<decltype(ret.x)>(0);
  if constexpr (cuda::std::tuple_size_v<VT> > 1)
  {
    ret.y = static_cast<decltype(ret.y)>(1);
  }
  if constexpr (cuda::std::tuple_size_v<VT> > 2)
  {
    ret.z = static_cast<decltype(ret.z)>(2);
  }
  if constexpr (cuda::std::tuple_size_v<VT> > 3)
  {
    ret.w = static_cast<decltype(ret.w)>(3);
  }
  return ret;
}

template <class VT, class T, size_t N, size_t I>
__host__ __device__ constexpr void test_get()
{
  {
    VT v = test_make_vector<VT>();
    static_assert(cuda::std::is_same_v<T&, decltype(cuda::std::get<I>(v))>);
    static_assert(noexcept(cuda::std::get<I>(v)));
    T& r = cuda::std::get<I>(v);
    assert(r == T{I});
  }
  {
    const VT v = test_make_vector<VT>();
    static_assert(cuda::std::is_same_v<const T&, decltype(cuda::std::get<I>(v))>);
    static_assert(noexcept(cuda::std::get<I>(v)));
    const T& r = cuda::std::get<I>(v);
    assert(r == T{I});
  }
  {
    VT v = test_make_vector<VT>();
    static_assert(cuda::std::is_same_v<T&&, decltype(cuda::std::get<I>(cuda::std::move(v)))>);
    static_assert(noexcept(cuda::std::get<I>(cuda::std::move(v))));
    T&& r = cuda::std::get<I>(cuda::std::move(v));
    assert(r == T{I});
  }
  {
    const VT v = test_make_vector<VT>();
    static_assert(cuda::std::is_same_v<const T&&, decltype(cuda::std::get<I>(cuda::std::move(v)))>);
    static_assert(noexcept(cuda::std::get<I>(cuda::std::move(v))));
    const T&& r = cuda::std::get<I>(cuda::std::move(v));
    assert(r == T{I});
  }
}

template <class VT, class T, size_t N>
__host__ __device__ constexpr void test_get()
{
  test_get<VT, T, N, 0>();
  if constexpr (N > 1)
  {
    test_get<VT, T, N, 1>();
  }
  if constexpr (N > 2)
  {
    test_get<VT, T, N, 2>();
  }
  if constexpr (N > 3)
  {
    test_get<VT, T, N, 3>();
  }
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_get<cuda::vector_type_t<T, 1>, T, 1>();
  test_get<cuda::vector_type_t<T, 2>, T, 2>();
  test_get<cuda::vector_type_t<T, 3>, T, 3>();
  test_get<cuda::vector_type_t<T, 4>, T, 4>();
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

  test_get<cuda::dim3, unsigned int, 3>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
