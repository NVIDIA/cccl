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

template <class VT, class T>
__host__ __device__ constexpr void test_structured_bindings_1()
{
  VT v = test_make_vector<VT>();

  {
    auto [x]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    assert(x == T{0});
  }
  {
    auto& [x]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    assert(x == T{0});
  }
}

template <class VT, class T>
__host__ __device__ constexpr void test_structured_bindings_2()
{
  VT v = test_make_vector<VT>();

  {
    auto [x, y]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    static_assert(cuda::std::is_same_v<T, decltype(y)>);
    assert(x == T{0});
    assert(y == T{1});
  }
  {
    auto& [x, y]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    static_assert(cuda::std::is_same_v<T, decltype(y)>);
    assert(x == T{0});
    assert(y == T{1});
  }
}

template <class VT, class T>
__host__ __device__ constexpr void test_structured_bindings_3()
{
  VT v = test_make_vector<VT>();

  {
    auto [x, y, z]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    static_assert(cuda::std::is_same_v<T, decltype(y)>);
    static_assert(cuda::std::is_same_v<T, decltype(z)>);
    assert(x == T{0});
    assert(y == T{1});
    assert(z == T{2});
  }
  {
    auto& [x, y, z]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    static_assert(cuda::std::is_same_v<T, decltype(y)>);
    static_assert(cuda::std::is_same_v<T, decltype(z)>);
    assert(x == T{0});
    assert(y == T{1});
    assert(z == T{2});
  }
}

template <class VT, class T>
__host__ __device__ constexpr void test_structured_bindings_4()
{
  VT v = test_make_vector<VT>();

  {
    auto [x, y, z, w]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    static_assert(cuda::std::is_same_v<T, decltype(y)>);
    static_assert(cuda::std::is_same_v<T, decltype(z)>);
    static_assert(cuda::std::is_same_v<T, decltype(w)>);
    assert(x == T{0});
    assert(y == T{1});
    assert(z == T{2});
    assert(w == T{3});
  }
  {
    auto& [x, y, z, w]{v};
    static_assert(cuda::std::is_same_v<T, decltype(x)>);
    static_assert(cuda::std::is_same_v<T, decltype(y)>);
    static_assert(cuda::std::is_same_v<T, decltype(z)>);
    static_assert(cuda::std::is_same_v<T, decltype(w)>);
    assert(x == T{0});
    assert(y == T{1});
    assert(z == T{2});
    assert(w == T{3});
  }
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_structured_bindings_1<cuda::vector_type_t<T, 1>, T>();
  test_structured_bindings_2<cuda::vector_type_t<T, 2>, T>();
  test_structured_bindings_3<cuda::vector_type_t<T, 3>, T>();
  test_structured_bindings_4<cuda::vector_type_t<T, 4>, T>();
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

  test_structured_bindings_3<cuda::dim3, unsigned int>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
