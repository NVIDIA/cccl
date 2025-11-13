//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__hierarchy_>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T, T... Vs>
__host__ __device__ constexpr cuda::hierarchy_query_result<T, sizeof...(Vs)>
make_instance(cuda::std::integer_sequence<T, Vs...>)
{
  return cuda::hierarchy_query_result<T, sizeof...(Vs)>{Vs...};
}

template <class T, cuda::std::size_t N, class Vec>
__host__ __device__ constexpr void compare_vector(cuda::hierarchy_query_result<T, N> v, Vec vec)
{
  assert(vec.x == v.x);
  if constexpr (N >= 2)
  {
    assert(vec.y == v.y);
  }
  if constexpr (N >= 3)
  {
    assert(vec.z == v.z);
  }
  if constexpr (N >= 4)
  {
    assert(vec.w == v.w);
  }
}

template <class T, cuda::std::size_t N>
__host__ __device__ constexpr void test()
{
  using Type = cuda::hierarchy_query_result<T, N>;

  // 1. Test value_type
  static_assert(cuda::std::is_same_v<typename Type::value_type, T>);

  // 2. Test constructors
  static_assert(cuda::std::is_trivially_default_constructible_v<Type>);
  static_assert(cuda::std::is_trivially_copyable_v<Type>);

  // 3. Test public members
  {
    auto v = make_instance(cuda::std::make_integer_sequence<T, N>{});
    if constexpr (N >= 1)
    {
      assert(v.x == static_cast<T>(N) - 1);
    }
    if constexpr (N >= 2)
    {
      assert(v.y == static_cast<T>(N) - 2);
    }
    if constexpr (N >= 3)
    {
      assert(v.z == static_cast<T>(N) - 3);
    }
  }

  // 4. Test operator[] const
  static_assert(cuda::std::is_same_v<const T&, decltype(cuda::std::declval<const Type>()[cuda::std::size_t{}])>);
  static_assert(noexcept(cuda::std::declval<const Type>()[cuda::std::size_t{}]));
  if constexpr (N > 0)
  {
    const auto v = make_instance(cuda::std::make_integer_sequence<T, N>{});
    for (cuda::std::size_t i = 0; i < N; ++i)
    {
      assert(v[i] == static_cast<T>(i));
    }
  }

  // 5. Test operator[]
  static_assert(cuda::std::is_same_v<T&, decltype(cuda::std::declval<Type>()[cuda::std::size_t{}])>);
  static_assert(noexcept(cuda::std::declval<Type>()[cuda::std::size_t{}]));
  if constexpr (N > 0)
  {
    auto v = make_instance(cuda::std::make_integer_sequence<T, N>{});
    for (cuda::std::size_t i = 0; i < N; ++i)
    {
      assert(v[i] == static_cast<T>(i));
    }
  }

  // // 6. Test operator vector-type
  // static_assert((N < 1 || N > 4) || cuda::std::is_nothrow_convertible_v<Type, cuda::__vector_type_t<T, N>>);
  // if constexpr (N > 0 && N <= 4)
  // {
  //   using Vec    = cuda::__vector_type_t<T, N>;
  //   const auto v = make_instance(cuda::std::make_integer_sequence<T, N>{});
  //   Vec vec      = v;
  //   compare_vector(v, vec);
  // }

  // // 7. Test operator dim3 for unsigned type of size 3
  // static_assert((!cuda::std::is_same_v<T, unsigned> || N != 3) || cuda::std::is_nothrow_convertible_v<Type, dim3>);
  // if constexpr (cuda::std::is_same_v<T, unsigned> && N == 3)
  // {
  //   const auto v = make_instance(cuda::std::make_integer_sequence<T, N>{});
  //   dim3 vec     = v;
  //   compare_vector(v, vec);
  // }
}

template <class T>
__host__ __device__ constexpr void test()
{
  test<T, 0>();
  test<T, 1>();
  test<T, 2>();
  test<T, 3>();
  test<T, 4>();
  test<T, 6>();
}

__host__ __device__ constexpr bool test()
{
  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128();

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
