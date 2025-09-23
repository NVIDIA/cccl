//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// UNSUPPORTED: c++17

// <cuda/complex>

#include <cuda/__complex_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#if !_CCCL_COMPILER(NVRTC)
#  include <tuple>
#endif // !_CCCL_COMPILER(NVRTC)

// vvv MyPair & tuple protocol for MyPair vvv

namespace my_namespace
{
template <class A, class B>
struct MyPair
{
  A first;
  B second;
};

template <cuda::std::size_t I, class A, class B>
[[nodiscard]] __host__ __device__ constexpr auto& get(MyPair<A, B>& p)
{
  if constexpr (I == 0)
  {
    return p.first;
  }
  else
  {
    return p.second;
  }
}

template <cuda::std::size_t I, class A, class B>
[[nodiscard]] __host__ __device__ constexpr auto&& get(MyPair<A, B>&& p)
{
  if constexpr (I == 0)
  {
    return cuda::std::move(p.first);
  }
  else
  {
    return cuda::std::move(p.second);
  }
}

template <cuda::std::size_t I, class A, class B>
[[nodiscard]] __host__ __device__ constexpr const auto& get(const MyPair<A, B>& p)
{
  if constexpr (I == 0)
  {
    return p.first;
  }
  else
  {
    return p.second;
  }
}

template <cuda::std::size_t I, class A, class B>
[[nodiscard]] __host__ __device__ constexpr const auto&& get(const MyPair<A, B>&& p)
{
  if constexpr (I == 0)
  {
    return cuda::std::move(p.first);
  }
  else
  {
    return cuda::std::move(p.second);
  }
}
} // namespace my_namespace

template <class A, class B>
struct cuda::std::tuple_size<my_namespace::MyPair<A, B>> : cuda::std::integral_constant<cuda::std::size_t, 2>
{};

template <class A, class B>
struct cuda::std::tuple_element<0, my_namespace::MyPair<A, B>>
{
  using type = A;
};

template <class A, class B>
struct cuda::std::tuple_element<1, my_namespace::MyPair<A, B>>
{
  using type = B;
};

// ^^^ MyPair & tuple protocol for MyPair ^^^

template <class T, bool is_noexcept, class TupleLike>
__host__ __device__ constexpr void test_constructor_from_tuple_like(const TupleLike& tuple_like)
{
  using C = cuda::complex<T>;

  // 1. Test that cuda::complex<T> is constructible from TupleLike
  static_assert(cuda::std::is_constructible_v<C, const TupleLike&>);

  // 2. Test that the constructor is noexcept
  static_assert(noexcept(C{tuple_like}) == is_noexcept);

  // 3. Test that the constructor is explicit
  static_assert(!cuda::std::is_convertible_v<const TupleLike&, C>);

  const C v{tuple_like};
  assert(v.real() == T(1));
  assert(v.imag() == T(2));
}

template <class T>
__host__ __device__ constexpr void test_constructor_from_tuple_like()
{
  T real_part(1);
  T imag_part(2);

  // 1. Test cuda::std::tuple
  test_constructor_from_tuple_like<T, true>(cuda::std::tuple<T, T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(cuda::std::tuple<const T&, const T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(cuda::std::tuple<T&, T&&>{real_part, T{imag_part}});

  // 2. Test cuda::std::pair
  test_constructor_from_tuple_like<T, true>(cuda::std::pair<T, T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(cuda::std::pair<const T&, const T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(cuda::std::pair<T&, T&&>{real_part, T{imag_part}});

  // 3. Test my_namespace::MyPair
  test_constructor_from_tuple_like<T, false>(my_namespace::MyPair<T, T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, false>(my_namespace::MyPair<const T&, const T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, false>(my_namespace::MyPair<T&, T&&>{real_part, T{imag_part}});
}

__host__ __device__ constexpr bool test()
{
  test_constructor_from_tuple_like<float>();
  test_constructor_from_tuple_like<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_constructor_from_tuple_like<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_constructor_from_tuple_like<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_constructor_from_tuple_like<signed char>();
  test_constructor_from_tuple_like<signed short>();
  test_constructor_from_tuple_like<signed int>();
  test_constructor_from_tuple_like<signed long>();
  test_constructor_from_tuple_like<signed long long>();
#if _CCCL_HAS_INT128()
  test_constructor_from_tuple_like<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_constructor_from_tuple_like<unsigned char>();
  test_constructor_from_tuple_like<unsigned short>();
  test_constructor_from_tuple_like<unsigned int>();
  test_constructor_from_tuple_like<unsigned long>();
  test_constructor_from_tuple_like<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_constructor_from_tuple_like<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

#if !_CCCL_COMPILER(NVRTC)

template <class... Args>
struct cuda::std::tuple_size<std::tuple<Args...>> : cuda::std::integral_constant<cuda::std::size_t, sizeof...(Args)>
{};

template <std::size_t I, class... Args>
struct cuda::std::tuple_element<I, std::tuple<Args...>> : std::tuple_element<I, cuda::std::tuple<Args...>>
{};

template <class A, class B>
struct cuda::std::tuple_size<std::pair<A, B>> : cuda::std::integral_constant<cuda::std::size_t, 2>
{};

template <cuda::std::size_t I, class A, class B>
struct cuda::std::tuple_element<I, std::pair<A, B>> : cuda::std::conditional<I == 0, A, B>
{};

template <class T>
void test_constructor_from_host_tuple_like()
{
  T real_part(1);
  T imag_part(2);

  // 1. Test std::tuple
  test_constructor_from_tuple_like<T, true>(std::tuple<T, T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(std::tuple<const T&, const T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(std::tuple<T&, T&&>(real_part, T{imag_part}));

  // 2. Test std::pair
  test_constructor_from_tuple_like<T, true>(std::pair<T, T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(std::pair<const T&, const T>{real_part, imag_part});
  test_constructor_from_tuple_like<T, true>(std::pair<T&, T&&>(real_part, T{imag_part}));
}

void test_host_types()
{
  test_constructor_from_host_tuple_like<float>();
  test_constructor_from_host_tuple_like<double>();
#  if _CCCL_HAS_LONG_DOUBLE()
  test_constructor_from_host_tuple_like<long double>();
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_FLOAT128()
  test_constructor_from_host_tuple_like<__float128>();
#  endif // _CCCL_HAS_FLOAT128()

  test_constructor_from_host_tuple_like<signed char>();
  test_constructor_from_host_tuple_like<signed short>();
  test_constructor_from_host_tuple_like<signed int>();
  test_constructor_from_host_tuple_like<signed long>();
  test_constructor_from_host_tuple_like<signed long long>();
#  if _CCCL_HAS_INT128()
  test_constructor_from_host_tuple_like<__int128_t>();
#  endif // _CCCL_HAS_INT128()

  test_constructor_from_host_tuple_like<unsigned char>();
  test_constructor_from_host_tuple_like<unsigned short>();
  test_constructor_from_host_tuple_like<unsigned int>();
  test_constructor_from_host_tuple_like<unsigned long>();
  test_constructor_from_host_tuple_like<unsigned long long>();
#  if _CCCL_HAS_INT128()
  test_constructor_from_host_tuple_like<__uint128_t>();
#  endif // _CCCL_HAS_INT128()
}

#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  test();
  static_assert(test());
#if !_CCCL_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test_host_types();))
#endif // !_CCCL_COMPILER(NVRTC)
  return 0;
}
