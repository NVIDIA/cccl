//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/numeric>

// template<class R, class T>
// constexpr overflow_result<R> saturate_overflow_cast(T x) noexcept;                     // freestanding

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

using IMIN = signed char;
using UMIN = signed char;

#if _CCCL_HAS_INT128()
using IMAX = __int128_t;
using UMAX = __uint128_t;
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
using IMAX = signed long long;
using UMAX = unsigned long long;
#endif // ^^^ !_CCCL_HAS_INT128() ^^^

template <class Ret, class T>
__host__ __device__ constexpr void test(T x, Ret res, bool of, int zero_value)
{
  const auto [result, overflow] = cuda::saturate_overflow_cast<Ret>(static_cast<T>(zero_value + x));
  assert(result == res);
  assert(overflow == of);
}

template <class S>
__host__ __device__ constexpr bool test_type(int zero_value)
{
  static_assert(cuda::std::is_integral_v<S> && cuda::std::is_signed_v<S>);

  using U = cuda::std::make_unsigned_t<S>;

  constexpr auto small_smax  = cuda::std::numeric_limits<IMIN>::max();
  constexpr auto small_szero = IMIN{0};
  constexpr auto small_smin  = cuda::std::numeric_limits<IMIN>::min();
  constexpr auto small_umax  = cuda::std::numeric_limits<UMIN>::max();
  constexpr auto small_uzero = UMIN{0};

  constexpr auto big_smax  = cuda::std::numeric_limits<IMAX>::max();
  constexpr auto big_szero = IMAX{0};
  constexpr auto big_smin  = cuda::std::numeric_limits<IMAX>::min();
  constexpr auto big_umax  = cuda::std::numeric_limits<UMAX>::max();
  constexpr auto big_uzero = UMAX{0};

  constexpr S smax  = cuda::std::numeric_limits<S>::max();
  constexpr S szero = S{0};
  constexpr S smin  = cuda::std::numeric_limits<S>::min();
  constexpr U umax  = cuda::std::numeric_limits<U>::max();
  constexpr U uzero = U{0};

  // test signed

  static_assert(cuda::std::is_same_v<cuda::overflow_result<S>, decltype(cuda::saturate_overflow_cast<S>(small_smax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<S>(small_smax)));
  test<S>(small_smin, static_cast<S>(small_smin), false, zero_value);
  test<S>(small_szero, szero, false, zero_value);
  test<S>(small_smax, static_cast<S>(small_smax), false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<S>, decltype(cuda::saturate_overflow_cast<S>(small_umax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<S>(small_umax)));
  test<S>(small_uzero, szero, false, zero_value);
  test<S>(small_umax, (sizeof(S) == sizeof(IMIN)) ? small_smax : static_cast<S>(small_umax), false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<S>, decltype(cuda::saturate_overflow_cast<S>(smax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<S>(smax)));
  test<S>(smin, smin, false, zero_value);
  test<S>(szero, szero, false, zero_value);
  test<S>(smax, smax, false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<S>, decltype(cuda::saturate_overflow_cast<S>(umax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<S>(umax)));
  test<S>(uzero, szero, false, zero_value);
  test<S>(umax, smax, true, zero_value); // saturated

  static_assert(cuda::std::is_same_v<cuda::overflow_result<S>, decltype(cuda::saturate_overflow_cast<S>(big_smax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<S>(big_smax)));
  test<S>(big_smin, smin, sizeof(S) < sizeof(IMAX), zero_value); // saturated
  test<S>(big_szero, szero, false, zero_value);
  test<S>(big_smax, smax, sizeof(S) < sizeof(IMAX), zero_value); // saturated

  static_assert(cuda::std::is_same_v<cuda::overflow_result<S>, decltype(cuda::saturate_overflow_cast<S>(big_umax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<S>(big_umax)));
  test<S>(big_uzero, szero, false, zero_value);
  test<S>(big_umax, smax, true, zero_value); // saturated

  // test unsigned

  static_assert(cuda::std::is_same_v<cuda::overflow_result<U>, decltype(cuda::saturate_overflow_cast<U>(small_smax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<U>(small_smax)));
  test<U>(small_smin, uzero, true, zero_value);
  test<U>(small_szero, uzero, false, zero_value);
  test<U>(small_smax, static_cast<U>(small_smax), false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<U>, decltype(cuda::saturate_overflow_cast<U>(small_umax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<U>(small_umax)));
  test<U>(small_uzero, uzero, false, zero_value);
  test<U>(small_umax, static_cast<U>(small_umax), false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<U>, decltype(cuda::saturate_overflow_cast<U>(smax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<U>(smax)));
  test<U>(smin, uzero, true, zero_value);
  test<U>(szero, uzero, false, zero_value);
  test<U>(smax, static_cast<U>(smax), false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<U>, decltype(cuda::saturate_overflow_cast<U>(umax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<U>(umax)));
  test<U>(uzero, uzero, false, zero_value);
  test<U>(umax, umax, false, zero_value);

  static_assert(cuda::std::is_same_v<cuda::overflow_result<U>, decltype(cuda::saturate_overflow_cast<U>(big_smax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<U>(big_smax)));
  test<U>(big_smin, uzero, true, zero_value); // saturated
  test<U>(big_szero, uzero, false, zero_value);
  test<U>(big_smax,
          (sizeof(U) == sizeof(UMAX)) ? static_cast<U>(smax) : umax,
          sizeof(U) < sizeof(UMAX),
          zero_value); // saturated

  static_assert(cuda::std::is_same_v<cuda::overflow_result<U>, decltype(cuda::saturate_overflow_cast<U>(big_umax))>);
  static_assert(noexcept(cuda::saturate_overflow_cast<U>(big_umax)));
  test<U>(big_uzero, uzero, false, zero_value);
  test<U>(big_umax, umax, sizeof(U) < sizeof(UMAX), zero_value); // saturated

  return true;
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_type<signed char>(zero_value);
  test_type<signed short>(zero_value);
  test_type<signed int>(zero_value);
  test_type<signed long>(zero_value);
  test_type<signed long long>(zero_value);
#if _CCCL_HAS_INT128()
  test_type<__int128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
  static_assert(test(0));
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
  static_assert(test(0));

  return 0;
}
