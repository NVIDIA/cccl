//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr auto __unsigned_abs(T value) -> cuda::std::make_unsigned_t<T>
{
  using unsigned_t = cuda::std::make_unsigned_t<T>;
  if constexpr (cuda::std::is_signed_v<T>)
  {
    return value < T{0} ? static_cast<unsigned_t>(-(value + T{1})) + unsigned_t{1} : static_cast<unsigned_t>(value);
  }
  else
  {
    return static_cast<unsigned_t>(value);
  }
}

template <typename T>
__host__ __device__ constexpr auto __min_abs() -> cuda::std::make_unsigned_t<T>
{
  using unsigned_t = cuda::std::make_unsigned_t<T>;
  if constexpr (cuda::std::is_signed_v<T>)
  {
    return static_cast<unsigned_t>(-(cuda::std::numeric_limits<T>::min() + T{1})) + unsigned_t{1};
  }
  else
  {
    return unsigned_t{0};
  }
}

template <typename Result, typename Lhs, typename Rhs>
__host__ __device__ constexpr cuda::overflow_result<Result> __expected_div_result(Lhs lhs, Rhs rhs)
{
  using common_t         = cuda::std::common_type_t<Lhs, Rhs>;
  using common_unsigned_t = cuda::std::make_unsigned_t<common_t>;

  const common_t lhs_common = static_cast<common_t>(lhs);
  const common_t rhs_common = static_cast<common_t>(rhs);

  if (rhs_common == common_t{0})
  {
    return {Result{}, true};
  }

  const common_unsigned_t lhs_abs = __unsigned_abs(lhs_common);
  const common_unsigned_t rhs_abs = __unsigned_abs(rhs_common);

  if (rhs_abs == common_unsigned_t{0})
  {
    return {Result{}, true};
  }

  const common_unsigned_t quotient_abs = lhs_abs / rhs_abs;
  const bool negative_result           = (lhs_common < common_t{0}) ^ (rhs_common < common_t{0});

  using result_unsigned_t = cuda::std::make_unsigned_t<Result>;
  using wider_unsigned_t  = cuda::std::conditional_t<(sizeof(common_unsigned_t) >= sizeof(result_unsigned_t)),
                                                     common_unsigned_t,
                                                     result_unsigned_t>;

  const wider_unsigned_t quotient_w = static_cast<wider_unsigned_t>(quotient_abs);

  if (negative_result)
  {
    if constexpr (!cuda::std::is_signed_v<Result>)
    {
      return {Result{}, true};
    }
    else
    {
      const wider_unsigned_t min_abs_w = static_cast<wider_unsigned_t>(__min_abs<Result>());
      if (quotient_w > min_abs_w)
      {
        return {Result{}, true};
      }
      if (quotient_w == min_abs_w)
      {
        return {cuda::std::numeric_limits<Result>::min(), false};
      }

      const result_unsigned_t magnitude_unsigned = static_cast<result_unsigned_t>(quotient_abs);
      const Result magnitude                      = static_cast<Result>(magnitude_unsigned);
      return {static_cast<Result>(-magnitude), false};
    }
  }
  else
  {
    const wider_unsigned_t max_w = static_cast<wider_unsigned_t>(
      static_cast<result_unsigned_t>(cuda::std::numeric_limits<Result>::max()));
    if (quotient_w > max_w)
    {
      return {Result{}, true};
    }

    const result_unsigned_t magnitude_unsigned = static_cast<result_unsigned_t>(quotient_abs);
    return {static_cast<Result>(magnitude_unsigned), false};
  }
}

template <typename Result, typename Lhs, typename Rhs>
__host__ __device__ constexpr void __test_division_case(const Lhs lhs, const Rhs rhs)
{
  const auto expected = __expected_div_result<Result>(lhs, rhs);

  {
    const auto result = cuda::div_overflow<Result>(lhs, rhs);
    if (!expected.overflow || cuda::std::is_unsigned_v<Result>)
    {
      assert(result.value == expected.value);
    }
    assert(result.overflow == expected.overflow);
  }

  {
    Result result_value{};
    const bool overflow = cuda::div_overflow<Result>(result_value, lhs, rhs);
    if (!expected.overflow || cuda::std::is_unsigned_v<Result>)
    {
      assert(result_value == expected.value);
    }
    assert(overflow == expected.overflow);
  }
}

template <typename Result, typename Lhs, typename Rhs>
__host__ __device__ constexpr void __test_type_impl()
{
  using cuda::std::common_type_t;
  using cuda::std::declval;
  using cuda::std::is_same_v;
  using cuda::std::is_signed_v;

  static_assert(is_same_v<decltype(cuda::div_overflow<Result>(Lhs{}, Rhs{})), cuda::overflow_result<Result>>);
  static_assert(noexcept(cuda::div_overflow<Result>(Lhs{}, Rhs{})));

  static_assert(is_same_v<decltype(cuda::div_overflow<Result>(declval<Result&>(), Lhs{}, Rhs{})), bool>);
  static_assert(noexcept(cuda::div_overflow<Result>(declval<Result&>(), Lhs{}, Rhs{})));

  constexpr Lhs lhs_min = cuda::std::numeric_limits<Lhs>::min();
  constexpr Lhs lhs_max = cuda::std::numeric_limits<Lhs>::max();
  constexpr Rhs rhs_min = cuda::std::numeric_limits<Rhs>::min();
  constexpr Rhs rhs_max = cuda::std::numeric_limits<Rhs>::max();

  __test_division_case<Result>(Lhs{}, Rhs{1});
  __test_division_case<Result>(Lhs{1}, Rhs{1});
  __test_division_case<Result>(Lhs{1}, Rhs{0});
  __test_division_case<Result>(Lhs{}, Rhs{});

  __test_division_case<Result>(lhs_max, Rhs{1});
  __test_division_case<Result>(lhs_min, Rhs{1});

  if constexpr (cuda::std::cmp_greater_equal(rhs_max, Rhs{2}))
  {
    __test_division_case<Result>(Lhs{1}, Rhs{2});
    __test_division_case<Result>(lhs_max, Rhs{2});
    __test_division_case<Result>(lhs_min, Rhs{2});
  }

  if constexpr (is_signed_v<Lhs>)
  {
    __test_division_case<Result>(Lhs{-1}, Rhs{1});
  }

  if constexpr (is_signed_v<Rhs>)
  {
    __test_division_case<Result>(Lhs{1}, Rhs{-1});
  }

  if constexpr (is_signed_v<Lhs> && is_signed_v<Rhs>)
  {
    __test_division_case<Result>(Lhs{-1}, Rhs{-1});
    __test_division_case<Result>(lhs_max, Rhs{-1});
  }

  if constexpr (rhs_max != Rhs{0})
  {
    __test_division_case<Result>(Lhs{}, rhs_max);
    __test_division_case<Result>(lhs_max, rhs_max);
  }

  if constexpr (rhs_min != Rhs{0})
  {
    __test_division_case<Result>(Lhs{1}, rhs_min);
    __test_division_case<Result>(lhs_max, rhs_min);
  }

  if constexpr (lhs_min != Lhs{})
  {
    if constexpr (rhs_min != Rhs{0})
    {
      __test_division_case<Result>(lhs_min, rhs_min);
    }
    __test_division_case<Result>(lhs_min, rhs_max == Rhs{0} ? Rhs{1} : rhs_max);
  }

  if constexpr (is_signed_v<Lhs> && is_signed_v<Rhs> && is_same_v<Lhs, Rhs> && is_same_v<Lhs, Result>)
  {
    __test_division_case<Result>(lhs_min, Rhs{-1});
  }
}

template <typename Lhs, typename Rhs, typename Result>
__host__ __device__ constexpr void __test_type()
{
  __test_type_impl<Result, Lhs, Rhs>();
}

template <typename Lhs, typename Rhs>
__host__ __device__ constexpr void __test_type()
{
  __test_type<Lhs, Rhs, cuda::std::common_type_t<Lhs, Rhs>>();
  __test_type<Lhs, Rhs, unsigned>();
  __test_type<Lhs, Rhs, int>();
#if _CCCL_COMPILER(GCC)
  __test_type<Lhs, Rhs, long long>();
  __test_type<Lhs, Rhs, unsigned long long>();
  __test_type<Lhs, Rhs, signed char>();
  __test_type<Lhs, Rhs, unsigned char>();
#endif
}

template <typename T>
__host__ __device__ constexpr void __test_type()
{
  __test_type<T, signed char>();
  __test_type<T, unsigned char>();
  __test_type<T, short>();
  __test_type<T, unsigned short>();
  __test_type<T, int>();
  __test_type<T, unsigned int>();
  __test_type<T, long>();
  __test_type<T, unsigned long>();
  __test_type<T, long long>();
  __test_type<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  __test_type<T, __int128_t>();
  __test_type<T, __uint128_t>();
#endif
}

__host__ __device__ constexpr bool __test()
{
  __test_type<signed char>();
  __test_type<unsigned char>();
  __test_type<short>();
  __test_type<unsigned short>();
  __test_type<int>();
  __test_type<unsigned int>();
  __test_type<long>();
  __test_type<unsigned long>();
  __test_type<long long>();
  __test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  __test_type<__int128_t>();
  __test_type<__uint128_t>();
#endif
  return true;
}

int main(int, char**)
{
  __test();
  static_assert(__test());
  return 0;
}


