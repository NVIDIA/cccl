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

template <typename Result, typename Lhs, typename Rhs>
__host__ __device__ constexpr void test_add_overflow(Lhs lhs, Rhs rhs, bool overflow)
{
  // test overflow_result<Result> add_overflow(Lhs lhs, Rhs rhs) overload
  {
    const auto result = cuda::add_overflow<Result>(lhs, rhs);
    // overflow result is well-defined only for unsigned types
    if (!overflow || cuda::std::is_unsigned_v<Result>)
    {
      assert(result.value == static_cast<Result>(static_cast<Result>(lhs) + static_cast<Result>(rhs)));
    }
    assert(result.overflow == overflow);
  }
  // test bool add_overflow(Lhs lhs, Rhs rhs, Result & result) overload
  {
    Result result{};
    bool has_overflow = cuda::add_overflow<Result>(result, lhs, rhs);
    // overflow result is well-defined only for unsigned types
    if (!overflow || cuda::std::is_unsigned_v<Result>)
    {
      assert(result == static_cast<Result>(static_cast<Result>(lhs) + static_cast<Result>(rhs)));
    }
    assert(has_overflow == overflow);
  }
}

template <typename Lhs, typename Rhs, typename Result>
__host__ __device__ constexpr void test_type()
{
  using cuda::std::is_same_v;
  using cuda::std::is_signed_v;
  using cuda::std::is_unsigned_v;
  static_assert(is_same_v<decltype(cuda::add_overflow<Result>(Lhs{}, Rhs{})), cuda::overflow_result<Result>>);
  static_assert(noexcept(cuda::add_overflow(Lhs{}, Rhs{})));

  static_assert(is_same_v<decltype(cuda::add_overflow<Result>(cuda::std::declval<Result&>(), Lhs{}, Rhs{})), bool>);
  static_assert(noexcept(cuda::add_overflow<Result>(cuda::std::declval<Result&>(), Lhs{}, Rhs{})));

  // 1. Adding zeros should never overflow
  test_add_overflow<Result>(Lhs{}, Rhs{}, false);

  // 2. Adding ones should never overflow
  test_add_overflow<Result>(Lhs{1}, Rhs{1}, false);

  // 3. Adding zero and negative one should overflow if the destination type is unsigned
  if constexpr (is_signed_v<Rhs>)
  {
    test_add_overflow<Result>(Lhs{}, Rhs{-1}, is_unsigned_v<Result>);
  }
  constexpr auto lhs_min    = cuda::std::numeric_limits<Lhs>::min();
  constexpr auto lhs_max    = cuda::std::numeric_limits<Lhs>::max();
  constexpr auto rhs_min    = cuda::std::numeric_limits<Rhs>::min();
  constexpr auto rhs_max    = cuda::std::numeric_limits<Rhs>::max();
  constexpr auto result_min = cuda::std::numeric_limits<Result>::min();
  constexpr auto result_max = cuda::std::numeric_limits<Result>::max();
  // 5. Adding max and zero
  test_add_overflow<Result>(lhs_max, Rhs{}, cuda::std::cmp_greater(lhs_max, result_max));

  // 6. Adding zero and max
  test_add_overflow<Result>(Lhs{}, rhs_max, cuda::std::cmp_greater(rhs_max, result_max));

  // 7. Adding one and max
  test_add_overflow<Result>(lhs_max, Rhs{1}, cuda::std::cmp_greater_equal(lhs_max, result_max));

  // 8. Adding max and one
  test_add_overflow<Result>(Lhs{1}, rhs_max, cuda::std::cmp_greater_equal(rhs_max, result_max));

  //  9. Adding max and max
  if constexpr (result_max >= lhs_max)
  {
    test_add_overflow<Result>(lhs_max, rhs_max, cuda::std::cmp_less(result_max - Result{lhs_max}, rhs_max));
  }
  else if constexpr (result_max >= rhs_max)
  {
    test_add_overflow<Result>(lhs_max, rhs_max, cuda::std::cmp_less(result_max - Result{rhs_max}, lhs_max));
  }
  else
  {
    test_add_overflow<Result>(lhs_max, rhs_max, true);
  }

  // 10. Adding min and zero
  test_add_overflow<Result>(lhs_min, Rhs{}, cuda::std::cmp_less(lhs_min, result_min));

  // 11. Adding zero and min
  test_add_overflow<Result>(Lhs{}, rhs_min, cuda::std::cmp_less(rhs_min, result_min));

  // 12. Adding min and minus one
  if constexpr (is_signed_v<Rhs>)
  {
    test_add_overflow<Result>(lhs_min, Rhs{-1}, cuda::std::cmp_less_equal(lhs_min, result_min));
  }
  // 13. Adding minus one and min
  if constexpr (is_signed_v<Lhs>)
  {
    test_add_overflow<Result>(Lhs{-1}, rhs_min, cuda::std::cmp_less_equal(rhs_min, result_min));
  }
  // 14. Adding min and min
  if constexpr (sizeof(Result) >= sizeof(Lhs) && is_signed_v<Result>)
  {
    test_add_overflow<Result>(lhs_min, rhs_min, cuda::std::cmp_greater(result_min - Result{lhs_min}, rhs_min));
  }
  else if constexpr (sizeof(Result) >= sizeof(Rhs) && is_signed_v<Result>)
  {
    test_add_overflow<Result>(lhs_min, rhs_min, cuda::std::cmp_greater(result_min - Result{rhs_min}, lhs_min));
  }
  else // Result is unsigned OR Result < Lhs and < Rhs
  {
    test_add_overflow<Result>(lhs_min, rhs_min, is_signed_v<Lhs> || is_signed_v<Rhs>);
  }
}

template <typename T, typename R>
__host__ __device__ constexpr void test_type()
{
  test_type<T, R, cuda::std::common_type_t<T, R>>();
  test_type<T, R, unsigned>();
  test_type<T, R, int>();
  // instantiation of all tests is very expensive. clang hits "constexpr evaluation hit maximum step limit"
#if _CCCL_COMPILER(GCC)
  test_type<T, R, long long>();
  test_type<T, R, unsigned long long>();
  test_type<T, R, signed char>();
  test_type<T, R, unsigned char>();
#endif // _CCCL_COMPILER_GCC() && !_CCCL_COMPILER_CLANG()
}

template <typename T>
__host__ __device__ constexpr void test_type()
{
  test_type<T, signed char>();
  test_type<T, unsigned char>();
  test_type<T, short>();
  test_type<T, unsigned short>();
  test_type<T, int>();
  test_type<T, unsigned int>();
  test_type<T, long>();
  test_type<T, unsigned long>();
  test_type<T, long long>();
  test_type<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<T, __int128_t>();
  test_type<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
