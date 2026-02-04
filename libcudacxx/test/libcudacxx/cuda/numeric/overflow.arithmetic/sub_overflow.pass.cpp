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
__host__ __device__ constexpr void test_sub_overflow(Lhs lhs, Rhs rhs, bool overflow, bool skip_special_cases = false)
{
  // test overflow_result<Result> sub_overflow(Lhs lhs, Rhs rhs) overload
  {
    const auto result = cuda::sub_overflow<Result>(lhs, rhs);
    // overflow result is well-defined only for unsigned types
    if ((!overflow && !skip_special_cases) || cuda::std::is_unsigned_v<Result>)
    {
      assert(result.value == static_cast<Result>(static_cast<Result>(lhs) - static_cast<Result>(rhs)));
    }
    assert(result.overflow == overflow);
  }
  // test bool sub_overflow(Lhs lhs, Rhs rhs, Result & result) overload
  {
    Result result{};
    bool has_overflow = cuda::sub_overflow<Result>(result, lhs, rhs);
    // overflow result is well-defined only for unsigned types
    if ((!overflow && !skip_special_cases) || cuda::std::is_unsigned_v<Result>)
    {
      assert(result == static_cast<Result>(static_cast<Result>(lhs) - static_cast<Result>(rhs)));
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
  static_assert(is_same_v<decltype(cuda::sub_overflow<Result>(Lhs{}, Rhs{})), cuda::overflow_result<Result>>);
  [[maybe_unused]] constexpr auto lhs_min        = cuda::std::numeric_limits<Lhs>::min();
  [[maybe_unused]] constexpr auto lhs_max        = cuda::std::numeric_limits<Lhs>::max();
  [[maybe_unused]] constexpr auto rhs_min        = cuda::std::numeric_limits<Rhs>::min();
  [[maybe_unused]] constexpr auto rhs_max        = cuda::std::numeric_limits<Rhs>::max();
  [[maybe_unused]] constexpr auto result_min     = cuda::std::numeric_limits<Result>::min();
  [[maybe_unused]] constexpr auto result_max     = cuda::std::numeric_limits<Result>::max();
  [[maybe_unused]] constexpr auto neg_result_min = cuda::uabs(result_min);
  [[maybe_unused]] constexpr auto neg_rhs_min    = cuda::uabs(rhs_min);
  //--------------------------------------------------------------------------------------------------------------------
  // trivial cases
  // 1. 0 - 0 -> should never overflow
  test_sub_overflow<Result>(Lhs{}, Rhs{}, false);

  // 2. 1 - 1 -> should never overflow
  test_sub_overflow<Result>(Lhs{1}, Rhs{1}, false);

  // 3. 0 - 1 -> should overflow if the destination type is unsigned
  if constexpr (is_signed_v<Rhs>)
  {
    test_sub_overflow<Result>(Lhs{}, Rhs{1}, is_unsigned_v<Result>);
  }
  // 4. 1 - 0 -> should never overflow
  test_sub_overflow<Result>(Lhs{1}, Rhs{0}, false);
  //--------------------------------------------------------------------------------------------------------------------
  // max cases
  // 5. max - 0 -> max >= result_max?
  test_sub_overflow<Result>(lhs_max, Rhs{}, cuda::std::cmp_greater(lhs_max, result_max));

  // 6. max - 1 -> max - 1 >= result_max?
  test_sub_overflow<Result>(lhs_max, Rhs{1}, cuda::std::cmp_greater(Lhs{lhs_max - 1}, result_max));

  // 7. 0 - max -> -rhs_max > result_min? -> rhs_max < -result_min?
  test_sub_overflow<Result>(Lhs{}, rhs_max, cuda::std::cmp_greater(rhs_max, neg_result_min));

  // 8. 1 - max -> (rhs_max - 1) < -result_min?
  test_sub_overflow<Result>(Lhs{1}, rhs_max, cuda::std::cmp_greater(Rhs{rhs_max - 1}, neg_result_min));

  // 9. max - max
  if constexpr (lhs_max >= rhs_max) // positive result
  {
    test_sub_overflow<Result>(lhs_max, rhs_max, cuda::std::cmp_greater(lhs_max - rhs_max, result_max));
  }
  else if constexpr (is_unsigned_v<Result>) // rhs_max > lhs_max -> negative result
  {
    test_sub_overflow<Result>(lhs_max, rhs_max, true);
  }
  else // rhs_max > lhs_max, negative result: lhs_max - rhs_max < result_min -> lhs_max < result_min + rhs_max
  {
    // *** very special case ***: test case cannot be validated in a constexpr context
    // example: int - unsigned = int
    // the expression 'static_cast<Result>(static_cast<Result>(lhs) - static_cast<Result>(rhs))' translate to:
    // INT_MAX - UINT_MAX = INT_MIN -> no overflow, but
    // INT_MIN - 1 -> is less than INT_MIN and constexpr fails to compile
    bool skip_special_case = sizeof(Lhs) == sizeof(Rhs) && sizeof(Rhs) == sizeof(Result) && is_signed_v<Lhs>
                          && is_unsigned_v<Rhs> && cuda::std::__cccl_default_is_constant_evaluated();
    test_sub_overflow<Result>(
      lhs_max, rhs_max, cuda::std::cmp_greater(rhs_max - lhs_max, neg_result_min), skip_special_case);
  }
  //--------------------------------------------------------------------------------------------------------------------
  // min cases
  if constexpr (is_signed_v<Rhs>) // if Rhs is unsigned, rhs_min = 0, already handled above
  {
    // 10. 0 - min -> -min > result_max? -> min < -result_max?
    if constexpr (is_same_v<Rhs, Result>) // e.g. 0 - INT_MIN -> overflow
    {
      test_sub_overflow<Result>(Lhs{}, rhs_min, true);
    }
    else // Result is unsigned -> 0 - min -> -min > result_max?
    {
      test_sub_overflow<Result>(Lhs{}, rhs_min, cuda::std::cmp_greater(neg_rhs_min, result_max));
    }

    // 11. (-1) - min -> -min < result_min + 1 -> min > -(result_min + 1)?
    if constexpr (is_signed_v<Lhs>)
    {
      test_sub_overflow<Result>(Lhs{-1}, rhs_min, cuda::std::cmp_greater(Rhs{-Rhs{rhs_min + 1}}, result_max));
    }
  }

  if constexpr (is_signed_v<Lhs>) // otherwise lhs_min = 0, already handled above
  {
    // 12. min - 0 -> should overflow only if the destination type is too small
    test_sub_overflow<Result>(lhs_min, Rhs{}, cuda::std::cmp_less(lhs_min, result_min));

    // 13. min - (-1) -> min + 1 < result_min?
    if constexpr (is_signed_v<Rhs>)
    {
      test_sub_overflow<Result>(lhs_min, Rhs{-1}, cuda::std::cmp_less(Lhs{lhs_min + 1}, Result{result_min}));
    }

    // 14. min - 1
    test_sub_overflow<Result>(lhs_min, Rhs{1}, cuda::std::cmp_less(lhs_min, Result{result_min + 1}));
  }

  // 15. min - min
  if constexpr (is_signed_v<Lhs> && is_signed_v<Rhs>)
  {
    if constexpr (cuda::std::cmp_equal(lhs_min, rhs_min)) // lhs_min == rhs_min
    {
      test_sub_overflow<Result>(lhs_min, rhs_min, false);
    }
    else if constexpr (cuda::std::cmp_less(lhs_min, rhs_min)) // lhs_min < rhs_min
    {
      // negative result -> lhs_min - rhs_min < result_min?
      using SignedCommon = cuda::std::make_signed_t<cuda::std::common_type_t<Lhs, Rhs>>;
      auto sub           = cuda::__sub_as_unsigned<SignedCommon>(lhs_min, rhs_min);
      test_sub_overflow<Result>(lhs_min, rhs_min, cuda::std::cmp_less(sub, result_min));
    }
    else // lhs_min > rhs_min -> positive result -> lhs_min - rhs_min > result_max + rhs_min?
    {
      using UnsignedCommon = cuda::std::make_unsigned_t<cuda::std::common_type_t<Lhs, Rhs>>;
      auto sub             = cuda::__sub_as_unsigned<UnsignedCommon>(lhs_min, rhs_min);
      test_sub_overflow<Result>(lhs_min, rhs_min, cuda::std::cmp_greater(sub, Result{result_max}));
    }
  }
}

template <typename T, typename R>
__host__ __device__ constexpr void test_type()
{
  test_type<T, R, cuda::std::common_type_t<T, R>>();
  test_type<T, R, unsigned>();
  test_type<T, R, int>();
  // instantiation of all tests is very expensive. clang hits "constexpr evaluation hit maximum step limit"
#if !_CCCL_COMPILER(CLANG)
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
  using cuda::std::is_same_v;
  static_assert(noexcept(cuda::sub_overflow(int{}, int{})));
  static_assert(noexcept(cuda::sub_overflow<unsigned>(cuda::std::declval<unsigned&>(), int{}, int{})));
  static_assert(is_same_v<decltype(cuda::sub_overflow<unsigned>(cuda::std::declval<unsigned&>(), int{}, int{})), bool>);

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
