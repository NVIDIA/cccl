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
__host__ __device__ constexpr void
test_div_overflow(const Lhs lhs, const Rhs rhs, bool overflow, bool special_case = false, Result expected = {})
{
  // test overflow_result<Result> div_overflow(Lhs lhs, Rhs rhs) overload
  //   skip_special_case is used to skip special cases that are not valid in a constexpr context, e.g. INT_MIN / -1
  //   but the result fits in the Result type
  {
    const auto result = cuda::div_overflow<Result>(lhs, rhs);
    if (!overflow)
    {
      if (special_case)
      {
        assert(result.value == expected);
      }
      else
      {
        assert(result.value == static_cast<Result>(static_cast<Result>(lhs) / static_cast<Result>(rhs)));
      }
    }
    assert(result.overflow == overflow);
  }
  // test bool div_overflow(Result& result, Lhs lhs, Rhs rhs) overload
  {
    Result result{};
    const bool has_overflow = cuda::div_overflow<Result>(result, lhs, rhs);
    if (!overflow)
    {
      if (special_case)
      {
        assert(result == expected);
      }
      else
      {
        assert(result == static_cast<Result>(static_cast<Result>(lhs) / static_cast<Result>(rhs)));
      }
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
  static_assert(is_same_v<decltype(cuda::div_overflow<Result>(Lhs{}, Rhs{})), cuda::overflow_result<Result>>);
  [[maybe_unused]] constexpr auto lhs_min     = cuda::std::numeric_limits<Lhs>::min();
  [[maybe_unused]] constexpr auto lhs_max     = cuda::std::numeric_limits<Lhs>::max();
  [[maybe_unused]] constexpr auto result_max  = cuda::std::numeric_limits<Result>::max();
  [[maybe_unused]] constexpr auto result_min  = cuda::std::numeric_limits<Result>::min();
  [[maybe_unused]] constexpr auto neg_lhs_min = cuda::uabs(lhs_min);
  //--------------------------------------------------------------------------------------------------------------------
  //  trivial cases
  //  1. 1 / 0 -> should overflow
  test_div_overflow<Result>(Lhs{1}, Rhs{0}, true);

  // 2. 0 / 0 -> should overflow
  test_div_overflow<Result>(Lhs{0}, Rhs{0}, true);

  // 3. 0 / 1 -> should not overflow
  test_div_overflow<Result>(Lhs{0}, Rhs{1}, false);

  // 4. 1 / 1 -> should not overflow
  test_div_overflow<Result>(Lhs{1}, Rhs{1}, false);

  // nvcc 12.0 doesn't support this case in a constexpr context
#if _CCCL_CUDA_COMPILER(NVCC, !=, 12, 0)
  // 5. 1 / -1 -> should overflow if the destination type is unsigned
  if constexpr (is_signed_v<Rhs>)
  {
    test_div_overflow<Result>(Lhs{1}, Rhs{-1}, is_unsigned_v<Result>);
  }
  // 6. -1 / 1 -> should overflow if the destination type is unsigned

  if constexpr (is_signed_v<Lhs>)
  {
    test_div_overflow<Result>(Lhs{-1}, Rhs{1}, is_unsigned_v<Result>);
  }
#endif // _CCCL_CUDA_COMPILER(NVCC, !=, 12, 0)

  // 7. 0 / -1
  if constexpr (is_signed_v<Rhs>)
  {
    test_div_overflow<Result>(Lhs{0}, Rhs{-1}, false);
  }
  //--------------------------------------------------------------------------------------------------------------------
  // min, max cases

  // 8. max / 1 -> max >= result_max?
  test_div_overflow<Result>(lhs_max, Rhs{1}, cuda::std::cmp_greater(lhs_max, result_max));

  // nvcc 12.0 doesn't support this case in a constexpr context
#if _CCCL_CUDA_COMPILER(NVCC, !=, 12, 0)
  // 9. min / 1 -> min < result_min?
  test_div_overflow<Result>(lhs_min, Rhs{1}, cuda::std::cmp_less(lhs_min, result_min));

  if constexpr (is_signed_v<Lhs> && is_signed_v<Rhs>)
  {
    // 10. min / -1
    // when the result type is larger than the common type, min / -1 produces a valid result
    bool overflow = cuda::std::cmp_greater(neg_lhs_min, result_max);
    test_div_overflow<Result>(lhs_min, Rhs{-1}, overflow, !overflow, static_cast<Result>(neg_lhs_min));

    // 11. min / -2
    bool overflow2 = cuda::std::cmp_greater(neg_lhs_min / 2, result_max);
    test_div_overflow<Result>(lhs_min, Rhs{-2}, overflow2, !overflow2, static_cast<Result>(neg_lhs_min / 2));
  }
#endif // _CCCL_CUDA_COMPILER(NVCC, !=, 12, 0)
}

template <typename Lhs, typename Rhs>
__host__ __device__ constexpr void test_type()
{
  test_type<Lhs, Rhs, cuda::std::common_type_t<Lhs, Rhs>>();
  test_type<Lhs, Rhs, unsigned>();
  test_type<Lhs, Rhs, int>();
  // instantiation of all tests is very expensive. clang hits "constexpr evaluation hit maximum step limit"
#if _CCCL_COMPILER(GCC)
  test_type<Lhs, Rhs, long long>();
  test_type<Lhs, Rhs, unsigned long long>();
  test_type<Lhs, Rhs, signed char>();
  test_type<Lhs, Rhs, unsigned char>();
#endif
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
#endif
}

__host__ __device__ constexpr bool test()
{
  using cuda::std::is_same_v;
  static_assert(noexcept(cuda::div_overflow(int{}, int{})));
  static_assert(noexcept(cuda::div_overflow<unsigned>(cuda::std::declval<unsigned&>(), int{}, int{})));
  static_assert(is_same_v<decltype(cuda::div_overflow<unsigned>(cuda::std::declval<unsigned&>(), int{}, int{})), bool>);

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
#endif
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
