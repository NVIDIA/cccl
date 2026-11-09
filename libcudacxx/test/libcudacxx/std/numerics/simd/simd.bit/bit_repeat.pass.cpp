//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// [simd.bit], bit_repeat

#include <cuda/std/__simd_>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T>
struct repeat_counts
{
  template <typename I_vec>
  TEST_HOST_DEVICE_FUNC constexpr T operator()(I_vec) const
  {
    return static_cast<T>(1 + I_vec::value % (cuda::std::numeric_limits<T>::digits + 1));
  }
};

template <typename T, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_bit_repeat_scalar()
{
  using U_vec            = simd::vec<T, N>;
  constexpr int digits   = cuda::std::numeric_limits<T>::digits;
  constexpr int counts[] = {1, 2, 3, digits - 1, digits, digits + 1, cuda::std::numeric_limits<int>::max()};
  constexpr T patterns[] = {T{}, T{1}, T{5}, static_cast<T>(T{1} << (digits - 1)), cuda::std::numeric_limits<T>::max()};

  for (const T pattern : patterns)
  {
    const U_vec input(pattern);
    for (const int count : counts)
    {
      const U_vec result = simd::bit_repeat(input, count);
      for (int i = 0; i < N; ++i)
      {
        assert(result[i] == cuda::std::bit_repeat(pattern, count));
      }
    }
  }
}

template <typename T, int N>
struct test_bit_repeat
{
  TEST_HOST_DEVICE_FUNC constexpr void operator()() const
  {
    using U_vec = simd::vec<T, N>;
    using I_vec = simd::rebind_t<cuda::std::make_signed_t<T>, U_vec>;
    using I     = I_vec::value_type;

    const I_vec vec_counts(repeat_counts<I>{});
    const U_vec vec_bits(bit_values<T>{});
    const U_vec result = simd::bit_repeat(vec_bits, vec_counts);

    static_assert(cuda::std::is_same_v<decltype(simd::bit_repeat(vec_bits, vec_counts)), U_vec>);
    static_assert(cuda::std::is_same_v<decltype(simd::bit_repeat(vec_bits, 1)), U_vec>);
    // Older nvcc, GCC and MSVC misreport potentially throwing constexpr calls.
#if !TEST_CUDA_COMPILER(NVCC, ==, 12, 0) && !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC, <, 19, 30)
    static_assert(!noexcept(simd::bit_repeat(vec_bits, vec_counts)));
    static_assert(!noexcept(simd::bit_repeat(vec_bits, 1)));
#endif

    for (int i = 0; i < N; ++i)
    {
      assert(result[i] == cuda::std::bit_repeat(vec_bits[i], static_cast<int>(vec_counts[i])));
    }

    test_bit_repeat_scalar<T, N>();
  }
};

struct bit_repeat_function
{
  template <typename U_vec, typename I_vec>
  TEST_HOST_DEVICE_FUNC auto operator()(const U_vec& value, const I_vec& count) const
    -> decltype(simd::bit_repeat(value, count));
};

TEST_HOST_DEVICE_FUNC constexpr void test_constraints()
{
  using U_vec = simd::vec<unsigned, 4>;
  using I_vec = simd::vec<int, 4>;
  using F_vec = simd::vec<float, 4>;

  static_assert(cuda::std::is_invocable_v<bit_repeat_function, U_vec, int>);
  static_assert(cuda::std::is_invocable_v<bit_repeat_function, U_vec, I_vec>);
  static_assert(cuda::std::is_invocable_v<bit_repeat_function, U_vec, U_vec>);

  static_assert(!cuda::std::is_invocable_v<bit_repeat_function, I_vec, int>);
  static_assert(!cuda::std::is_invocable_v<bit_repeat_function, I_vec, I_vec>);
  static_assert(!cuda::std::is_invocable_v<bit_repeat_function, F_vec, int>);
  static_assert(!cuda::std::is_invocable_v<bit_repeat_function, U_vec, F_vec>);
  static_assert(!cuda::std::is_invocable_v<bit_repeat_function, U_vec, simd::vec<int, 3>>);
  static_assert(!cuda::std::is_invocable_v<bit_repeat_function, simd::vec<unsigned, 0>, int>);
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  _SIMD_BIT_TEST_UNSIGNED_TYPES(test_bit_repeat)
#if _CCCL_HAS_INT128()
  test_bit_repeat<__uint128_t, 1>{}();
  test_bit_repeat<__uint128_t, 4>{}();
#endif
  test_constraints();
  return true;
}

int main(int, char**)
{
  test(); // NOLINT(misc-static-assert)
  static_assert(test());
  return 0;
}
