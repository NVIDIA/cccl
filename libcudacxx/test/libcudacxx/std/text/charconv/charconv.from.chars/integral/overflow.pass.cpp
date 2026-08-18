//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function from a __host__ __device__ __tile__ function is not allowed

#include <cuda/std/cassert>
#include <cuda/std/charconv>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_HOST_DEVICE_FUNC constexpr char digit_to_char(unsigned int digit)
{
  return static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr cuda::std::size_t write_digits(char* output, T value, int base)
{
  char reversed[cuda::std::numeric_limits<T>::digits]{};
  cuda::std::size_t size = 0;

  do
  {
    reversed[size++] = digit_to_char(static_cast<unsigned int>(value % static_cast<T>(base)));
    value /= static_cast<T>(base);
  } while (value != 0);

  for (cuda::std::size_t i = 0; i != size; ++i)
  {
    output[i] = reversed[size - i - 1];
  }
  return size;
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr cuda::std::size_t make_upward_wrap_input(char* output, int base, bool negative)
{
  using U = cuda::std::make_unsigned_t<T>;

  U max_value = (cuda::std::numeric_limits<U>::max)();
  U prefix    = base == 2 ? max_value : static_cast<U>(max_value / static_cast<U>(base - 1) + 1);

  cuda::std::size_t size = 0;
  if (negative)
  {
    output[size++] = '-';
  }
  size += write_digits(output + size, prefix, base);
  output[size++] = digit_to_char(base == 2 ? 1 : 0);
  return size;
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_upward_wrap(int base, bool negative = false)
{
  using U = cuda::std::make_unsigned_t<T>;
  char input[cuda::std::numeric_limits<U>::digits + 2]{};
  cuda::std::size_t size = make_upward_wrap_input<T>(input, base, negative);

  T value     = static_cast<T>(23);
  auto result = cuda::std::from_chars(input, input + size, value, base);
  assert(result.ptr == input + size);
  assert(result.ec == cuda::std::errc::result_out_of_range);
  assert(value == static_cast<T>(23));
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_unsigned_boundary(int base)
{
  char input[cuda::std::numeric_limits<T>::digits]{};
  T max_value            = (cuda::std::numeric_limits<T>::max)();
  cuda::std::size_t size = write_digits(input, max_value, base);
  T value                = 0;
  auto result            = cuda::std::from_chars(input, input + size, value, base);
  assert(result.ptr == input + size);
  assert(result.ec == cuda::std::errc{});
  assert(value == max_value);
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_signed_boundaries(int base)
{
  using U = cuda::std::make_unsigned_t<T>;
  char input[cuda::std::numeric_limits<U>::digits + 1]{};

  {
    T max_value            = (cuda::std::numeric_limits<T>::max)();
    cuda::std::size_t size = write_digits(input, static_cast<U>(max_value), base);
    T value                = 0;
    auto result            = cuda::std::from_chars(input, input + size, value, base);
    assert(result.ptr == input + size);
    assert(result.ec == cuda::std::errc{});
    assert(value == max_value);
  }

  U min_magnitude = static_cast<U>(static_cast<U>((cuda::std::numeric_limits<T>::max)()) + U{1});
  input[0]        = '-';

  {
    cuda::std::size_t size = 1 + write_digits(input + 1, min_magnitude, base);
    T value                = 0;
    auto result            = cuda::std::from_chars(input, input + size, value, base);
    assert(result.ptr == input + size);
    assert(result.ec == cuda::std::errc{});
    assert(value == (cuda::std::numeric_limits<T>::min)());
  }

  {
    cuda::std::size_t size = write_digits(input, min_magnitude, base);
    T value                = 23;
    auto result            = cuda::std::from_chars(input, input + size, value, base);
    assert(result.ptr == input + size);
    assert(result.ec == cuda::std::errc::result_out_of_range);
    assert(value == 23);
  }

  {
    input[0]               = '-';
    cuda::std::size_t size = 1 + write_digits(input + 1, static_cast<U>(min_magnitude + 1), base);
    T value                = 23;
    auto result            = cuda::std::from_chars(input, input + size, value, base);
    assert(result.ptr == input + size);
    assert(result.ec == cuda::std::errc::result_out_of_range);
    assert(value == 23);
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_unsigned_width()
{
  for (int base = 2; base <= 36; ++base)
  {
    test_unsigned_boundary<T>(base);
    test_upward_wrap<T>(base);
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_signed_width()
{
  for (int base = 2; base <= 36; ++base)
  {
    test_signed_boundaries<T>(base);
  }

  // For bases 2 and 3, an upward wrap cannot land within the signed range, so
  // the signed range check detects it independently of the generic accumulator.
  for (int base = 4; base <= 36; ++base)
  {
    test_upward_wrap<T>(base);
    test_upward_wrap<T>(base, true);
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_width()
{
  if constexpr (cuda::std::is_signed_v<T>)
  {
    test_signed_width<T>();
  }
  else
  {
    test_unsigned_width<T>();
  }
}

TEST_HOST_DEVICE_FUNC void test_runtime()
{
  {
    constexpr char input[]   = "kz"; // 755 does not fit in uint8_t, but wraps upward to 243.
    cuda::std::uint8_t value = 17;
    auto result              = cuda::std::from_chars(input, input + 2, value, 36);
    assert(result.ptr == input + 2);
    assert(result.ec == cuda::std::errc::result_out_of_range);
    assert(value == 17);
  }

  test_width<char>();
  test_signed_width<cuda::std::int8_t>();
  test_unsigned_width<cuda::std::uint8_t>();
  test_signed_width<cuda::std::int16_t>();
  test_unsigned_width<cuda::std::uint16_t>();
  test_signed_width<cuda::std::int32_t>();
  test_unsigned_width<cuda::std::uint32_t>();
  test_signed_width<cuda::std::int64_t>();
  test_unsigned_width<cuda::std::uint64_t>();
#if _CCCL_HAS_INT128()
  test_signed_width<__int128_t>();
  test_unsigned_width<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

template <class T, cuda::std::size_t Size>
TEST_HOST_DEVICE_FUNC constexpr void test_constexpr_success(const char (&input)[Size], int base, T expected)
{
  T value     = static_cast<T>(23);
  auto result = cuda::std::from_chars(input, input + Size - 1, value, base);
  assert(result.ptr == input + Size - 1);
  assert(result.ec == cuda::std::errc{});
  assert(value == expected);
}

template <class T, cuda::std::size_t Size>
TEST_HOST_DEVICE_FUNC constexpr void test_constexpr_out_of_range(const char (&input)[Size], int base)
{
  T value     = static_cast<T>(23);
  auto result = cuda::std::from_chars(input, input + Size - 1, value, base);
  assert(result.ptr == input + Size - 1);
  assert(result.ec == cuda::std::errc::result_out_of_range);
  assert(value == static_cast<T>(23));
}

TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_8_bit()
{
  using I = cuda::std::int8_t;
  using U = cuda::std::uint8_t;

  test_constexpr_success<U>("255", 10, (cuda::std::numeric_limits<U>::max)());

  // Multiplication overflow that wraps upward to a representable value.
  test_constexpr_out_of_range<U>("kz", 36);

  // The multiplication (25 * 10) fits, but adding the final digit does not.
  test_constexpr_out_of_range<U>("256", 10);

  test_constexpr_success<I>("127", 10, (cuda::std::numeric_limits<I>::max)());
  test_constexpr_success<I>("-128", 10, (cuda::std::numeric_limits<I>::min)());
  test_constexpr_out_of_range<I>("128", 10);
  test_constexpr_out_of_range<I>("-129", 10);

  if constexpr (cuda::std::is_signed_v<char>)
  {
    test_constexpr_success<char>("127", 10, (cuda::std::numeric_limits<char>::max)());
    test_constexpr_success<char>("-128", 10, (cuda::std::numeric_limits<char>::min)());
    test_constexpr_out_of_range<char>("128", 10);
    test_constexpr_out_of_range<char>("-129", 10);
  }
  else
  {
    test_constexpr_success<char>("255", 10, (cuda::std::numeric_limits<char>::max)());
    test_constexpr_out_of_range<char>("256", 10);
  }

  // Parsing must continue after overflow to locate the first invalid character.
  {
    constexpr char input[] = "kz0!";
    U value                = 23;
    auto result            = cuda::std::from_chars(input, input + 4, value, 36);
    assert(result.ptr == input + 3);
    assert(result.ec == cuda::std::errc::result_out_of_range);
    assert(value == 23);
  }
  return true;
}

TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_16_bit()
{
  using I = cuda::std::int16_t;
  using U = cuda::std::uint16_t;

  test_constexpr_success<U>("65535", 10, (cuda::std::numeric_limits<U>::max)());
  test_constexpr_out_of_range<U>("65536", 10);
  test_constexpr_out_of_range<U>("1g10", 36);

  test_constexpr_success<I>("32767", 10, (cuda::std::numeric_limits<I>::max)());
  test_constexpr_success<I>("-32768", 10, (cuda::std::numeric_limits<I>::min)());
  test_constexpr_out_of_range<I>("32768", 10);
  test_constexpr_out_of_range<I>("-32769", 10);
  return true;
}

TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_32_bit()
{
  using I = cuda::std::int32_t;
  using U = cuda::std::uint32_t;

  test_constexpr_success<U>("4294967295", 10, (cuda::std::numeric_limits<U>::max)());
  test_constexpr_out_of_range<U>("4294967296", 10);
  test_constexpr_out_of_range<U>("2126880", 36);

  test_constexpr_success<I>("2147483647", 10, (cuda::std::numeric_limits<I>::max)());
  test_constexpr_success<I>("-2147483648", 10, (cuda::std::numeric_limits<I>::min)());
  test_constexpr_out_of_range<I>("2147483648", 10);
  test_constexpr_out_of_range<I>("-2147483649", 10);
  return true;
}

TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_64_bit()
{
  using I = cuda::std::int64_t;
  using U = cuda::std::uint64_t;

  test_constexpr_success<U>("18446744073709551615", 10, (cuda::std::numeric_limits<U>::max)());
  test_constexpr_out_of_range<U>("18446744073709551616", 10);
  test_constexpr_out_of_range<U>("405jklntyr810", 36);

  test_constexpr_success<I>("9223372036854775807", 10, (cuda::std::numeric_limits<I>::max)());
  test_constexpr_success<I>("-9223372036854775808", 10, (cuda::std::numeric_limits<I>::min)());
  test_constexpr_out_of_range<I>("9223372036854775808", 10);
  test_constexpr_out_of_range<I>("-9223372036854775809", 10);
  return true;
}

#if _CCCL_HAS_INT128()
TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_128_bit()
{
  using I = __int128_t;
  using U = __uint128_t;

  test_constexpr_success<U>("340282366920938463463374607431768211455", 10, (cuda::std::numeric_limits<U>::max)());
  test_constexpr_out_of_range<U>("340282366920938463463374607431768211456", 10);
  test_constexpr_out_of_range<U>("fl7524438ymb31pfwhzzmf480", 36);

  test_constexpr_success<I>("170141183460469231731687303715884105727", 10, (cuda::std::numeric_limits<I>::max)());
  test_constexpr_success<I>("-170141183460469231731687303715884105728", 10, (cuda::std::numeric_limits<I>::min)());
  test_constexpr_out_of_range<I>("170141183460469231731687303715884105728", 10);
  test_constexpr_out_of_range<I>("-170141183460469231731687303715884105729", 10);
  return true;
}
#endif // _CCCL_HAS_INT128()

TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_syntax()
{
  {
    constexpr char input[]    = "7B!";
    cuda::std::uint32_t value = 23;
    auto result               = cuda::std::from_chars(input, input + 3, value, 16);
    assert(result.ptr == input + 2);
    assert(result.ec == cuda::std::errc{});
    assert(value == 123);
  }

  {
    constexpr char input[]   = "+1";
    cuda::std::int32_t value = 23;
    auto result              = cuda::std::from_chars(input, input + 2, value);
    assert(result.ptr == input);
    assert(result.ec == cuda::std::errc::invalid_argument);
    assert(value == 23);
  }

  {
    constexpr char input[]    = "-1";
    cuda::std::uint32_t value = 23;
    auto result               = cuda::std::from_chars(input, input + 2, value);
    assert(result.ptr == input);
    assert(result.ec == cuda::std::errc::invalid_argument);
    assert(value == 23);
  }
  return true;
}

int main(int, char**)
{
  test_runtime();
  static_assert(test_constexpr_8_bit());
  static_assert(test_constexpr_16_bit());
  static_assert(test_constexpr_32_bit());
  static_assert(test_constexpr_64_bit());
#if _CCCL_HAS_INT128()
  static_assert(test_constexpr_128_bit());
#endif // _CCCL_HAS_INT128()
  static_assert(test_constexpr_syntax());
  return 0;
}
