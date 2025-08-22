//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// For each `charT`, for each cv-unqualified arithmetic type `ArithmeticT`
// other than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>
//
// This file tests with `ArithmeticT = signed integer`, for each valid `charT`.
// Where `signed integer` is one of:
// - signed char
// - short
// - int
// - long
// - long long
// - __int128_t

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class CharT, class T>
__host__ __device__ void test_signed_int_formatter(
  cuda::std::basic_string_view<CharT> fmt,
  T value,
  cuda::std::size_t offset,
  cuda::std::basic_string_view<CharT> expected)
{
  using Container     = cuda::std::inplace_vector<CharT, 100>;
  using OutIt         = cuda::std::__back_insert_iterator<Container>;
  using ParseContext  = cuda::std::basic_format_parse_context<CharT>;
  using FormatContext = cuda::std::basic_format_context<OutIt, CharT>;

  Container container{};

  auto store   = cuda::std::make_format_args<FormatContext>(value);
  auto args    = cuda::std::basic_format_args{store};
  auto context = cuda::std::__fmt_make_format_context(OutIt{container}, args);

  cuda::std::formatter<T, CharT> formatter{};
  static_assert(cuda::std::semiregular<decltype(formatter)>);

  ParseContext parse_ctx{fmt};
  auto it = formatter.parse(parse_ctx);
  static_assert(cuda::std::is_same_v<decltype(it), typename cuda::std::basic_string_view<CharT>::const_iterator>);

  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(cuda::std::to_address(it) == cuda::std::to_address(fmt.end()) - offset);

  formatter.format(value, context);
  assert((cuda::std::basic_string_view{container.data(), container.size()} == expected));
}

template <class CharT, class T, class ValueT>
__host__ __device__ void test_termination_condition(
  cuda::std::basic_string_view<CharT> fmt, ValueT value, cuda::std::basic_string_view<CharT> expected)
{
  // Skip the test if the value is out of range for the type.
  if (!cuda::std::in_range<T>(value))
  {
    return;
  }

  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  assert(fmt.back() == TEST_CHARLIT(CharT, '}'));

  test_signed_int_formatter(fmt, static_cast<T>(value), 1, expected);
  fmt.remove_suffix(1);
  test_signed_int_formatter(fmt, static_cast<T>(value), 0, expected);
}

template <class CharT, class T>
__host__ __device__ void test_type()
{
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), -128, TEST_STRLIT(CharT, "-128"));
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), 0, TEST_STRLIT(CharT, "0"));
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), 127, TEST_STRLIT(CharT, "127"));
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), -32768, TEST_STRLIT(CharT, "-32768"));
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), 32767, TEST_STRLIT(CharT, "32767"));
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), -2147483647, TEST_STRLIT(CharT, "-2147483647"));
  test_termination_condition<CharT, T>(TEST_STRLIT(CharT, "}"), 2147483647, TEST_STRLIT(CharT, "2147483647"));
  test_termination_condition<CharT, T>(
    TEST_STRLIT(CharT, "}"),
    cuda::std::numeric_limits<cuda::std::int64_t>::min(),
    TEST_STRLIT(CharT, "-9223372036854775808"));
  test_termination_condition<CharT, T>(
    TEST_STRLIT(CharT, "}"),
    cuda::std::numeric_limits<cuda::std::int64_t>::max(),
    TEST_STRLIT(CharT, "9223372036854775807"));
#if _CCCL_HAS_INT128()
  test_termination_condition<CharT, T>(
    TEST_STRLIT(CharT, "}"),
    cuda::std::numeric_limits<__int128_t>::min(),
    TEST_STRLIT(CharT, "-170141183460469231731687303715884105728"));
  test_termination_condition<CharT, T>(
    TEST_STRLIT(CharT, "}"),
    cuda::std::numeric_limits<__int128_t>::max(),
    TEST_STRLIT(CharT, "170141183460469231731687303715884105727"));
#endif // _CCCL_HAS_INT128()
}

template <class CharT>
__host__ __device__ void test_type()
{
  test_type<CharT, signed char>();
  test_type<CharT, signed short>();
  test_type<CharT, signed int>();
  test_type<CharT, signed long>();
  test_type<CharT, signed long long>();
#if _CCCL_HAS_INT128()
  test_type<CharT, __int128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ bool test()
{
  test_type<char>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
