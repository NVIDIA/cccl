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
// This file tests with `ArithmeticT = bool`, for each valid `charT`.

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "literal.h"

template <class CharT>
__host__ __device__ void test_bool_formatter(
  cuda::std::basic_string_view<CharT> fmt,
  bool value,
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

  cuda::std::formatter<bool, CharT> formatter{};
  static_assert(cuda::std::semiregular<decltype(formatter)>);

  ParseContext parse_ctx{fmt};
  auto it = formatter.parse(parse_ctx);
  static_assert(cuda::std::is_same_v<decltype(it), typename cuda::std::basic_string_view<CharT>::const_iterator>);

  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(cuda::std::to_address(it) == cuda::std::to_address(fmt.end()) - offset);

  formatter.format(value, context);
  assert((cuda::std::basic_string_view{container.data(), container.size()} == expected));
}

template <class CharT>
__host__ __device__ void test_termination_condition(
  cuda::std::basic_string_view<CharT> fmt, bool value, cuda::std::basic_string_view<CharT> expected)
{
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  assert(fmt.back() == TEST_CHARLIT(CharT, '}'));

  test_bool_formatter(fmt, value, 1, expected);
  fmt.remove_suffix(1);
  test_bool_formatter(fmt, value, 0, expected);
}

template <class CharT>
__host__ __device__ void test_type()
{
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), true, TEST_STRLIT(CharT, "true"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, ">6}"), true, TEST_STRLIT(CharT, "  true"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "*>5}"), true, TEST_STRLIT(CharT, "*true"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_^7}"), true, TEST_STRLIT(CharT, "_true__"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_<7}"), true, TEST_STRLIT(CharT, "true___"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "b}"), true, TEST_STRLIT(CharT, "1"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_^3o}"), true, TEST_STRLIT(CharT, "_1_"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_<5d}"), true, TEST_STRLIT(CharT, "1____"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, ">7X}"), true, TEST_STRLIT(CharT, "      1"));

  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), false, TEST_STRLIT(CharT, "false"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, ">7}"), false, TEST_STRLIT(CharT, "  false"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "*>8}"), false, TEST_STRLIT(CharT, "***false"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_^7}"), false, TEST_STRLIT(CharT, "_false_"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_<7}"), false, TEST_STRLIT(CharT, "false__"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "b}"), false, TEST_STRLIT(CharT, "0"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_^3o}"), false, TEST_STRLIT(CharT, "_0_"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_<5d}"), false, TEST_STRLIT(CharT, "0____"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, ">7X}"), false, TEST_STRLIT(CharT, "      0"));
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
