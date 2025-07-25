//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// C++23 the formatter is a debug-enabled specialization.
// [format.functions]/25
//   Preconditions: formatter<remove_cvref_t<Ti>, charT> meets the
//   BasicFormatter requirements ([formatter.requirements]) for each Ti in
//   Args.

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "literal.h"

template <class CharT, cuda::std::size_t N>
__host__ __device__ void test_array_formatter(
  cuda::std::basic_string_view<CharT> fmt,
  const CharT (&value)[N],
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

  cuda::std::formatter<CharT[N], CharT> formatter{};
  static_assert(cuda::std::semiregular<decltype(formatter)>);

  ParseContext parse_ctx{fmt};
  auto it = formatter.parse(parse_ctx);
  static_assert(cuda::std::is_same_v<decltype(it), typename cuda::std::basic_string_view<CharT>::const_iterator>);

  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(cuda::std::to_address(it) == cuda::std::to_address(fmt.end()) - offset);

  formatter.format(value, context);
  assert((cuda::std::basic_string_view{container.data(), container.size()} == expected));
}

template <class CharT, cuda::std::size_t N>
__host__ __device__ void test_termination_condition(
  cuda::std::basic_string_view<CharT> fmt, const CharT (&value)[N], cuda::std::basic_string_view<CharT> expected)
{
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  assert(fmt.back() == TEST_CHARLIT(CharT, '}'));

  test_array_formatter(fmt, value, 1, expected);
  fmt.remove_suffix(1);
  test_array_formatter(fmt, value, 0, expected);
}

template <class CharT>
__host__ __device__ void test_type()
{
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "}"), TEST_STRLIT(CharT, " azAZ09,./<>?"), TEST_STRLIT(CharT, " azAZ09,./<>?"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_STRLIT(CharT, "abc\0abc"), TEST_STRLIT(CharT, "abc"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "world"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "_>}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "world"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, ">8}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "   world"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "_>8}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "___world"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "_^8}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "_world__"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "_<8}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "world___"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, ".5}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "world"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, ".5}"), TEST_STRLIT(CharT, "universe"), TEST_STRLIT(CharT, "unive"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "%^7.7}"), TEST_STRLIT(CharT, "world"), TEST_STRLIT(CharT, "%world%"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "%^7.7}"), TEST_STRLIT(CharT, "universe"), TEST_STRLIT(CharT, "univers"));
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
