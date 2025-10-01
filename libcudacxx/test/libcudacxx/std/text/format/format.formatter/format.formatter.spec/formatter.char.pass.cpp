//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// C++23 the formatter is a debug-enabled specialization.
// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// The specializations
//   template<> struct formatter<char, char>;
//   template<> struct formatter<char, wchar_t>;
//   template<> struct formatter<wchar_t, wchar_t>;

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "literal.h"

template <class CharT, class ArgT>
__host__ __device__ void test_char_formatter(
  cuda::std::basic_string_view<CharT> fmt,
  ArgT value,
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

  cuda::std::formatter<ArgT, CharT> formatter{};
  static_assert(cuda::std::semiregular<decltype(formatter)>);

  ParseContext parse_ctx{fmt};
  auto it = formatter.parse(parse_ctx);
  static_assert(cuda::std::is_same_v<decltype(it), typename cuda::std::basic_string_view<CharT>::const_iterator>);

  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(cuda::std::to_address(it) == cuda::std::to_address(fmt.end()) - offset);

  formatter.format(value, context);
  assert((cuda::std::basic_string_view{container.data(), container.size()} == expected));
}

template <class CharT, class ArgT>
__host__ __device__ void test_termination_condition(
  cuda::std::basic_string_view<CharT> fmt, ArgT value, cuda::std::basic_string_view<CharT> expected)
{
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  assert(fmt.back() == TEST_CHARLIT(CharT, '}'));

  test_char_formatter(fmt, value, 1, expected);
  fmt.remove_suffix(1);
  test_char_formatter(fmt, value, 0, expected);
}

template <class CharT, class ArgT>
__host__ __device__ void test_type()
{
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_CHARLIT(ArgT, 'a'), TEST_STRLIT(CharT, "a"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_CHARLIT(ArgT, 'z'), TEST_STRLIT(CharT, "z"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_CHARLIT(ArgT, 'A'), TEST_STRLIT(CharT, "A"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_CHARLIT(ArgT, 'Z'), TEST_STRLIT(CharT, "Z"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_CHARLIT(ArgT, '0'), TEST_STRLIT(CharT, "0"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), TEST_CHARLIT(ArgT, '9'), TEST_STRLIT(CharT, "9"));
}

__host__ __device__ bool test()
{
  test_type<char, char>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t, char>();
  test_type<wchar_t, wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
