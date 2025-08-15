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
// ...
// For each charT, the pointer type specializations
// - template<> struct formatter<nullptr_t, charT>;
// - template<> struct formatter<void*, charT>;
// - template<> struct formatter<const void*, charT>;

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "literal.h"

template <class CharT, class PointerT>
__host__ __device__ void test_ptr_formatter(
  cuda::std::basic_string_view<CharT> fmt,
  PointerT arg,
  cuda::std::size_t offset,
  cuda::std::basic_string_view<CharT> expected)
{
  using Container     = cuda::std::inplace_vector<CharT, 100>;
  using OutIt         = cuda::std::__back_insert_iterator<Container>;
  using ParseContext  = cuda::std::basic_format_parse_context<CharT>;
  using FormatContext = cuda::std::basic_format_context<OutIt, CharT>;

  Container container{};

  auto store   = cuda::std::make_format_args<FormatContext>(arg);
  auto args    = cuda::std::basic_format_args{store};
  auto context = cuda::std::__fmt_make_format_context(OutIt{container}, args);

  cuda::std::formatter<PointerT, CharT> formatter;
  static_assert(cuda::std::semiregular<decltype(formatter)>);

  ParseContext parse_ctx{fmt};
  auto it = formatter.parse(parse_ctx);
  static_assert(cuda::std::is_same_v<decltype(it), typename cuda::std::basic_string_view<CharT>::const_iterator>);

  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(cuda::std::to_address(it) == cuda::std::to_address(fmt.end()) - offset);

  formatter.format(arg, context);
  assert((cuda::std::basic_string_view{container.data(), container.size()} == expected));
}

template <class CharT, class PointerT>
__host__ __device__ void test_termination_condition(
  cuda::std::basic_string_view<CharT> fmt, PointerT arg, cuda::std::basic_string_view<CharT> expected)
{
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  assert(fmt.back() == TEST_CHARLIT(CharT, '}'));

  test_ptr_formatter(fmt, arg, 1, expected);
  fmt.remove_suffix(1);
  test_ptr_formatter(fmt, arg, 0, expected);
}

template <class CharT>
__host__ __device__ void test_type()
{
  // 1. Test for nullptr_t
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), nullptr, TEST_STRLIT(CharT, "0x0"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "p}"), nullptr, TEST_STRLIT(CharT, "0x0"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "P}"), nullptr, TEST_STRLIT(CharT, "0X0"));

  // 2. Test for void*
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), (void*) (0), TEST_STRLIT(CharT, "0x0"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "p}"), (void*) (0x42), TEST_STRLIT(CharT, "0x42"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "P}"), (void*) (0xffff), TEST_STRLIT(CharT, "0XFFFF"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), (void*) (-1), TEST_STRLIT(CharT, "0xffffffffffffffff"));

  // 3. Test for const void*
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), (const void*) (0), TEST_STRLIT(CharT, "0x0"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "p}"), (const void*) (0x42), TEST_STRLIT(CharT, "0x42"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "P}"), (const void*) (0xffff), TEST_STRLIT(CharT, "0XFFFF"));
  test_termination_condition<CharT>(
    TEST_STRLIT(CharT, "}"), (const void*) (-1), TEST_STRLIT(CharT, "0xffffffffffffffff"));
}

__host__ __device__ bool test()
{
  test_type<char>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#endif // _CCCL_HAS_WIDE_CHARACTERS

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
