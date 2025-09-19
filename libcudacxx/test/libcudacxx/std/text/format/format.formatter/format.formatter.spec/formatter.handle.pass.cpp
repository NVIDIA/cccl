//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// A user defined formatter using
// template<class Context>
// class basic_format_arg<Context>::handle

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "literal.h"

enum class color
{
  black,
  red,
  gold
};

template <class CharT>
struct cuda::std::formatter<color, CharT> : cuda::std::formatter<const char*, CharT>
{
  template <class Context>
  __host__ __device__ auto format(color c, Context& ctx) const
  {
    const CharT* color_names[]{TEST_STRLIT(CharT, "black"), TEST_STRLIT(CharT, "red"), TEST_STRLIT(CharT, "gold")};
    return cuda::std::formatter<const CharT*>::format(color_names[static_cast<int>(c)], ctx);
  }
};

template <class CharT>
__host__ __device__ void test_handle(
  cuda::std::basic_string_view<CharT> fmt,
  color value,
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

  cuda::std::formatter<color, CharT> formatter{};
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
  cuda::std::basic_string_view<CharT> fmt, color value, cuda::std::basic_string_view<CharT> expected)
{
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  assert(fmt.back() == TEST_CHARLIT(CharT, '}'));

  test_handle(fmt, value, 1, expected);
  fmt.remove_suffix(1);
  test_handle(fmt, value, 0, expected);
}

template <class CharT>
__host__ __device__ void test_type()
{
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), color::black, TEST_STRLIT(CharT, "black"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), color::red, TEST_STRLIT(CharT, "red"));
  test_termination_condition<CharT>(TEST_STRLIT(CharT, "}"), color::gold, TEST_STRLIT(CharT, "gold"));
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
