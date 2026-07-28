//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/string_view>

#include "format_functions_common.h"
#include "test_macros.h"

// Provided by the selected checker.
TEST_FUNC bool check(...);
TEST_FUNC bool check_exception(...);

// Using a const ref for world and universe so a string literal will be a character array.
// When passed as character array W and U have different types.
template <class CharT, class W, class U>
TEST_FUNC void test_string(const W& world, const U& universe)
{
  // *** Valid input tests ***
  // Unused argument is ignored. TODO FMT what does the Standard mandate?
  assert(check(SV("hello world"), SV("hello {}"), world, universe));
  assert(check(SV("hello world and universe"), SV("hello {} and {}"), world, universe));
  assert(check(SV("hello world"), SV("hello {0}"), world, universe));
  assert(check(SV("hello universe"), SV("hello {1}"), world, universe));
  assert(check(SV("hello universe and world"), SV("hello {1} and {0}"), world, universe));

  assert(check(SV("hello world"), SV("hello {:_>}"), world));
  assert(check(SV("hello world   "), SV("hello {:8}"), world));
  assert(check(SV("hello    world"), SV("hello {:>8}"), world));
  assert(check(SV("hello ___world"), SV("hello {:_>8}"), world));
  assert(check(SV("hello _world__"), SV("hello {:_^8}"), world));
  assert(check(SV("hello world___"), SV("hello {:_<8}"), world));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("hello :::world"), SV("hello {::>8}"), world));
  assert(check(SV("hello <<<world"), SV("hello {:<>8}"), world));
  assert(check(SV("hello ^^^world"), SV("hello {:^>8}"), world));

  assert(check(SV("hello $world"), SV("hello {:$>{}}"), world, 6));
  assert(check(SV("hello $world"), SV("hello {0:$>{1}}"), world, 6));
  assert(check(SV("hello $world"), SV("hello {1:$>{0}}"), 6, world));

  assert(check(SV("hello world"), SV("hello {:.5}"), world));
  assert(check(SV("hello unive"), SV("hello {:.5}"), universe));

  assert(check(SV("hello univer"), SV("hello {:.{}}"), universe, 6));
  assert(check(SV("hello univer"), SV("hello {0:.{1}}"), universe, 6));
  assert(check(SV("hello univer"), SV("hello {1:.{0}}"), 6, universe));

  assert(check(SV("hello %world%"), SV("hello {:%^7.7}"), world));
  assert(check(SV("hello univers"), SV("hello {:%^7.7}"), universe));
  assert(check(SV("hello %world%"), SV("hello {:%^{}.{}}"), world, 7, 7));
  assert(check(SV("hello %world%"), SV("hello {0:%^{1}.{2}}"), world, 7, 7));
  assert(check(SV("hello %world%"), SV("hello {0:%^{2}.{1}}"), world, 7, 7));
  assert(check(SV("hello %world%"), SV("hello {1:%^{0}.{2}}"), 7, world, 7));

  assert(check(SV("hello world"), SV("hello {:_>s}"), world));
  assert(check(SV("hello $world"), SV("hello {:$>{}s}"), world, 6));
  assert(check(SV("hello world"), SV("hello {:.5s}"), world));
  assert(check(SV("hello univer"), SV("hello {:.{}s}"), universe, 6));
  assert(check(SV("hello %world%"), SV("hello {:%^7.7s}"), world));

  assert(check(SV("hello #####uni"), SV("hello {:#>8.3s}"), universe));
  assert(check(SV("hello ##uni###"), SV("hello {:#^8.3s}"), universe));
  assert(check(SV("hello uni#####"), SV("hello {:#<8.3s}"), universe));

  // *** sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("hello {:-}"), world));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("hello {:#}"), world));

  // *** zero-padding ***
  assert(check_exception("The width option should not have a leading zero", SV("hello {:0}"), world));

  // *** width ***
  // Width 0 allowed, but not useful for string arguments.
  assert(check(SV("hello world"), SV("hello {:{}}"), world, 0));

  // This limit isn't specified in the Standard.
  static_assert(cuda::std::__fmt_number_max == 2'147'483'647, "Update the assert and the test.");
  assert(check_exception("The numeric value of the format specifier is too large", SV("{:2147483648}"), world));
  assert(check_exception("The numeric value of the format specifier is too large", SV("{:5000000000}"), world));
  assert(check_exception("The numeric value of the format specifier is too large", SV("{:10000000000}"), world));

  assert(check_exception("An argument index may not have a negative value", SV("hello {:{}}"), world, -1));
  assert(check_exception(
    "The value of the argument index exceeds its maximum value", SV("hello {:{}}"), world, unsigned(-1)));
  assert(check_exception(
    "The argument index value is too large for the number of arguments supplied", SV("hello {:{}}"), world));
  assert(check_exception(
    "Replacement argument isn't a standard signed or unsigned integer type", SV("hello {:{}}"), world, universe));
  assert(check_exception(
    "Using manual argument numbering in automatic argument numbering mode", SV("hello {:{0}}"), world, 1));
  assert(check_exception(
    "Using automatic argument numbering in manual argument numbering mode", SV("hello {0:{}}"), world, 1));
  assert(check_exception("The argument index is invalid", SV("hello {0:{01}}"), world, 1));

  // This limit isn't specified in the Standard.
  static_assert(cuda::std::__fmt_number_max == 2'147'483'647, "Update the assert and the test.");
  assert(check_exception("The numeric value of the format specifier is too large", SV("{:.2147483648}"), world));
  assert(check_exception("The numeric value of the format specifier is too large", SV("{:.5000000000}"), world));
  assert(check_exception("The numeric value of the format specifier is too large", SV("{:.10000000000}"), world));

  // Precision 0 allowed, but not useful for string arguments.
  assert(check(SV("hello "), SV("hello {:.{}}"), world, 0));
  // Precision may have leading zeros. Secondly tests the value is still base 10.
  assert(check(SV("hello 0123456789"), SV("hello {:.000010}"), SV("0123456789abcdef")));
  assert(check_exception("An argument index may not have a negative value", SV("hello {:.{}}"), world, -1));
  assert(check_exception("The value of the argument index exceeds its maximum value", SV("hello {:.{}}"), world, ~0u));
  assert(check_exception(
    "The argument index value is too large for the number of arguments supplied", SV("hello {:.{}}"), world));
  assert(check_exception(
    "Replacement argument isn't a standard signed or unsigned integer type", SV("hello {:.{}}"), world, universe));
  assert(check_exception(
    "Using manual argument numbering in automatic argument numbering mode", SV("hello {:.{0}}"), world, 1));
  assert(check_exception(
    "Using automatic argument numbering in manual argument numbering mode", SV("hello {0:.{}}"), world, 1));
  assert(check_exception("The argument index is invalid", SV("hello {0:.{01}}"), world, 1));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // #if TEST_STD_VER > 20
  //   const char* valid_types = "s?";
  // #else
  //   const char* valid_types = "s";
  // #endif
  // for (const auto& fmt : invalid_types<CharT>(valid_types))
  // {
  //   assert(check_exception("The type option contains an invalid value for a string formatting argument", fmt,
  //   world));
  // }
}

template <class CharT>
TEST_FUNC void test_string()
{
  const auto& world    = TEST_STRLIT(CharT, "world");
  const auto& universe = TEST_STRLIT(CharT, "universe");

  // Test a string literal in a way it won't decay to a pointer.
  if constexpr (cuda::std::same_as<CharT, char>)
  {
    test_string<CharT>("world", "universe");
  }
#if _CCCL_HAS_WCHAR_T()
  else
  {
    test_string<CharT>(L"world", L"universe");
  }
#endif // _CCCL_HAS_WCHAR_T()

  test_string<CharT>(world, universe);
  test_string<CharT>(const_cast<CharT*>(world), const_cast<CharT*>(universe));
  test_string<CharT>(cuda::std::basic_string_view<CharT>{world}, cuda::std::basic_string_view<CharT>{universe});

  // todo(dabayer): Host stdlib interop.
  // test_string<CharT>(std::basic_string<CharT>{world}, std::basic_string<CharT>{universe});
  // test_string<CharT>(
  //   std::basic_string_view<CharT>{world}, std::basic_string_view<CharT>{universe});
}

TEST_FUNC void test()
{
  test_string<char>();
#if _CCCL_HAS_WCHAR_T()
  test_string<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  // test();
  return 0;
}
