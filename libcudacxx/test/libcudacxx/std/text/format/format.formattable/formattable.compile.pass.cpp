//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class T, class charT>
// concept formattable = ...

#include <cuda/std/__format_>
#include <cuda/std/array>
#include <cuda/std/bitset>
#include <cuda/std/chrono>
#include <cuda/std/complex>
#include <cuda/std/concepts>
#include <cuda/std/memory>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "min_allocator.h"
#include "test_macros.h"

template <class T, class CharT>
TEST_FUNC void assert_is_not_formattable()
{
  static_assert(!cuda::std::formattable<T, CharT>);
  static_assert(!cuda::std::formattable<T&, CharT>);
  static_assert(!cuda::std::formattable<T&&, CharT>);
  static_assert(!cuda::std::formattable<const T, CharT>);
  static_assert(!cuda::std::formattable<const T&, CharT>);
  static_assert(!cuda::std::formattable<const T&&, CharT>);
}

template <class T, class CharT>
TEST_FUNC void assert_is_formattable()
{
  // Only formatters for CharT == char || CharT == wchar_t are enabled for the
  // standard formatters. When CharT is a different type the formatter should
  // be disabled.
  if constexpr (cuda::std::same_as<CharT, char>
#if _CCCL_HAS_WCHAR_T()
                || cuda::std::same_as<CharT, wchar_t>
#endif // _CCCL_HAS_WCHAR_T()
  )
  {
    static_assert(cuda::std::formattable<T, CharT>);
    static_assert(cuda::std::formattable<T&, CharT>);
    static_assert(cuda::std::formattable<T&&, CharT>);
    static_assert(cuda::std::formattable<const T, CharT>);
    static_assert(cuda::std::formattable<const T&, CharT>);
    static_assert(cuda::std::formattable<const T&&, CharT>);
  }
  else
  {
    assert_is_not_formattable<T, CharT>();
  }
}

// Tests for P0645 Text Formatting
template <class CharT>
TEST_FUNC void test_P0645()
{
#if _CCCL_HAS_WCHAR_T()
  // Tests the special formatter that converts a char to a wchar_t.
  assert_is_formattable<char, wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
  assert_is_formattable<CharT, CharT>();

  assert_is_formattable<CharT*, CharT>();
  assert_is_formattable<const CharT*, CharT>();
  assert_is_formattable<CharT[42], CharT>();
  if constexpr (!cuda::std::same_as<CharT, int>)
  { // string and string_view only work with proper character types
    assert_is_formattable<cuda::std::basic_string_view<CharT>, CharT>();
  }

  assert_is_formattable<bool, CharT>();

  assert_is_formattable<signed char, CharT>();
  assert_is_formattable<signed short, CharT>();
  assert_is_formattable<signed int, CharT>();
  assert_is_formattable<signed long, CharT>();
  assert_is_formattable<signed long long, CharT>();
#if _CCCL_HAS_INT128()
  assert_is_formattable<__int128_t, CharT>();
#endif // _CCCL_HAS_INT128()

  assert_is_formattable<unsigned char, CharT>();
  assert_is_formattable<unsigned short, CharT>();
  assert_is_formattable<unsigned int, CharT>();
  assert_is_formattable<unsigned long, CharT>();
  assert_is_formattable<unsigned long long, CharT>();
#if _CCCL_HAS_INT128()
  assert_is_formattable<__uint128_t, CharT>();
#endif // _CCCL_HAS_INT128()

  // floating-point types are tested in concept.formattable.float.compile.pass.cpp

  assert_is_formattable<cuda::std::nullptr_t, CharT>();
  assert_is_formattable<void*, CharT>();
  assert_is_formattable<const void*, CharT>();
}

// Tests for P1636 Formatters for library types
//
// The paper hasn't been voted in so currently all formatters are disabled.
// Note the paper has been abandoned, the types are kept since other papers may
// introduce these formatters.
template <class CharT>
TEST_FUNC void test_P1636()
{
  assert_is_not_formattable<cuda::std::complex<double>, CharT>();
  assert_is_not_formattable<cuda::std::unique_ptr<int>, CharT>();
}

// todo(dabayer): Enable once formatters for ranges are implemented.
// Tests for P2286 Formatting ranges
// template <class CharT>
// TEST_FUNC void test_P2286() {
//   assert_is_formattable<cuda::std::array<int, 42>, CharT>();
//   assert_is_formattable<cuda::std::vector<int>, CharT>();
//   assert_is_formattable<cuda::std::deque<int>, CharT>();
//   assert_is_formattable<cuda::std::forward_list<int>, CharT>();
//   assert_is_formattable<cuda::std::list<int>, CharT>();

//   assert_is_formattable<cuda::std::set<int>, CharT>();
//   assert_is_formattable<cuda::std::map<int, int>, CharT>();
//   assert_is_formattable<cuda::std::multiset<int>, CharT>();
//   assert_is_formattable<cuda::std::multimap<int, int>, CharT>();

//   assert_is_formattable<cuda::std::unordered_set<int>, CharT>();
//   assert_is_formattable<cuda::std::unordered_map<int, int>, CharT>();
//   assert_is_formattable<cuda::std::unordered_multiset<int>, CharT>();
//   assert_is_formattable<cuda::std::unordered_multimap<int, int>, CharT>();

//   assert_is_formattable<cuda::std::stack<int>, CharT>();
//   assert_is_formattable<cuda::std::queue<int>, CharT>();
//   assert_is_formattable<cuda::std::priority_queue<int>, CharT>();

//   assert_is_formattable<cuda::std::span<int>, CharT>();

//   assert_is_formattable<cuda::std::valarray<int>, CharT>();

//   assert_is_formattable<cuda::std::pair<int, int>, CharT>();
//   assert_is_formattable<cuda::std::tuple<int>, CharT>();

//   test_P2286_vector_bool<CharT, cuda::std::vector<bool>>();
//   test_P2286_vector_bool<CharT, cuda::std::vector<bool, cuda::std::allocator<bool>>>();
//   test_P2286_vector_bool<CharT, cuda::std::vector<bool, min_allocator<bool>>>();
// }

// Tests volatile qualified objects are no longer formattable.
template <class CharT>
TEST_FUNC void test_LWG3631()
{
  assert_is_not_formattable<volatile CharT, CharT>();

  assert_is_not_formattable<volatile bool, CharT>();

  assert_is_not_formattable<volatile signed int, CharT>();
  assert_is_not_formattable<volatile unsigned int, CharT>();

  assert_is_not_formattable<volatile cuda::std::chrono::microseconds, CharT>();
  assert_is_not_formattable<volatile cuda::std::chrono::sys_time<cuda::std::chrono::microseconds>, CharT>();
  assert_is_not_formattable<volatile cuda::std::chrono::day, CharT>();

  assert_is_not_formattable<cuda::std::array<volatile int, 42>, CharT>();

  assert_is_not_formattable<cuda::std::pair<volatile int, int>, CharT>();
  assert_is_not_formattable<cuda::std::pair<int, volatile int>, CharT>();
  assert_is_not_formattable<cuda::std::pair<volatile int, volatile int>, CharT>();
}

TEST_FUNC void test_LWG3944()
{
#if _CCCL_HAS_WCHAR_T()
  assert_is_not_formattable<char*, wchar_t>();
  assert_is_not_formattable<const char*, wchar_t>();
  assert_is_not_formattable<char[42], wchar_t>();
  assert_is_not_formattable<cuda::std::string, wchar_t>();
  assert_is_not_formattable<cuda::std::string_view, wchar_t>();

  assert_is_formattable<cuda::std::vector<char>, wchar_t>();
  assert_is_formattable<cuda::std::set<char>, wchar_t>();
  assert_is_formattable<cuda::std::map<char, char>, wchar_t>();
  assert_is_formattable<cuda::std::tuple<char>, wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

class c
{
  TEST_FUNC void f();
  TEST_FUNC void fc() const;
  TEST_FUNC static void sf();
};
enum e
{
  a
};
enum class ec
{
  a
};
template <class CharT>
TEST_FUNC void test_disabled()
{
#if _CCCL_HAS_WCHAR_T()
  assert_is_not_formattable<const char*, wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
#if _CCCL_HAS_CHAR8_T()
  assert_is_not_formattable<const char*, char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  assert_is_not_formattable<const char*, char16_t>();
  assert_is_not_formattable<const char*, char32_t>();

  assert_is_not_formattable<c, CharT>();
  assert_is_not_formattable<const c, CharT>();
  assert_is_not_formattable<volatile c, CharT>();
  assert_is_not_formattable<const volatile c, CharT>();

  assert_is_not_formattable<e, CharT>();
  assert_is_not_formattable<const e, CharT>();
  assert_is_not_formattable<volatile e, CharT>();
  assert_is_not_formattable<const volatile e, CharT>();

  assert_is_not_formattable<ec, CharT>();
  assert_is_not_formattable<const ec, CharT>();
  assert_is_not_formattable<volatile ec, CharT>();
  assert_is_not_formattable<const volatile ec, CharT>();

  assert_is_not_formattable<int*, CharT>();
  assert_is_not_formattable<const int*, CharT>();
  assert_is_not_formattable<volatile int*, CharT>();
  assert_is_not_formattable<const volatile int*, CharT>();

  assert_is_not_formattable<c*, CharT>();
  assert_is_not_formattable<const c*, CharT>();
  assert_is_not_formattable<volatile c*, CharT>();
  assert_is_not_formattable<const volatile c*, CharT>();

  assert_is_not_formattable<e*, CharT>();
  assert_is_not_formattable<const e*, CharT>();
  assert_is_not_formattable<volatile e*, CharT>();
  assert_is_not_formattable<const volatile e*, CharT>();

  assert_is_not_formattable<ec*, CharT>();
  assert_is_not_formattable<const ec*, CharT>();
  assert_is_not_formattable<volatile ec*, CharT>();
  assert_is_not_formattable<const volatile ec*, CharT>();

  assert_is_not_formattable<void (*)(), CharT>();
  assert_is_not_formattable<void (c::*)(), CharT>();
  assert_is_not_formattable<void (c::*)() const, CharT>();

  assert_is_not_formattable<cuda::std::optional<int>, CharT>();
  assert_is_not_formattable<cuda::std::variant<int>, CharT>();

  assert_is_not_formattable<cuda::std::unique_ptr<c>, CharT>();

  assert_is_not_formattable<cuda::std::array<c, 42>, CharT>();

  assert_is_not_formattable<cuda::std::span<c>, CharT>();

  assert_is_not_formattable<cuda::std::pair<c, int>, CharT>();
  assert_is_not_formattable<cuda::std::tuple<c>, CharT>();

  assert_is_not_formattable<cuda::std::optional<c>, CharT>();
  assert_is_not_formattable<cuda::std::variant<c>, CharT>();
}

struct abstract
{
  TEST_FUNC virtual ~abstract() = 0;
};

template <>
struct cuda::std::formatter<abstract, char>
{
  template <class ParseContext>
  TEST_FUNC constexpr typename ParseContext::iterator parse(ParseContext& parse_ctx)
  {
    return parse_ctx.begin();
  }

  template <class FormatContext>
  TEST_FUNC typename FormatContext::iterator format(const abstract&, FormatContext& ctx) const
  {
    return ctx.out();
  }
};

#if _CCCL_HAS_WCHAR_T()
template <>
struct cuda::std::formatter<abstract, wchar_t>
{
  template <class ParseContext>
  TEST_FUNC constexpr typename ParseContext::iterator parse(ParseContext& parse_ctx)
  {
    return parse_ctx.begin();
  }

  template <class FormatContext>
  TEST_FUNC typename FormatContext::iterator format(const abstract&, FormatContext& ctx) const
  {
    return ctx.out();
  }
};
#endif // _CCCL_HAS_WCHAR_T()

template <class CharT>
TEST_FUNC void test_abstract_class()
{
  assert_is_formattable<abstract, CharT>();
}

enum class TypeWithNonSemiregularFormatter : int
{
};

template <>
struct cuda::std::formatter<TypeWithNonSemiregularFormatter, char>
{
  formatter(const formatter&) = delete;

  template <class ParseContext>
  TEST_FUNC constexpr typename ParseContext::iterator parse(ParseContext& parse_ctx)
  {
    return parse_ctx.begin();
  }

  template <class FormatContext>
  TEST_FUNC typename FormatContext::iterator format(const TypeWithNonSemiregularFormatter&, FormatContext& ctx) const
  {
    return ctx.out();
  }
};

TEST_FUNC void test_non_semiregular()
{
  assert_is_not_formattable<TypeWithNonSemiregularFormatter, char>();
}

template <class CharT>
TEST_FUNC void test()
{
  test_P0645<CharT>();
  test_P1636<CharT>();
  test_LWG3631<CharT>();
  test_LWG3944();
  test_abstract_class<CharT>();
  test_disabled<CharT>();
  test_non_semiregular();
}

TEST_FUNC void test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();
}

int main(int, char**)
{
  return 0;
}
