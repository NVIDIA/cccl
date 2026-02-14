//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg); // Deprecated in C++26

#include <cuda/std/__format_>
#include <cuda/std/__string_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "literal.h"

struct HandleTag
{};

template <class Context, class To>
struct Visitor
{
  template <class T>
  __host__ __device__ To operator()([[maybe_unused]] T v) const
  {
    constexpr auto fmt_arg =
      (cuda::std::is_same_v<To, HandleTag>)
        ? cuda::std::__fmt_arg_t::__handle
        : cuda::std::__fmt_determine_arg_t<Context, To>();

    if constexpr (fmt_arg != cuda::std::__fmt_arg_t::__handle)
    {
      if constexpr (cuda::std::is_same_v<T, To>)
      {
        return v;
      }
      else
      {
        assert(false);
        return To{};
      }
    }
    else
    {
      assert((cuda::std::is_same_v<T, typename cuda::std::basic_format_arg<Context>::handle>) );
      return HandleTag{};
    }
  }
};

template <class CharT, class To, class From>
__host__ __device__ void test_visit_format_arg(From value)
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  constexpr auto fmt_arg = cuda::std::__fmt_determine_arg_t<Context, From>();

  auto store = cuda::std::make_format_args<Context>(value);
  auto args  = cuda::std::basic_format_args<Context>{store};

  assert(args.__size() == 1);
  assert(args.get(0));

  [[maybe_unused]] auto result = cuda::std::visit_format_arg(Visitor<Context, To>{}, args.get(0));

  if constexpr (fmt_arg == cuda::std::__fmt_arg_t::__string_view)
  {
    assert(cuda::std::equal(value.begin(), value.end(), result.begin(), result.end()));
  }
  else if constexpr (fmt_arg != cuda::std::__fmt_arg_t::__handle)
  {
    using Common = cuda::std::common_type_t<From, To>;
    assert(static_cast<Common>(result) == static_cast<Common>(value));
  }
}

template <class CharT>
__host__ __device__ void test_boolean()
{
  test_visit_format_arg<CharT, bool>(true);
  test_visit_format_arg<CharT, bool>(false);
}

template <class CharT>
__host__ __device__ void test_char()
{
  test_visit_format_arg<CharT, CharT, CharT>('a');
  test_visit_format_arg<CharT, CharT, CharT>('z');
  test_visit_format_arg<CharT, CharT, CharT>('0');
  test_visit_format_arg<CharT, CharT, CharT>('9');

  if (cuda::std::is_same_v<CharT, char>)
  {
    // char to char -> char
    test_visit_format_arg<CharT, CharT, char>('a');
    test_visit_format_arg<CharT, CharT, char>('z');
    test_visit_format_arg<CharT, CharT, char>('0');
    test_visit_format_arg<CharT, CharT, char>('9');
  }
#if _CCCL_HAS_WCHAR_T()
  else if (cuda::std::is_same_v<CharT, wchar_t>)
  {
    // char to wchar_t -> wchar_t
    test_visit_format_arg<CharT, wchar_t, char>('a');
    test_visit_format_arg<CharT, wchar_t, char>('z');
    test_visit_format_arg<CharT, wchar_t, char>('0');
    test_visit_format_arg<CharT, wchar_t, char>('9');
  }
#endif // _CCCL_HAS_WCHAR_T()
}

template <class CharT>
__host__ __device__ void test_signed_integers()
{
  test_visit_format_arg<CharT, int, signed char>(cuda::std::numeric_limits<signed char>::min());
  test_visit_format_arg<CharT, int, signed char>(0);
  test_visit_format_arg<CharT, int, signed char>(cuda::std::numeric_limits<signed char>::max());

  test_visit_format_arg<CharT, int, short>(cuda::std::numeric_limits<short>::min());
  test_visit_format_arg<CharT, int, short>(cuda::std::numeric_limits<signed char>::min());
  test_visit_format_arg<CharT, int, short>(0);
  test_visit_format_arg<CharT, int, short>(cuda::std::numeric_limits<signed char>::max());
  test_visit_format_arg<CharT, int, short>(cuda::std::numeric_limits<short>::max());

  test_visit_format_arg<CharT, int, int>(cuda::std::numeric_limits<int>::min());
  test_visit_format_arg<CharT, int, int>(cuda::std::numeric_limits<short>::min());
  test_visit_format_arg<CharT, int, int>(cuda::std::numeric_limits<signed char>::min());
  test_visit_format_arg<CharT, int, int>(0);
  test_visit_format_arg<CharT, int, int>(cuda::std::numeric_limits<signed char>::max());
  test_visit_format_arg<CharT, int, int>(cuda::std::numeric_limits<short>::max());
  test_visit_format_arg<CharT, int, int>(cuda::std::numeric_limits<int>::max());

  using LongToType = cuda::std::conditional_t<sizeof(long) == sizeof(int), int, long long>;

  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<long>::min());
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<int>::min());
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<short>::min());
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<signed char>::min());
  test_visit_format_arg<CharT, LongToType, long>(0);
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<signed char>::max());
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<short>::max());
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<int>::max());
  test_visit_format_arg<CharT, LongToType, long>(cuda::std::numeric_limits<long>::max());

  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<long long>::min());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<long>::min());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<int>::min());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<short>::min());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<signed char>::min());
  test_visit_format_arg<CharT, long long, long long>(0);
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<signed char>::max());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<short>::max());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<int>::max());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<long>::max());
  test_visit_format_arg<CharT, long long, long long>(cuda::std::numeric_limits<long long>::max());

#if _CCCL_HAS_INT128()
  test_visit_format_arg<CharT, HandleTag>(__int128_t{});
#endif // _CCCL_HAS_INT128()
}

template <class CharT>
__host__ __device__ void test_unsigned_integers()
{
  test_visit_format_arg<CharT, unsigned, unsigned char>(0);
  test_visit_format_arg<CharT, unsigned, unsigned char>(cuda::std::numeric_limits<unsigned char>::max());

  test_visit_format_arg<CharT, unsigned, unsigned short>(0);
  test_visit_format_arg<CharT, unsigned, unsigned short>(cuda::std::numeric_limits<unsigned char>::max());
  test_visit_format_arg<CharT, unsigned, unsigned short>(cuda::std::numeric_limits<unsigned short>::max());

  test_visit_format_arg<CharT, unsigned, unsigned>(0);
  test_visit_format_arg<CharT, unsigned, unsigned>(cuda::std::numeric_limits<unsigned char>::max());
  test_visit_format_arg<CharT, unsigned, unsigned>(cuda::std::numeric_limits<unsigned short>::max());
  test_visit_format_arg<CharT, unsigned, unsigned>(cuda::std::numeric_limits<unsigned>::max());

  using UnsignedLongToType =
    cuda::std::conditional_t<sizeof(unsigned long) == sizeof(unsigned), unsigned, unsigned long long>;

  test_visit_format_arg<CharT, UnsignedLongToType, unsigned long>(0);
  test_visit_format_arg<CharT, UnsignedLongToType, unsigned long>(cuda::std::numeric_limits<unsigned char>::max());
  test_visit_format_arg<CharT, UnsignedLongToType, unsigned long>(cuda::std::numeric_limits<unsigned short>::max());
  test_visit_format_arg<CharT, UnsignedLongToType, unsigned long>(cuda::std::numeric_limits<unsigned>::max());
  test_visit_format_arg<CharT, UnsignedLongToType, unsigned long>(cuda::std::numeric_limits<unsigned long>::max());

  test_visit_format_arg<CharT, unsigned long long, unsigned long long>(0);
  test_visit_format_arg<CharT, unsigned long long, unsigned long long>(cuda::std::numeric_limits<unsigned char>::max());
  test_visit_format_arg<CharT, unsigned long long, unsigned long long>(
    cuda::std::numeric_limits<unsigned short>::max());
  test_visit_format_arg<CharT, unsigned long long, unsigned long long>(cuda::std::numeric_limits<unsigned>::max());
  test_visit_format_arg<CharT, unsigned long long, unsigned long long>(cuda::std::numeric_limits<unsigned long>::max());
  test_visit_format_arg<CharT, unsigned long long, unsigned long long>(
    cuda::std::numeric_limits<unsigned long long>::max());

#if _CCCL_HAS_INT128()
  test_visit_format_arg<CharT, HandleTag>(__uint128_t{});
#endif // _CCCL_HAS_INT128()
}

template <class CharT>
__host__ __device__ void test_floating_point_types()
{
  test_visit_format_arg<CharT, float>(-cuda::std::numeric_limits<float>::max());
  test_visit_format_arg<CharT, float>(-cuda::std::numeric_limits<float>::min());
  test_visit_format_arg<CharT, float>(-0.0f);
  test_visit_format_arg<CharT, float>(0.0f);
  test_visit_format_arg<CharT, float>(cuda::std::numeric_limits<float>::min());
  test_visit_format_arg<CharT, float>(cuda::std::numeric_limits<float>::max());

  test_visit_format_arg<CharT, double>(-cuda::std::numeric_limits<double>::max());
  test_visit_format_arg<CharT, double>(-cuda::std::numeric_limits<double>::min());
  test_visit_format_arg<CharT, double>(-0.0);
  test_visit_format_arg<CharT, double>(0.0);
  test_visit_format_arg<CharT, double>(cuda::std::numeric_limits<double>::min());
  test_visit_format_arg<CharT, double>(cuda::std::numeric_limits<double>::max());

#if _CCCL_HAS_LONG_DOUBLE()
  test_visit_format_arg<CharT, long double>(-cuda::std::numeric_limits<long double>::max());
  test_visit_format_arg<CharT, long double>(-cuda::std::numeric_limits<long double>::min());
  test_visit_format_arg<CharT, long double>(-0.0l);
  test_visit_format_arg<CharT, long double>(0.0l);
  test_visit_format_arg<CharT, long double>(cuda::std::numeric_limits<long double>::min());
  test_visit_format_arg<CharT, long double>(cuda::std::numeric_limits<long double>::max());
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <class CharT>
__host__ __device__ void test_const_char_pointers()
{
  test_visit_format_arg<CharT, const CharT*>(TEST_STRLIT(CharT, ""));
  test_visit_format_arg<CharT, const CharT*>(TEST_STRLIT(CharT, "abc"));
}

template <class CharT>
__host__ __device__ void test_string_view()
{
  constexpr auto empty = TEST_STRLIT(CharT, "");
  constexpr auto str   = TEST_STRLIT(CharT, "abc");

  {
    using SV = cuda::std::basic_string_view<CharT>;

    test_visit_format_arg<CharT, SV>(SV{});
    test_visit_format_arg<CharT, SV>(SV{empty});
    test_visit_format_arg<CharT, SV>(SV{str});
  }
  {
    struct test_char_traits : cuda::std::char_traits<CharT>
    {};
    using SV = cuda::std::basic_string_view<CharT, test_char_traits>;

    test_visit_format_arg<CharT, SV>(SV{});
    test_visit_format_arg<CharT, SV>(SV{empty});
    test_visit_format_arg<CharT, SV>(SV{str});
  }
}

template <class CharT>
__host__ __device__ void test_pointers()
{
  test_visit_format_arg<CharT, const void*>(nullptr);
  int i = 0;
  test_visit_format_arg<CharT, const void*>(static_cast<void*>(&i));
  const int ci = 0;
  test_visit_format_arg<CharT, const void*>(static_cast<const void*>(&ci));
}

template <class CharT>
__host__ __device__ void test()
{
  test_boolean<CharT>();
  test_char<CharT>();
  test_signed_integers<CharT>();
  test_unsigned_integers<CharT>();
  test_floating_point_types<CharT>();
  test_const_char_pointers<CharT>();
  test_string_view<CharT>();
  test_pointers<CharT>();
}

__host__ __device__ void test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  return 0;
}
