//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// The Standard does not specify a constructor
// basic_format_context(Out out, basic_format_args<basic_format_context> args);

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "literal.h"

template <class Expected>
struct Visitor
{
  template <class T>
  [[nodiscard]] __host__ __device__ bool operator()([[maybe_unused]] T v) const
  {
    if constexpr (cuda::std::is_same_v<T, Expected>)
    {
      return v == expected;
    }
    else
    {
      return false;
    }
  }
  Expected expected;
};

template <class Context, class T>
__host__ __device__ bool test_basic_format_arg(cuda::std::basic_format_arg<Context> arg, T expected)
{
  return cuda::std::visit_format_arg(Visitor<T>{expected}, arg);
}

template <class CharT>
__host__ __device__ void test_constructor()
{
  using Container = cuda::std::inplace_vector<CharT, 3>;
  using OutIt     = cuda::std::__back_insert_iterator<Container>;
  using Context   = cuda::std::basic_format_context<OutIt, CharT>;

  auto b = true;
  auto c = TEST_CHARLIT(CharT, 'a');
  auto n = 42;
  auto s = cuda::std::basic_string_view{TEST_STRLIT(CharT, "string")};

  Container container{};

  auto store   = cuda::std::make_format_args<Context>(b, c, n, s);
  auto args    = cuda::std::basic_format_args{store};
  auto context = cuda::std::__fmt_make_format_context(OutIt{container}, args);

  assert(args.__size() == 4);

  assert(test_basic_format_arg(context.arg(0), b));
  assert(test_basic_format_arg(context.arg(1), c));
  assert(test_basic_format_arg(context.arg(2), n));
  assert(test_basic_format_arg(context.arg(3), s));

  context.out() = CharT('a');
  assert(container.size() == 1);
  assert(container.front() == CharT('a'));

  // Format context cannot be constructed by user, is not default constructible, copyable, movable, assignable and
  // move-assignable
  static_assert(!cuda::std::is_default_constructible_v<Context>);
  static_assert(!cuda::std::is_copy_constructible_v<Context>);
  static_assert(!cuda::std::is_move_constructible_v<Context>);
  static_assert(!cuda::std::is_copy_assignable_v<Context>);
  static_assert(!cuda::std::is_move_assignable_v<Context>);
}

__host__ __device__ void test()
{
  test_constructor<char>();
#if _CCCL_HAS_WCHAR_T()
  test_constructor<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
