//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// basic_format_arg<Context> get(size_t i) const noexcept;

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "literal.h"

template <class Context>
__host__ __device__ bool
test_format_arg_eq(const cuda::std::basic_format_arg<Context>& lhs, const cuda::std::basic_format_arg<Context>& rhs)
{
  if (lhs.__type_ != rhs.__type_)
  {
    return false;
  }
  switch (lhs.__type_)
  {
    case cuda::std::__fmt_arg_t::__none:
      return true;
    case cuda::std::__fmt_arg_t::__boolean:
      return lhs.__value_.__boolean_ == rhs.__value_.__boolean_;
    case cuda::std::__fmt_arg_t::__char_type:
      return lhs.__value_.__char_type_ == rhs.__value_.__char_type_;
    case cuda::std::__fmt_arg_t::__int:
      return lhs.__value_.__int_ == rhs.__value_.__int_;
    case cuda::std::__fmt_arg_t::__long_long:
      return lhs.__value_.__long_long_ == rhs.__value_.__long_long_;
    case cuda::std::__fmt_arg_t::__unsigned:
      return lhs.__value_.__unsigned_ == rhs.__value_.__unsigned_;
    case cuda::std::__fmt_arg_t::__unsigned_long_long:
      return lhs.__value_.__unsigned_long_long_ == rhs.__value_.__unsigned_long_long_;
    case cuda::std::__fmt_arg_t::__float:
      return lhs.__value_.__float_ == rhs.__value_.__float_;
    case cuda::std::__fmt_arg_t::__double:
      return lhs.__value_.__double_ == rhs.__value_.__double_;
    case cuda::std::__fmt_arg_t::__long_double:
      return lhs.__value_.__long_double_ == rhs.__value_.__long_double_;
    case cuda::std::__fmt_arg_t::__const_char_type_ptr:
      return lhs.__value_.__const_char_type_ptr_ == rhs.__value_.__const_char_type_ptr_;
    case cuda::std::__fmt_arg_t::__string_view:
      return lhs.__value_.__string_view_.data() == rhs.__value_.__string_view_.data()
          && lhs.__value_.__string_view_.size() == rhs.__value_.__string_view_.size();
    case cuda::std::__fmt_arg_t::__ptr:
      return lhs.__value_.__ptr_ == rhs.__value_.__ptr_;
    case cuda::std::__fmt_arg_t::__handle:
      return lhs.__value_.__handle_.__ptr_ == rhs.__value_.__handle_.__ptr_
          && lhs.__value_.__handle_.__format_ == rhs.__value_.__handle_.__format_;
    default:
      assert(false);
      return false;
  }
}

template <class CharT>
__host__ __device__ void test_get()
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  // 1. test packed format args
  {
    constexpr auto nargs = 3;
    static_assert(nargs <= cuda::std::__fmt_packed_types_max);

    auto n = 1;
    auto c = 'c';
    auto p = nullptr;

    auto store = cuda::std::make_format_args<Context>(n, c, p);
    auto args  = cuda::std::basic_format_args<Context>{store};

    assert(args.__size() == nargs);
    assert(test_format_arg_eq(
      args.get(0), cuda::std::basic_format_arg<Context>{cuda::std::__fmt_arg_t::__int, store.__storage.__values_[0]}));
    assert(test_format_arg_eq(
      args.get(1),
      cuda::std::basic_format_arg<Context>{cuda::std::__fmt_arg_t::__char_type, store.__storage.__values_[1]}));
    assert(test_format_arg_eq(
      args.get(2), cuda::std::basic_format_arg<Context>{cuda::std::__fmt_arg_t::__ptr, store.__storage.__values_[2]}));
    assert(!args.get(3));
  }

  // 2. test unpacked format args
  {
    constexpr auto nargs = 16;
    static_assert(nargs > cuda::std::__fmt_packed_types_max);

    auto n1 = 1;
    auto n2 = 2u;
    auto n3 = 3ll;
    auto n4 = 4ull;
    auto c1 = 'c';
    auto c2 = '1';
    auto c3 = '?';
    auto c4 = '*';
    auto p1 = nullptr;
    auto p2 = TEST_STRLIT(CharT, "test");
    auto p3 = static_cast<void*>(&n1);
    auto p4 = static_cast<const void*>(&n2);
    auto f1 = 3.14f;
    auto f2 = 10.0;
    auto f3 = 2.718281828459045;
    auto f4 = 1.4142135623730951f;

    auto store = cuda::std::make_format_args<Context>(n1, n2, n3, n4, c1, c2, c3, c4, p1, p2, p3, p4, f1, f2, f3, f4);
    auto args  = cuda::std::basic_format_args<Context>{store};

    assert(args.__size() == nargs);

    for (cuda::std::size_t i = 0; i < nargs; ++i)
    {
      assert(test_format_arg_eq(args.get(i), store.__storage.__args_[i]));
    }
    assert(!args.get(nargs));
  }
}

__host__ __device__ void test()
{
  test_get<char>();
#if _CCCL_HAS_WCHAR_T()
  test_get<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  return 0;
}
