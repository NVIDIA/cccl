//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// void advance_to(iterator it);

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class CharT>
__host__ __device__ void test_advance_to()
{
  using Container = cuda::std::inplace_vector<CharT, 3>;
  using OutIt     = cuda::std::__back_insert_iterator<Container>;
  using Context   = cuda::std::basic_format_context<OutIt, CharT>;

  Container container{};

  auto store   = cuda::std::make_format_args<Context>();
  auto args    = cuda::std::basic_format_args{store};
  auto context = cuda::std::__fmt_make_format_context(OutIt{container}, args);

  static_assert(cuda::std::is_same_v<void, decltype(context.advance_to(cuda::std::declval<OutIt>()))>);

  {
    auto it = context.out();
    *it     = CharT('a');
    context.advance_to(cuda::std::move(it));

    assert(container.size() == 1);
    assert(container[0] == CharT('a'));
  }
  {
    auto it = context.out();
    *it     = CharT('b');
    context.advance_to(cuda::std::move(it));

    assert(container.size() == 2);
    assert(container[1] == CharT('b'));
  }
  {
    auto it = context.out();
    *it     = CharT('c');
    context.advance_to(cuda::std::move(it));

    assert(container.size() == 3);
    assert(container[2] == CharT('c'));
  }
}

__host__ __device__ void test()
{
  test_advance_to<char>();
#if _CCCL_HAS_CHAR8_T()
  test_advance_to<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_advance_to<char16_t>();
  test_advance_to<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_advance_to<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
