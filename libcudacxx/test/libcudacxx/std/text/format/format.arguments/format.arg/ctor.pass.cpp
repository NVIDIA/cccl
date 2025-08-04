//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// basic_format_arg() noexcept;

// The class has several exposition only private constructors. These are tested
// in visit_format_arg.pass.cpp

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

template <class CharT>
__host__ __device__ void test_constructor()
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::basic_format_arg<Context>>);

  cuda::std::basic_format_arg<Context> format_arg{};
  assert(!format_arg);
}

__host__ __device__ void test()
{
  test_constructor<char>();
#if _CCCL_HAS_CHAR8_T()
  test_constructor<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_constructor<char16_t>();
  test_constructor<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_constructor<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
