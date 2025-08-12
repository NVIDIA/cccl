//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class... Args>
//   basic_format_args(const format-arg-store<Context, Args...>& store) noexcept;

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

template <class CharT>
__host__ __device__ void test_constructor()
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  auto i = 1;
  auto c = 'c';
  auto p = nullptr;

  static_assert(!cuda::std::is_default_constructible_v<cuda::std::basic_format_args<Context>>);

  {
    auto store = cuda::std::make_format_args<Context>(i);
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::basic_format_args<Context>, decltype(store)>);
    cuda::std::basic_format_args<Context> format_args{store};
    assert(format_args.get(0));
    assert(!format_args.get(1));
  }
  {
    auto store = cuda::std::make_format_args<Context>(i, c);
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::basic_format_args<Context>, decltype(store)>);
    cuda::std::basic_format_args<Context> format_args{store};
    assert(format_args.get(0));
    assert(format_args.get(1));
    assert(!format_args.get(2));
  }
  {
    auto store = cuda::std::make_format_args<Context>(i, c, p);
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::basic_format_args<Context>, decltype(store)>);
    cuda::std::basic_format_args<Context> format_args{store};
    assert(format_args.get(0));
    assert(format_args.get(1));
    assert(format_args.get(2));
    assert(!format_args.get(3));
  }
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
