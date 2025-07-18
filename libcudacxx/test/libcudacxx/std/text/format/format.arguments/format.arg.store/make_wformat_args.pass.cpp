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
//   format-arg-store<wformat_context, Args...>
//   make_wformat_args(Args&... args);

#include <cuda/std/__format_>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if _CCCL_HAS_WCHAR_T()
template <class Arg, class = void>
inline constexpr bool can_make_wformat_args = false;
template <class Arg>
inline constexpr bool
  can_make_wformat_args<Arg, cuda::std::void_t<decltype(cuda::std::make_wformat_args(cuda::std::declval<Arg>()))>> =
    true;

static_assert(can_make_wformat_args<int&>);
static_assert(!can_make_wformat_args<int>);
static_assert(!can_make_wformat_args<int&&>);

__host__ __device__ void test()
{
  auto i = 1;
  auto c = 'c';
  auto p = nullptr;
  auto b = false;

  [[maybe_unused]] auto store = cuda::std::make_wformat_args(i, p, b, c);
  static_assert(cuda::std::is_same_v<
                decltype(store),
                cuda::std::__format_arg_store<cuda::std::wformat_context, int, cuda::std::nullptr_t, bool, char>>);
}
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
#if _CCCL_HAS_WCHAR_T()
  test();
#endif // _CCCL_HAS_WCHAR_T()
  return 0;
}
