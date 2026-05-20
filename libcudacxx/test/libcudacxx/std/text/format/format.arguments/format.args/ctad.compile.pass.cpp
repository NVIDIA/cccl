//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class Context, class... Args>
//   basic_format_args(format-arg-store<Context, Args...>) -> basic_format_args<Context>;

#include <cuda/std/__format_>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

using namespace cuda::std;

static_assert(is_same_v<format_args, decltype(basic_format_args{make_format_args(declval<int&>())})>);
#if _CCCL_HAS_WCHAR_T()
static_assert(is_same_v<wformat_args, decltype(basic_format_args{make_wformat_args(declval<int&>())})>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  return 0;
}
