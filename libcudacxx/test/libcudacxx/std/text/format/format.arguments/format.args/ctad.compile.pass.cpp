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

static_assert(
  cuda::std::is_same_v<cuda::std::format_args,
                       decltype(cuda::std::basic_format_args{cuda::std::make_format_args(cuda::std::declval<int&>())})>);
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<
              cuda::std::wformat_args,
              decltype(cuda::std::basic_format_args{cuda::std::make_wformat_args(cuda::std::declval<int&>())})>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  return 0;
}
