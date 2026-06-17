//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// Namespace std typedefs:
// using format_args = basic_format_args<format_context>;
// using wformat_args = basic_format_args<wformat_context>;

#include <cuda/std/__format_>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::format_args, cuda::std::basic_format_args<cuda::std::format_context>>);
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<cuda::std::wformat_args, cuda::std::basic_format_args<cuda::std::wformat_context>>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  return 0;
}
