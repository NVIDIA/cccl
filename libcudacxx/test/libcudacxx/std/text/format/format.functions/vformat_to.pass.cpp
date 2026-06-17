//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class Out>
//   Out vformat_to(Out out, string_view fmt, format_args args);
// template<class Out>
//    Out vformat_to(Out out, wstring_view fmt, wformat_args_t args);

#include <cuda/std/__format_>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

static_assert(cuda::std::is_same_v<char*,
                                   decltype(cuda::std::vformat_to(cuda::std::declval<char*>(),
                                                                  cuda::std::declval<cuda::std::string_view>(),
                                                                  cuda::std::declval<cuda::std::format_args>()))>);
static_assert(!noexcept(cuda::std::vformat_to(cuda::std::declval<char*>(),
                                              cuda::std::declval<cuda::std::string_view>(),
                                              cuda::std::declval<cuda::std::format_args>())));
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<wchar_t*,
                                   decltype(cuda::std::vformat_to(cuda::std::declval<wchar_t*>(),
                                                                  cuda::std::declval<cuda::std::wstring_view>(),
                                                                  cuda::std::declval<cuda::std::wformat_args>()))>);
static_assert(!noexcept(cuda::std::vformat_to(cuda::std::declval<wchar_t*>(),
                                              cuda::std::declval<cuda::std::wstring_view>(),
                                              cuda::std::declval<cuda::std::wformat_args>())));
#endif // _CCCL_HAS_WCHAR_T()

#include "checkers/vformat_to.h"
#include "tests/smoke.h"
