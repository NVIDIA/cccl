//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class Out, class... Args>
//   Out format_to(Out out, format-string<Args...> fmt, const Args&... args);
// template<class Out, class... Args>
//   Out format_to(Out out, wformat-string<Args...> fmt, const Args&... args);

#include <cuda/std/__format_>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

static_assert(cuda::std::is_same_v<char*, decltype(cuda::std::format_to(cuda::std::declval<char*>(), ""))>);
static_assert(!noexcept(cuda::std::format_to(cuda::std::declval<char*>(), "")));
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<wchar_t*, decltype(cuda::std::format_to(cuda::std::declval<wchar_t*>(), ""))>);
static_assert(!noexcept(cuda::std::format_to(cuda::std::declval<wchar_t*>(), "")));
#endif // _CCCL_HAS_WCHAR_T()

#include "checkers/format_to.h"
#include "tests/smoke.h"
