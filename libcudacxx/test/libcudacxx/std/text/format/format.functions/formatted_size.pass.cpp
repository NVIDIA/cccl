//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class... Args>
//   size_t formatted_size(format-string<Args...> fmt, const Args&... args);
// template<class... Args>
//   size_t formatted_size(wformat-string<Args...> fmt, const Args&... args);

#include <cuda/std/__format_>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::std::formatted_size(""))>);
static_assert(!noexcept(cuda::std::formatted_size("")));
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::std::formatted_size(L""))>);
static_assert(!noexcept(cuda::std::formatted_size(L"")));
#endif // _CCCL_HAS_WCHAR_T()

#include "checkers/formatted_size.h"
#include "tests/smoke.h"
