//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__utility/typeid.h>

// template <auto _Vp>
// cuda::std::__string_view cuda::std::__pretty_nameof_v() noexcept

#include <cuda/std/__utility/typeid.h>
#include <cuda/std/cassert>

#include "test_macros.h"

// The exact spelling of a value is not guaranteed to be identical across
// compilers (e.g. an unsigned literal might be spelled "42" or "42U", a char
// "'A'" or 65, an enumerator by name or by a cast). Integral and boolean values
// such as `42` and `true` are spelled identically everywhere, so those are the
// ones checked exactly here.

// The smoke test in the header is guarded the same way; mirror that here.
#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG)

static_assert(cuda::std::__pretty_nameof_v<42>() == cuda::std::__string_view("42"), "");
static_assert(cuda::std::__pretty_nameof_v<-7>() == cuda::std::__string_view("-7"), "");
static_assert(cuda::std::__pretty_nameof_v<0>() == cuda::std::__string_view("0"), "");

static_assert(cuda::std::__pretty_nameof_v<true>() == cuda::std::__string_view("true"), "");
static_assert(cuda::std::__pretty_nameof_v<false>() == cuda::std::__string_view("false"), "");

// Distinct values yield distinct spellings.
static_assert(cuda::std::__pretty_nameof_v<42>() != cuda::std::__pretty_nameof_v<43>(), "");
static_assert(cuda::std::__pretty_nameof_v<true>() != cuda::std::__pretty_nameof_v<false>(), "");

#endif // !_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF && !_CCCL_BROKEN_MSVC_FUNCSIG

int main(int, char**)
{
#if !defined(_CCCL_BROKEN_MSVC_FUNCSIG)
  // Runtime checks on the host. The constexpr path is exercised by the
  // static_asserts above (in both the host and device compilation passes).
  NV_IF_TARGET(NV_IS_HOST,
               (assert(cuda::std::__pretty_nameof_v<42>() == cuda::std::__string_view("42"));
                assert(cuda::std::__pretty_nameof_v<-7>() == cuda::std::__string_view("-7"));
                assert(cuda::std::__pretty_nameof_v<true>() == cuda::std::__string_view("true"));
                assert(cuda::std::__pretty_nameof_v<42>() != cuda::std::__pretty_nameof_v<43>());))
#endif // !_CCCL_BROKEN_MSVC_FUNCSIG

  return 0;
}
