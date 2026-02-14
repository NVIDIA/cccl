//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// cuda::std::__fmt_arg_t

#include <cuda/std/__format_>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<cuda::std::__fmt_arg_t>, cuda::std::uint8_t>);

static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__none) == 0);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__boolean) == 1);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__char_type) == 2);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__int) == 3);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__long_long) == 4);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__unsigned) == 5);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__unsigned_long_long) == 6);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__float) == 7);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__double) == 8);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__long_double) == 9);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__const_char_type_ptr) == 10);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__string_view) == 11);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__ptr) == 12);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__handle) == 13);

int main(int, char**)
{
  return 0;
}
