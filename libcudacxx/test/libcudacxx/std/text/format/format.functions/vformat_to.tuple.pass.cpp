//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: bit field read/write is unsupported in tile code

// Limit the number of iterators that are being tested per each check(...) call.
// ADDITIONAL_COMPILE_DEFINITIONS: TEST_FORMAT_LIMITED_ITERATOR_TESTING

// <cuda/std/format>

// template<class Out>
//   Out vformat_to(Out out, string_view fmt, format_args args);
// template<class Out>
//    Out vformat_to(Out out, wstring_view fmt, wformat_args_t args);

#include "checkers/vformat_to.h"

#include "tests/tuple.h"
