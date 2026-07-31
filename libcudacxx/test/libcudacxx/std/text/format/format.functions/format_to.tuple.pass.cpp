//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Limit the number of iterators that are being tested per each check(...) call.
// ADDITIONAL_COMPILE_DEFINITIONS: TEST_FORMAT_LIMITED_ITERATOR_TESTING

// <cuda/std/format>

// template<class Out, class... Args>
//   Out format_to(Out out, format-string<Args...> fmt, const Args&... args);
// template<class Out, class... Args>
//   Out format_to(Out out, wformat-string<Args...> fmt, const Args&... args);

#include "checkers/format_to.h"

#include "tests/tuple.h"
