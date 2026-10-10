//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>

// A type without a cuda::std::numeric_limits specialization has no meaningful implicit bounds. Default-constructing
// runtime_bounds for such a type must be rejected at compile time instead of silently producing a degenerate range.
struct unspecialized_type
{};

[[maybe_unused]] cuda::args::runtime_bounds<unspecialized_type> invalid_bounds{};

int main(int, char**)
{
  return 0;
}
