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

// Reading the implicit bounds of __traits for an element type without a cuda::std::numeric_limits specialization must
// fail to compile rather than silently yielding a value-initialized (and therefore meaningless) bound. This exercises
// the __traits_impl primary-template path, which is the bound surface read by generic consumers.
struct unspecialized_type
{};

[[maybe_unused]] constexpr auto invalid_lowest = cuda::args::__traits<unspecialized_type>::lowest;

int main(int, char**)
{
  return 0;
}
