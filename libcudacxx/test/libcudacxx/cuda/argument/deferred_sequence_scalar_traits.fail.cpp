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

using traits = cuda::argument::__traits<cuda::argument::deferred_sequence<int>>;

[[maybe_unused]] constexpr bool invalid_traits = traits::is_deferred;

int main(int, char**)
{
  return 0;
}
