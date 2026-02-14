//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// ceil

// template <class Rep, class Period>
//   constexpr duration<Rep, Period> abs(duration<Rep, Period> d)

// This function shall not participate in overload resolution unless numeric_limits<Rep>::is_signed is true.

#include <cuda/std/chrono>

using unsigned_secs = cuda::std::chrono::duration<unsigned>;

int main(int, char**)
{
  cuda::std::chrono::abs(unsigned_secs(0));

  return 0;
}
