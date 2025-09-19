//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class ToDuration, class Rep, class Period>
//   ToDuration
//   duration_cast(const duration<Rep, Period>& d);

// ToDuration shall be an instantiation of duration.

#include <cuda/std/chrono>

int main(int, char**)
{
  cuda::std::chrono::duration_cast<int>(cuda::std::chrono::milliseconds(3));

  return 0;
}
