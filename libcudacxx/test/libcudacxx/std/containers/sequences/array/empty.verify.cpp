//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// nvrtc will not generate warnings/failures on nodiscard attribute
// UNSUPPORTED: nvrtc

// <cuda/std/array>

// class array

// bool empty() const noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cuda/std/array>

void f()
{
  cuda::std::array<int, 1> c;
  c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cuda::std::array<int, 0> c0;
  c0.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
