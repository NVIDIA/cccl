//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Before GCC 6, aggregate initialization kicks in.
// See https://stackoverflow.com/q/41799015/627587.
// UNSUPPORTED: gcc-5

// <memory>

// struct allocator_arg_t { explicit allocator_arg_t() = default; };
// const allocator_arg_t allocator_arg = allocator_arg_t();

// This test checks for LWG 2510.

#include <cuda/std/__memory_>

cuda::std::allocator_arg_t f()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**)
{
  return 0;
}
