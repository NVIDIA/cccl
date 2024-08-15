//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

#include <cuda/memory_resource>
#include <cuda/std/type_traits>

using resource = cuda::mr::device_memory_resource;
static_assert(!cuda::std::is_trivial<resource>::value, "");
static_assert(!cuda::std::is_trivially_default_constructible<resource>::value, "");
static_assert(cuda::std::is_trivially_copy_constructible<resource>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<resource>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<resource>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<resource>::value, "");
static_assert(cuda::std::is_trivially_destructible<resource>::value, "");
static_assert(!cuda::std::is_empty<resource>::value, "");

int main(int, char**)
{
  return 0;
}
