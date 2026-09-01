// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using array_alias = cuda::std::array<int, 4>;

[[gnu::noinline]] void inspect_normal(const cuda::std::array<int, 3>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::std::array<int, 0>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::array<int, 2>, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_alias(const array_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::array<int, 3>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::array<int, 3>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  const cuda::std::array<int, 3> normal                      = {-7, 0, 42};
  const cuda::std::array<int, 0> empty                       = {};
  const cuda::std::array<cuda::std::array<int, 2>, 2> nested = {{{13, -5}, {0, 88}}};
  const array_alias alias                                    = {-31, 17, 8, -64};
  cuda::std::array<int, 3> updated_values                    = {6, -91, 52};

  inspect_normal(normal);
  inspect_empty(empty);
  inspect_nested(nested);
  inspect_alias(alias);
  inspect_before_update(updated_values);
  updated_values = {3, 85, -12};
  inspect_after_update(updated_values);
}
