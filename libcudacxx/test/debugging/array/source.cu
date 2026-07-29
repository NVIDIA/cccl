// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

using array_alias = cuda::std::array<int, 4>;

[[gnu::noinline]] void inspect_normal(const cuda::std::array<int, 3>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::std::array<int, 0>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::array<int, 2>, 2>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_alias(const array_alias& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::array<int, 3>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::array<int, 3>& values)
{
  keep_for_debugger(values);
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
