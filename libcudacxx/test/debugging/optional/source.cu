// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/optional>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

[[gnu::noinline]] void inspect_disengaged(const cuda::std::optional<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_engaged_int(const cuda::std::optional<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_engaged_double(const cuda::std::optional<double>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  const cuda::std::optional<int> disengaged_val{};
  inspect_disengaged(disengaged_val);

  const cuda::std::optional<int> engaged_int_val{42};
  inspect_engaged_int(engaged_int_val);

  const cuda::std::optional<double> engaged_double_val{3.14};
  inspect_engaged_double(engaged_double_val);
}
