// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/inplace_vector>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using nested_vector = cuda::std::inplace_vector<cuda::std::inplace_vector<int, 2>, 3>;

[[gnu::noinline]] void inspect_empty(const cuda::std::inplace_vector<int, 4>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_partial(const cuda::std::inplace_vector<int, 5>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_full(const cuda::std::inplace_vector<int, 3>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_zero_capacity(const cuda::std::inplace_vector<int, 0>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const nested_vector& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::inplace_vector<int, 4>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::inplace_vector<int, 4>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  const cuda::std::inplace_vector<int, 4> empty{};
  const cuda::std::inplace_vector<int, 5> partial = {-7, 0, 42};
  const cuda::std::inplace_vector<int, 3> full    = {1, 2, 3};
  const cuda::std::inplace_vector<int, 0> zero{};
  nested_vector nested;
  nested.push_back({13, -5});
  nested.push_back({88});
  cuda::std::inplace_vector<int, 4> updated = {6, -91};

  inspect_empty(empty);
  inspect_partial(partial);
  inspect_full(full);
  inspect_zero_capacity(zero);
  inspect_nested(nested);
  inspect_before_update(updated);
  updated.push_back(52);
  inspect_after_update(updated);
}
