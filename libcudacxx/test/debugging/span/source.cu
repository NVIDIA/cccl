// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/span>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using span_alias = cuda::std::span<int, 4>;

[[gnu::noinline]] void inspect_static(const cuda::std::span<int, 3>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_dynamic(const cuda::std::span<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_static_empty(const cuda::std::span<int, 0>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_default(const cuda::std::span<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_const(const cuda::std::span<const int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::span<cuda::std::array<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_alias(const span_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::span<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::span<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  int storage[3]                             = {-7, 0, 42};
  int wide_storage[4]                        = {-31, 17, 8, -64};
  const int const_storage[2]                 = {5, -9};
  cuda::std::array<int, 2> nested_storage[2] = {{{13, -5}}, {{0, 88}}};
  int mutable_storage[3]                     = {6, -91, 52};

  const cuda::std::span<int, 3> static_extent{storage};
  const cuda::std::span<int> dynamic{wide_storage, 4};
  const cuda::std::span<int, 0> static_empty{};
  const cuda::std::span<int> default_constructed{};
  const cuda::std::span<const int> const_elements{const_storage, 2};
  const cuda::std::span<cuda::std::array<int, 2>> nested{nested_storage, 2};
  const span_alias alias{wide_storage};
  const cuda::std::span<int> updated{mutable_storage, 3};

  inspect_static(static_extent);
  inspect_dynamic(dynamic);
  inspect_static_empty(static_empty);
  inspect_default(default_constructed);
  inspect_const(const_elements);
  inspect_nested(nested);
  inspect_alias(alias);
  inspect_before_update(updated);
  mutable_storage[0] = 3;
  mutable_storage[1] = 85;
  mutable_storage[2] = -12;
  inspect_after_update(updated);
}
