// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/optional>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using optional_alias = cuda::std::optional<int>;

// A payload with a user-provided destructor selects the non-trivial storage base.
struct non_trivial
{
  int id;
  double weight;
  ~non_trivial() {}
};

[[gnu::noinline]] void inspect_engaged(const cuda::std::optional<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_disengaged(const cuda::std::optional<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::optional<cuda::std::optional<int>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested_disengaged(const cuda::std::optional<cuda::std::optional<int>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_non_trivial(const cuda::std::optional<non_trivial>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_alias(const optional_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_reference(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_reference_disengaged(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::optional<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::optional<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  const cuda::std::optional<int> engaged                                = 42;
  const cuda::std::optional<int> disengaged                             = cuda::std::nullopt;
  const cuda::std::optional<cuda::std::optional<int>> nested            = cuda::std::optional<int>{7};
  const cuda::std::optional<cuda::std::optional<int>> nested_disengaged = cuda::std::optional<int>{};
  const cuda::std::optional<non_trivial> non_trivial_value              = non_trivial{3, 2.5};
  const optional_alias alias                                            = -8;
  int referenced                                                        = 11;
  const cuda::std::optional<int&> reference(referenced);
  const cuda::std::optional<int&> reference_disengaged{};
  cuda::std::optional<int> updated = 99;

  inspect_engaged(engaged);
  inspect_disengaged(disengaged);
  inspect_nested(nested);
  inspect_nested_disengaged(nested_disengaged);
  inspect_non_trivial(non_trivial_value);
  inspect_alias(alias);
  inspect_reference(reference);
  inspect_reference_disengaged(reference_disengaged);
  inspect_before_update(updated);
  // The payload storage keeps its old bytes after reset(); only the engaged flag changes.
  updated.reset();
  inspect_after_update(updated);
}
