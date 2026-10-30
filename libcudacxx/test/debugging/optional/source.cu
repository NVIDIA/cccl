// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/optional>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using optional_alias     = cuda::std::optional<int>;
using optional_ref_alias = cuda::std::optional<int&>;

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

[[gnu::noinline]] void inspect_engaged_zero(const cuda::std::optional<int>& values)
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

[[gnu::noinline]] void inspect_alias(const optional_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::optional<int>, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// Reference specialization tests
[[gnu::noinline]] void inspect_ref_disengaged(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ref_engaged(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ref_engaged_zero(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ref_before_update(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ref_after_update(const cuda::std::optional<int&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ref_alias(const optional_ref_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ref_nested(const cuda::std::array<cuda::std::optional<int&>, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  // Normal optional tests
  const cuda::std::optional<int> disengaged_val{};
  inspect_disengaged(disengaged_val);

  const cuda::std::optional<int> engaged_int_val{42};
  inspect_engaged_int(engaged_int_val);

  const cuda::std::optional<double> engaged_double_val{3.14};
  inspect_engaged_double(engaged_double_val);

  const cuda::std::optional<int> engaged_zero_val{0};
  inspect_engaged_zero(engaged_zero_val);

  cuda::std::optional<int> updated{1};
  inspect_before_update(updated);
  updated = 2;
  inspect_after_update(updated);

  const optional_alias alias_val{17};
  inspect_alias(alias_val);

  const cuda::std::array<cuda::std::optional<int>, 2> nested_val = {{{13}, {}}};
  inspect_nested(nested_val);

  // Reference specialization tests
  const cuda::std::optional<int&> ref_disengaged_val{};
  inspect_ref_disengaged(ref_disengaged_val);

  int ref_target = 88;
  const cuda::std::optional<int&> ref_engaged_val{ref_target};
  inspect_ref_engaged(ref_engaged_val);

  int zero_val = 0;
  const cuda::std::optional<int&> ref_engaged_zero_val{zero_val};
  inspect_ref_engaged_zero(ref_engaged_zero_val);

  int ref_update_target_1 = 1;
  int ref_update_target_2 = 2;
  cuda::std::optional<int&> ref_updated{ref_update_target_1};
  inspect_ref_before_update(ref_updated);
  ref_updated = ref_update_target_2;
  inspect_ref_after_update(ref_updated);

  const optional_ref_alias ref_alias_val{ref_target};
  inspect_ref_alias(ref_alias_val);

  const cuda::std::array<cuda::std::optional<int&>, 2> ref_nested_val = {{{ref_target}, {}}};
  inspect_ref_nested(ref_nested_val);
}
