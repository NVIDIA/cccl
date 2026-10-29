// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/expected>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

struct error_info
{
  int code;
  int line;
};

using expected_alias = cuda::std::expected<int, error_info>;

[[gnu::noinline]] void inspect_value_int(const cuda::std::expected<int, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_value_double(const cuda::std::expected<double, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_value_zero(const cuda::std::expected<int, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_error_int(const cuda::std::expected<int, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_error_struct(const cuda::std::expected<int, error_info>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::expected<int, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::expected<int, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_alias(const expected_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::expected<int, int>, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_void_value(const cuda::std::expected<void, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_void_error(const cuda::std::expected<void, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  const cuda::std::expected<int, int> value_int_val{42};
  inspect_value_int(value_int_val);

  const cuda::std::expected<double, int> value_double_val{3.14};
  inspect_value_double(value_double_val);

  const cuda::std::expected<int, int> value_zero_val{0};
  inspect_value_zero(value_zero_val);

  const cuda::std::expected<int, int> error_int_val{cuda::std::unexpect, 7};
  inspect_error_int(error_int_val);

  const cuda::std::expected<int, error_info> error_struct_val{cuda::std::unexpect, error_info{5, 120}};
  inspect_error_struct(error_struct_val);

  cuda::std::expected<int, int> updated{1};
  inspect_before_update(updated);
  updated = cuda::std::unexpected<int>{2};
  inspect_after_update(updated);

  const expected_alias alias_val{17};
  inspect_alias(alias_val);

  const cuda::std::array<cuda::std::expected<int, int>, 2> nested_val = {
    {cuda::std::expected<int, int>{13}, cuda::std::expected<int, int>{cuda::std::unexpect, 9}}};
  inspect_nested(nested_val);

  const cuda::std::expected<void, int> void_value_val{};
  inspect_void_value(void_value_val);

  const cuda::std::expected<void, int> void_error_val{cuda::std::unexpect, 3};
  inspect_void_error(void_error_val);
}
