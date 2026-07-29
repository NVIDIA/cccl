// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/tuple>

#include <vector>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

// An empty, non-final type triggers the tuple element's empty-base-class
// optimization instead of storing the element in a __value_ data member.
struct empty_type
{};

[[gnu::noinline]] void inspect_basic(const cuda::std::tuple<int, double, char>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::std::tuple<>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_single(const cuda::std::tuple<int>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_ebco(const cuda::std::tuple<int, empty_type, double>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::tuple<int, cuda::std::tuple<double, char>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_reference(const cuda::std::tuple<int&, double&>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_const(const cuda::std::tuple<int, double>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_vector(const std::vector<cuda::std::tuple<int, int>>& values)
{
  keep_for_debugger(values[0]);
  keep_for_debugger(values[1]);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::tuple<int, double>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::tuple<int, double>& values)
{
  keep_for_debugger(values);
}

int main()
{
  const cuda::std::tuple<int, double, char> basic_values{42, 3.14, 'x'};
  inspect_basic(basic_values);

  const cuda::std::tuple<> empty_values{};
  inspect_empty(empty_values);

  const cuda::std::tuple<int> single_values{7};
  inspect_single(single_values);

  const cuda::std::tuple<int, empty_type, double> ebco_values{9, empty_type{}, 2.5};
  inspect_ebco(ebco_values);

  const cuda::std::tuple<int, cuda::std::tuple<double, char>> nested_values{1, cuda::std::tuple<double, char>{2.5, 'z'}};
  inspect_nested(nested_values);

  int referenced_int       = 5;
  double referenced_double = 6.5;
  const cuda::std::tuple<int&, double&> reference_values{referenced_int, referenced_double};
  inspect_reference(reference_values);

  const cuda::std::tuple<int, double> const_values{11, -12.5};
  inspect_const(const_values);

  std::vector<cuda::std::tuple<int, int>> tuple_vector;
  tuple_vector.emplace_back(1, 2);
  tuple_vector.emplace_back(3, 4);
  inspect_vector(tuple_vector);

  cuda::std::tuple<int, double> mutable_values{1, 2.5};
  inspect_before_update(mutable_values);
  cuda::std::get<0>(mutable_values) = 9;
  cuda::std::get<1>(mutable_values) = -3.5;
  inspect_after_update(mutable_values);
}
