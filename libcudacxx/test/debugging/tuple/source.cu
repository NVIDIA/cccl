// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/tuple>

#include <vector>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

// An empty, non-final type triggers the tuple element's empty-base-class
// optimization instead of storing the element in a __value_ data member.
struct empty_type
{};

[[gnu::noinline]] void inspect_basic(const cuda::std::tuple<int, double, char>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::std::tuple<>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_single(const cuda::std::tuple<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ebco(const cuda::std::tuple<int, empty_type, double>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// An empty leading element shares its storage offset with the element that
// follows it, and an empty trailing element shares the offset of whatever
// precedes it. Two adjacent empty elements share that same offset with each
// other too. Debuggers that build per-element values by (parent, offset) can
// mix these up unless every case is exercised on its own.
[[gnu::noinline]] void inspect_ebco_first(const cuda::std::tuple<empty_type, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ebco_last(const cuda::std::tuple<int, empty_type>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_ebco_adjacent(const cuda::std::tuple<empty_type, empty_type, int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_pointer(const cuda::std::tuple<int*, const char*>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested_empty(const cuda::std::tuple<int, cuda::std::tuple<>, double>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::tuple<int, cuda::std::tuple<double, char>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_reference(const cuda::std::tuple<int&, double&>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_const(const cuda::std::tuple<int, double>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// An optimized build inlines every use of std::vector::operator[], so no
// out-of-line copy is left for the debugger to call. Instantiate that one member
// explicitly to keep a copy, which lets the debugger evaluate values[i].
template typename std::vector<cuda::std::tuple<int, int>>::const_reference
  std::vector<cuda::std::tuple<int, int>>::operator[](size_type) const;

[[gnu::noinline]] void inspect_vector(const std::vector<cuda::std::tuple<int, int>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::tuple<int, double>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::tuple<int, double>& values)
{
  KEEP_FOR_DEBUGGER(values);
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

  const cuda::std::tuple<empty_type, int> ebco_first_values{empty_type{}, 21};
  inspect_ebco_first(ebco_first_values);

  const cuda::std::tuple<int, empty_type> ebco_last_values{22, empty_type{}};
  inspect_ebco_last(ebco_last_values);

  const cuda::std::tuple<empty_type, empty_type, int> ebco_adjacent_values{empty_type{}, empty_type{}, 23};
  inspect_ebco_adjacent(ebco_adjacent_values);

  int pointee = 13;
  const cuda::std::tuple<int*, const char*> pointer_values{&pointee, "tuple"};
  inspect_pointer(pointer_values);

  const cuda::std::tuple<int, cuda::std::tuple<>, double> nested_empty_values{1, cuda::std::tuple<>{}, 2.5};
  inspect_nested_empty(nested_empty_values);

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
