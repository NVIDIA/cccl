// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/mdspan>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

// An explicit bool remains eligible, even when the value is dependent.
// CHECK-MESSAGES-NOT: warning:
template <bool Value>
using explicit_bool = cuda::std::integral_constant<bool, Value>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: ReplacementText: ''
// CHECK-FIXES-NOT: DiagnosticName:

// Instantiation must not turn a generic value type into a written bool argument.
template <typename... Types>
struct type_list {};
template <typename T, T... Values>
struct type_value_list : type_list<cuda::std::integral_constant<T, Values>...> {};
template <typename T, T Value>
struct dependent_constant : cuda::std::integral_constant<T, Value>
{
  using type = cuda::std::integral_constant<T, Value>;
  type member;
};
template <typename T, T Value>
auto make_constant() -> cuda::std::integral_constant<T, Value>
{
  return cuda::std::integral_constant<T, Value>{};
}
template <typename T, T Value>
using dependent_alias = cuda::std::integral_constant<T, Value>;
template <typename Traits>
struct nested_type
{
  using type = cuda::std::integral_constant<typename Traits::value_type, Traits::value>;
  type member;
};

// These rules also remove a type argument, but only for explicitly written index types.
template <typename Index>
struct dependent_indices
{
  cuda::std::integer_sequence<Index, 0, 1> sequence;
  cuda::std::make_integer_sequence<Index, 2> generated_sequence;
  cuda::std::dextents<Index, 2> extents;
};

template <typename Offset>
struct dependent_strides
{
  cuda::dstrides<Offset, 2> strides;
};

// A substituted value must not be mistaken for a bool literal in the source.
template <bool Value>
struct dependent_boolean
{
  cuda::std::bool_constant<Value> member;
};

void instantiate_traits()
{
  [[maybe_unused]] type_value_list<bool, true, false> boolean_values;
  [[maybe_unused]] type_value_list<int, 0, 2> integer_values;
  [[maybe_unused]] dependent_constant<bool, true> boolean_constant;
  [[maybe_unused]] dependent_constant<int, 2> integer_constant;
  [[maybe_unused]] auto boolean_result = make_constant<bool, false>();
  [[maybe_unused]] auto integer_result = make_constant<int, 2>();
  [[maybe_unused]] dependent_alias<bool, true> boolean_alias;
  [[maybe_unused]] dependent_alias<int, 2> integer_alias;
  [[maybe_unused]] nested_type<cuda::std::true_type> nested_boolean;
  [[maybe_unused]] nested_type<dependent_alias<int, 2>> nested_integer;
  [[maybe_unused]] dependent_indices<cuda::std::size_t> size_indices;
  [[maybe_unused]] dependent_indices<cuda::std::ptrdiff_t> difference_indices;
  [[maybe_unused]] dependent_indices<int> integer_indices;
  [[maybe_unused]] dependent_strides<cuda::std::ptrdiff_t> difference_strides;
  [[maybe_unused]] dependent_strides<int> integer_strides;
  [[maybe_unused]] dependent_boolean<true> true_value;
  [[maybe_unused]] dependent_boolean<false> false_value;

  static_assert(decltype(boolean_constant)::value);
  static_assert(decltype(integer_constant)::value == 2);
  static_assert(!decltype(boolean_result)::value);
  static_assert(decltype(integer_result)::value == 2);
  static_assert(cuda::std::is_same_v<decltype(size_indices.extents), cuda::std::dims<2>>);
  static_assert(!cuda::std::is_same_v<decltype(integer_indices.extents), cuda::std::dims<2>>);
  static_assert(cuda::std::is_same_v<decltype(difference_strides.strides), cuda::steps<2>>);
}
// clang-format on
