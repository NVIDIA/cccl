// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// Checks: cccl-prefer-cuda-traits
// CheckOptions: cccl-prefer-cuda-traits.Traits=::foo::bar::baz,::foo::replacement_baz

namespace foo
{
namespace bar
{
template <typename T>
struct baz
{};

template <typename T>
inline constexpr bool baz_v = true;

template <typename T>
using baz_t = T;
} // namespace bar

template <typename T>
struct replacement_baz
{};

template <typename T>
inline constexpr bool replacement_baz_v = true;

template <typename T>
using replacement_baz_t = T;
} // namespace foo

template <typename T>
void qualified_names()
{
  foo::bar::baz<T> qualified;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'foo::replacement_baz' instead of 'foo::bar::baz<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 13
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}foo::replacement_baz{{'?}}{{$}}
  ::foo::bar::baz<T> global;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::foo::replacement_baz' instead of 'foo::bar::baz<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 15
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::foo::replacement_baz{{'?}}{{$}}
  static_assert(foo::bar::baz_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'foo::replacement_baz_v' instead of 'foo::bar::baz_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 15
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}foo::replacement_baz_v{{'?}}{{$}}
  static_assert(::foo::bar::baz_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::foo::replacement_baz_v' instead of 'foo::bar::baz_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 17
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::foo::replacement_baz_v{{'?}}{{$}}
  using qualified_alias [[maybe_unused]] = foo::bar::baz_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'foo::replacement_baz_t' instead of 'T' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 15
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}foo::replacement_baz_t{{'?}}{{$}}
  using global_alias [[maybe_unused]] = ::foo::bar::baz_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::foo::replacement_baz_t' instead of 'T' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 17
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::foo::replacement_baz_t{{'?}}{{$}}
}

namespace foo
{
template <typename T>
void relative_names()
{
  bar::baz<T> relative;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'replacement_baz' instead of 'foo::bar::baz<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 8
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}replacement_baz{{'?}}{{$}}
  static_assert(bar::baz_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'replacement_baz_v' instead of 'foo::bar::baz_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 10
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}replacement_baz_v{{'?}}{{$}}
  using relative_alias [[maybe_unused]] = bar::baz_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'replacement_baz_t' instead of 'T' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 10
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}replacement_baz_t{{'?}}{{$}}
}
} // namespace foo

// clang-format on
