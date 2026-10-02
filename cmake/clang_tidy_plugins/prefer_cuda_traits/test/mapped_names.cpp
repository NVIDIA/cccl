// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-prefer-cuda-traits
// CheckOptions: cccl-prefer-cuda-traits.Traits=::first::foo,::first::bar;::second::foo,::second::bar;::plain,::replaced_plain
// CLANG_TIDY_CONFIG_END

namespace first
{
inline namespace version
{
template <typename T> struct foo {};
template <typename T> inline constexpr bool foo_v = true;
template <typename T> using foo_t = T;
} // namespace version
template <typename T> struct bar {};
template <typename T> inline constexpr bool bar_v = true;
template <typename T> using bar_t = T;
} // namespace first

namespace second
{
template <typename T> struct foo {};
template <typename T> inline constexpr bool foo_v = true;
template <typename T> using foo_t = T;
template <typename T> struct bar {};
template <typename T> inline constexpr bool bar_v = true;
template <typename T> using bar_t = T;
} // namespace second

template <typename T> struct plain {};
template <typename T> struct replaced_plain {};

// Identical base names must retain their separate namespace mappings, including inline namespaces.
// CHECK-MESSAGES-NOT: warning:
template <typename T>
void collisions()
{
  [[maybe_unused]] first::foo<T> first_type;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'first::bar'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}first::bar{{'?}}{{$}}
  [[maybe_unused]] second::foo<T> second_type;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'second::bar'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}second::bar{{'?}}{{$}}
  static_assert(first::foo_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'first::bar_v'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}first::bar_v{{'?}}{{$}}
  static_assert(second::foo_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'second::bar_v'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}second::bar_v{{'?}}{{$}}
  using first_alias [[maybe_unused]] = first::foo_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'first::bar_t'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}first::bar_t{{'?}}{{$}}
  using second_alias [[maybe_unused]] = second::foo_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'second::bar_t'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}second::bar_t{{'?}}{{$}}
}

template <typename T>
void unqualified_names()
{
  using first::foo;
  using second::foo_v;
  [[maybe_unused]] foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'first::bar'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}first::bar{{'?}}{{$}}
  static_assert(foo_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'second::bar_v'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}second::bar_v{{'?}}{{$}}
  [[maybe_unused]] plain<T> global;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'replaced_plain'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}replaced_plain{{'?}}{{$}}
}

namespace cuda::std
{
template <typename T>
void standard_namespace()
{
  [[maybe_unused]] ::first::foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::first::bar'
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: ReplacementText: {{'?}}::first::bar{{'?}}{{$}}
}
} // namespace cuda::std

// Instantiations must not repeat diagnostics from the template definitions.
template void collisions<int>();
template void unqualified_names<int>();
template void cuda::std::standard_namespace<int>();

// clang-format on
