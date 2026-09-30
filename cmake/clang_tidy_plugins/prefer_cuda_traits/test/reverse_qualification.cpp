// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// Checks: cccl-prefer-cuda-traits
// CheckOptions: cccl-prefer-cuda-traits.Traits=::cuda::foo,::cuda::std::foo

namespace cuda
{
template <typename T>
struct foo
{};

template <typename T>
inline constexpr bool foo_v = true;

template <typename T>
using foo_t = T;

namespace std
{
template <typename T>
struct foo
{};

template <typename T>
inline constexpr bool foo_v = true;

template <typename T>
using foo_t = T;
} // namespace std

template <typename T>
void enclosing_namespace()
{
  foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 3
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}std::foo{{'?}}{{$}}
  static_assert(foo_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::foo_v' instead of 'cuda::foo_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 5
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}std::foo_v{{'?}}{{$}}
  using type [[maybe_unused]] = foo_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::foo_t' instead of 'T' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 5
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}std::foo_t{{'?}}{{$}}
  ::cuda::foo<T> global;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::cuda::std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 11
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::foo{{'?}}{{$}}
}
} // namespace cuda

template <typename T>
void qualified_names()
{
  cuda::foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 9
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo{{'?}}{{$}}
  ::cuda::foo<T> global;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::cuda::std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 11
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::foo{{'?}}{{$}}
}

template <typename T>
void using_declarations()
{
  using cuda::foo;
  using cuda::foo_v;
  using cuda::foo_t;

  foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 3
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo{{'?}}{{$}}
  static_assert(foo_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo_v' instead of 'cuda::foo_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 5
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo_v{{'?}}{{$}}
  using type [[maybe_unused]] = foo_t<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo_t' instead of 'T' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 5
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo_t{{'?}}{{$}}
}

template <typename T>
void using_directive()
{
  using namespace cuda;

  foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 3
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo{{'?}}{{$}}
  static_assert(foo_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo_v' instead of 'cuda::foo_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 5
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo_v{{'?}}{{$}}
}

namespace some_random_alias_name = cuda;

template <typename T>
void namespace_alias()
{
  some_random_alias_name::foo<T> value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::std::foo' instead of 'cuda::foo<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 27
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::std::foo{{'?}}{{$}}
}

// clang-format on
