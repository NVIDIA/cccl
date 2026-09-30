// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// Checks: cccl-prefer-cuda-traits

#include <cuda/std/type_traits>
#include <cuda/type_traits>

template <typename T>
void qualified_names()
{
  static_assert(cuda::std::is_trivially_copyable<T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(::cuda::std::is_trivially_copyable<T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(cuda::std::is_trivially_copyable_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
  static_assert(::cuda::std::is_trivially_copyable_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use '::cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 36
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::is_trivially_copyable_v{{'?}}{{$}}
}

namespace standard = cuda::std;

template <typename T>
void namespace_alias()
{
  static_assert(standard::is_trivially_copyable<T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 31
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(standard::is_trivially_copyable_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 33
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
}

template <typename T>
void using_declarations()
{
  using cuda::std::is_trivially_copyable;
  using cuda::std::is_trivially_copyable_v;

  static_assert(is_trivially_copyable<T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 21
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(is_trivially_copyable_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 23
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
}

template <typename T>
void using_directive()
{
  using namespace cuda::std;

  static_assert(is_trivially_copyable<T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 21
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(is_trivially_copyable_v<T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 23
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
}
// clang-format on
