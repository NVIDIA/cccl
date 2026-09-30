// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// Checks: cccl-prefer-cuda-traits

#include <cuda/std/type_traits>
#include <cuda/type_traits>

template <typename T>
void dependent_types()
{
  static_assert(cuda::std::is_trivially_copyable<const T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<const T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(cuda::std::is_trivially_copyable<T*>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T *>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(cuda::std::is_trivially_copyable<T[4]>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T[4]>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(cuda::std::is_trivially_copyable_v<volatile T>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
  static_assert(cuda::std::is_trivially_copyable_v<typename T::value_type>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
}

template <typename T>
struct wrapper
{
  T value;
};

template <typename T>
void composite_types()
{
  static_assert(cuda::std::is_trivially_copyable<wrapper<T>>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<wrapper<T>>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(cuda::std::is_trivially_copyable_v<wrapper<T>>);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
}

template <typename T>
using copyable_trait = cuda::std::is_trivially_copyable<T>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
// CHECK-FIXES: Length: 32
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}

template <typename T>
struct inherited_trait : cuda::std::is_trivially_copyable<T>
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
// CHECK-FIXES: Length: 32
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
{};

template <typename T>
void trait_objects()
{
  cuda::std::is_trivially_copyable<T> trait;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  (void) trait;
  (void) cuda::std::is_trivially_copyable<T>{};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 32
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
}
// clang-format on
