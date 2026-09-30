// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// Checks: cccl-prefer-cuda-traits

#include <cuda/std/type_traits>
#include <cuda/type_traits>

// Clang can replace the complete COPYABLE_TRAIT expansion or a name inside a macro argument.
// COPYABLE_VALUE includes template arguments outside the name range, so Clang cannot replace its partial expansion.
#define COPYABLE_TRAIT       cuda::std::is_trivially_copyable
#define COPYABLE_VALUE(Type) cuda::std::is_trivially_copyable_v<Type>
#define IDENTITY(Value) Value

template <typename T>
void macro_uses()
{
  static_assert(COPYABLE_TRAIT<T>::value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 14
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
  static_assert(COPYABLE_VALUE(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: FileOffset:
  // CHECK-FIXES-NEXT: Replacements: []
  static_assert(IDENTITY(cuda::std::is_trivially_copyable_v<T>));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
}

#undef COPYABLE_TRAIT
#undef COPYABLE_VALUE
#undef IDENTITY
// clang-format on
