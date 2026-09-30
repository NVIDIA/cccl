// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// Checks: cccl-prefer-cuda-traits

#include <cuda/std/type_traits>
#include <cuda/type_traits>

template <typename T>
bool branch_on_trait()
{
  if constexpr (cuda::std::is_trivially_copyable_v<T>)
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
  // CHECK-FIXES: Length: 34
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
  {
    return true;
  }
  return false;
}

template <typename T, bool = cuda::std::is_trivially_copyable_v<T>>
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
// CHECK-FIXES: Length: 34
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
struct default_argument
{};

template <typename T>
cuda::std::enable_if_t<cuda::std::is_trivially_copyable<T>::value, bool> constrained_function(T)
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable' instead of 'cuda::std::is_trivially_copyable<T>' for generic types [cccl-prefer-cuda-traits]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
// CHECK-FIXES: Length: 32
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable{{'?}}{{$}}
{
  return true;
}

template <typename T>
bool exception_specification() noexcept(cuda::std::is_trivially_copyable_v<T>)
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'cuda::is_trivially_copyable_v' instead of 'cuda::std::is_trivially_copyable_v' for generic types [cccl-prefer-cuda-traits]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-cuda-traits
// CHECK-FIXES: Length: 34
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}cuda::is_trivially_copyable_v{{'?}}{{$}}
{
  return true;
}
// clang-format on
