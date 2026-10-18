// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>

#define REMOVE_CONST(T) cuda::std::remove_const_t<T>
#define REMOVE_VOLATILE(T) cuda::std::remove_volatile_t<T>
#define REMOVE_CONST_VOLATILE(T) cuda::std::remove_const_t<cuda::std::remove_volatile_t<T>>
#define REMOVE_VOLATILE_CONST(T) cuda::std::remove_volatile_t<cuda::std::remove_const_t<T>>

// CHECK-MESSAGES-NOT: warning:
template <typename = REMOVE_CONST(cuda::std::remove_volatile_t<const volatile int>)>
struct const_outer_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = cuda::std::remove_const_t<REMOVE_VOLATILE(const volatile int)>>
struct volatile_inner_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = REMOVE_CONST_VOLATILE(const volatile int)>
struct const_volatile_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = REMOVE_VOLATILE(cuda::std::remove_const_t<const volatile int>)>
struct volatile_outer_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = cuda::std::remove_volatile_t<REMOVE_CONST(const volatile int)>>
struct const_inner_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = REMOVE_VOLATILE_CONST(const volatile int)>
struct volatile_const_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []
// clang-format on
