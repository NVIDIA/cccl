// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-nodebug-on-aliases
// CLANG_TIDY_CONFIG_END

#include <cuda/std/detail/__config>

// CHECK-MESSAGES-NOT: warning:
using __internal = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: ' _CCCL_NODEBUG'

// CHECK-MESSAGES-NOT: warning:
using _Internal [[maybe_unused]] = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: ' _CCCL_NODEBUG'

// CHECK-MESSAGES-NOT: warning:
template <class T>
using __template_alias = T;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: ' _CCCL_NODEBUG'

// CHECK-MESSAGES-NOT: warning:
struct ordinary_record
{
  using __member /* Preserve the comment after the name. */ = int;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
  // CHECK-FIXES: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: ' _CCCL_NODEBUG'
};

// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-NOT: DiagnosticName:
using public_alias = int;
using _lowercase = int;
using _ = int;
using __annotated _CCCL_NODEBUG = int;
using _Annotated _CCCL_NODEBUG = int;
typedef int __legacy_typedef;

template <class T>
using __annotated_template _CCCL_NODEBUG = T;

void function_scope()
{
  using __local = int;
  using _Local = int;
  static_assert(sizeof(__local) == sizeof(int));
  static_assert(sizeof(_Local) == sizeof(int));
}
// clang-format on
