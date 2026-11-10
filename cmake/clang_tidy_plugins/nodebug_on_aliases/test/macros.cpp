// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-nodebug-on-aliases
// CLANG_TIDY_CONFIG_END

// Macro-generated declarations must not change the shared macro definition or its arguments.
#define INTERNAL_ALIAS using __macro_alias = int
#define NAMED_ALIAS(Name) using Name = int
#define ALIAS_NAME __expanded_name
#define TRAIT_ALIAS using type = int

// CHECK-MESSAGES-NOT: warning:
INTERNAL_ALIAS;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
// CHECK-FIXES: FileOffset:
// CHECK-FIXES-NEXT: Replacements: []

// CHECK-MESSAGES-NOT: warning:
NAMED_ALIAS(__argument_alias);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
// CHECK-FIXES: FileOffset:
// CHECK-FIXES-NEXT: Replacements: []

// CHECK-MESSAGES-NOT: warning:
using ALIAS_NAME = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Internal aliases should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
// CHECK-FIXES: FileOffset:
// CHECK-FIXES-NEXT: Replacements: []

// CHECK-MESSAGES-NOT: warning:
struct macro_trait
{
  TRAIT_ALIAS;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: The alias of type traits should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
  // CHECK-FIXES: FileOffset:
  // CHECK-FIXES-NEXT: Replacements: []
};
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-NOT: DiagnosticName:
// clang-format on
