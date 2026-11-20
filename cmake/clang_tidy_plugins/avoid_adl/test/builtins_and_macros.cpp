// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-avoid-adl
// CheckOptions: cccl-avoid-adl.Namespaces=cub;cuda
// CheckOptions: cccl-avoid-adl.NamespaceMacros=cub=_CUB::
// CLANG_TIDY_CONFIG_END

#define _CUB cub

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace cub
{
long value();
void __builtin_user_function();

void compiler_builtins(char* output, const char* input, int& integer)
{
  (void) __builtin_expect(integer != 0, 1);
  (void) __builtin_strlen(input);
  (void) __builtin_memcpy(output, input, 1);
  (void) __builtin_addressof(integer);
  (void) ::__builtin_expect(integer != 0, 1);
}

template <class T>
long dependent_builtin(T condition)
{
  return __builtin_expect(static_cast<bool>(condition), 1);
}

void ordinary_function_with_builtin_prefix()
{
  __builtin_user_function();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::__builtin_user_function();
}

void builtin_argument()
{
  (void) __builtin_expect(value(), 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) __builtin_expect(::cub::value(), 1);
}
}

int target();
int set_device(int);
namespace other_ns
{
int target();
}

#define CALL_BARE() target()
#define CALL_RELATIVE() other_ns::target()
#define CALL_GLOBAL() ::target()
#define EXPECT_TRUE(expression) __builtin_expect(static_cast<bool>(expression), 1)
#define FORWARD_EXPRESSION(expression) (expression)
#define CHECK_ERROR(expression) static_cast<void>(expression)

namespace cub
{
// The call is written in the macro argument, so we can qualify it at the call site.
void macro_argument(int device)
{
  CHECK_ERROR(set_device(device));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

void macro_calls()
{
  CALL_BARE();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  CALL_RELATIVE();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  CALL_GLOBAL();
  (void) EXPECT_TRUE(true);
  (void) EXPECT_TRUE(CALL_BARE());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) FORWARD_EXPRESSION(target());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

#define CALL_DEFINED_IN_LIBRARY() target()
}

void outside_library()
{
  CALL_DEFINED_IN_LIBRARY();
}

#define BEGIN_CUDA_STD namespace cuda { namespace std {
#define END_CUDA_STD } }

BEGIN_CUDA_STD
void generated_namespace()
{
  target();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
END_CUDA_STD

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace other_ns
{
void call();
}

#define FUNCTION call

namespace cub
{
void partially_expanded_call()
{
  other_ns::FUNCTION();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}

#undef FUNCTION

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
