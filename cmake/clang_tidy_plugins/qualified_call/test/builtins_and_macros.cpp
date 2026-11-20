// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-qualified-call
// CLANG_TIDY_CONFIG_END

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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::__builtin_user_function();
}

void builtin_argument()
{
  (void) __builtin_expect(value(), 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) __builtin_expect(::cub::value(), 1);
}
}

int target();
namespace other_ns
{
int target();
}

#define CALL_BARE() target()
#define CALL_RELATIVE() other_ns::target()
#define CALL_GLOBAL() ::target()
#define EXPECT_TRUE(expression) __builtin_expect(static_cast<bool>(expression), 1)
#define FORWARD_EXPRESSION(expression) (expression)

namespace cub
{
void macro_calls()
{
  CALL_BARE();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  CALL_RELATIVE();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  CALL_GLOBAL();
  (void) EXPECT_TRUE(true);
  (void) EXPECT_TRUE(CALL_BARE());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) FORWARD_EXPRESSION(target());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
END_CUDA_STD

// CHECK-FIXES: DiagnosticName: cccl-qualified-call

// clang-format on
