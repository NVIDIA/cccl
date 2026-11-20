// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-qualified-call
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

constexpr int target()
{
  return 1;
}

namespace cub
{
int namespace_initializer = target();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

inline int inline_initializer = target();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

void default_argument(int value = target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

struct object
{
  int member = target();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

  object() : member(target())
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  {}

  int method()
  {
    return target();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  }
};

void lambda_contexts()
{
  auto lambda = [value = target()] {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
    return target() + value;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  };
  (void) lambda();
  (void) [] {
    return target();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  }();
}

void local_class_context()
{
  struct local
  {
    int method()
    {
      return target();
      // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
      // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
    }
  };
}

using expression_type = decltype(target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

using function_type = void() noexcept(noexcept(target()));
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

inline constexpr int array_size = sizeof(target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

enum
{
  enumerator = target()
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
};

int array[target()];
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <int Value = target()>
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
struct default_template_argument {};

template <class T>
int variable_template = target();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

auto trailing_return() -> decltype(target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

int exception_specification() noexcept(noexcept(target()));
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

void default_argument_use()
{
  ::cub::default_argument();
}

extern "C"
{
int linkage_context()
{
  return target();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}
}


// CHECK-FIXES: DiagnosticName: cccl-qualified-call

// clang-format on
