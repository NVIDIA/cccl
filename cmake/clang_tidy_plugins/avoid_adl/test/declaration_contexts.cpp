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

namespace other_ns
{
constexpr int target()
{
  return 1;
}
}
using other_ns::target;

namespace cub
{
int namespace_initializer = target();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

inline int inline_initializer = target();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

void default_argument(int value = target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

struct object
{
  int member = target();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

  object() : member(target())
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  {}

  int method()
  {
    return target();
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
};

void lambda_contexts()
{
  auto lambda = [value = target()] {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
    return target() + value;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
    // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
    // CHECK-FIXES: FileOffset: [[#CALL:]]
    // CHECK-FIXES: Replacements:
    // CHECK-FIXES-NEXT: - FilePath:
    // CHECK-FIXES-NEXT: Offset: [[#CALL]]
    // CHECK-FIXES-NEXT: Length: 0
    // CHECK-FIXES-NEXT: ReplacementText: '::'
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  };
  (void) lambda();
  (void) [] {
    return target();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
    // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
    // CHECK-FIXES: FileOffset: [[#CALL:]]
    // CHECK-FIXES: Replacements:
    // CHECK-FIXES-NEXT: - FilePath:
    // CHECK-FIXES-NEXT: Offset: [[#CALL]]
    // CHECK-FIXES-NEXT: Length: 0
    // CHECK-FIXES-NEXT: ReplacementText: '::'
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
  };
}

using expression_type = decltype(target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

using function_type = void() noexcept(noexcept(target()));
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

inline constexpr int array_size = sizeof(target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

enum
{
  enumerator = target()
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
};

int array[target()];
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <int Value = target()>
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
struct default_template_argument {};

template <class T>
int variable_template = target();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

auto trailing_return() -> decltype(target());
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

int exception_specification() noexcept(noexcept(target()));
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::'
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
}

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace cub
{
// These functions belong to cub, but namespace lookup can't find their block-scope declarations.
void resolved()
{
  void local();
  local();
}

void explicit_extern()
{
  extern void external();
  external();
}

template <class T>
void unresolved(T value)
{
  void overloaded(int);
  void overloaded(long);
  overloaded(value);
}

void target();

void namespace_scope()
{
  target();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
}
}

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
