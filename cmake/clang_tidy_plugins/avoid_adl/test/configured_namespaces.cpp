// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-avoid-adl
// CheckOptions: cccl-avoid-adl.Namespaces=library;cuda
// CheckOptions: cccl-avoid-adl.NamespaceMacros=library=LIBRARY::;other_ns=OTHER_NS::
// CLANG_TIDY_CONFIG_END

#define LIBRARY library
#define OTHER_NS other_ns

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace other_ns { void call(); void external(); }

namespace library
{
void call();

void example()
{
  call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: 'LIBRARY::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::library::call();
}

namespace detail
{
void call();
void example()
{
  call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: 'LIBRARY::detail::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}

}

namespace cuda
{
void call();
void example()
{
  call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::cuda::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

void imported()
{
  using namespace ::other_ns;
  external();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: 'OTHER_NS::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}

namespace cub { void example() { other_ns::call(); } }
namespace library_extra { void example() { other_ns::call(); } }
namespace user::library { void example() { other_ns::call(); } }

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
