// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-avoid-adl
// CheckOptions: cccl-avoid-adl.Namespaces=cub;cuda
// CheckOptions: cccl-avoid-adl.NamespaceMacros=cub=_CUB::
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

#define _CUB cub

namespace cub::foo::bar::baz
{
void call();
void baz_extra();

void partial_name()
{
  baz_extra();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::foo::bar::baz::'
}
}

namespace cub::foo::bar
{
void one_component()
{
  baz::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::foo::bar::'
}
}

namespace cub::foo
{
void multiple_components()
{
  bar::baz::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::foo::'
}
}

namespace cub::repeat::repeat
{
void call();
}

namespace cub
{
void full_namespace()
{
  cub::foo::bar::baz::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  ::cub::foo::bar::baz::call();
}

void repeated_components()
{
  repeat::repeat::call();
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
