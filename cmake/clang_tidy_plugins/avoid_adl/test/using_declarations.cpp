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
constexpr int pick(long) { return 1; }
}

namespace cub
{
using ::other_ns::pick;
}

namespace other_ns
{
constexpr int pick(int) { return 2; }
}

namespace cub
{
// This using declaration only exposes pick(long). Qualifying through other_ns would select pick(int) instead.
constexpr int imported_overload()
{
  return pick(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

static_assert(::cub::imported_overload() == 1);

inline namespace abi
{
constexpr int inline_target() { return 3; }
}

constexpr int inline_overload()
{
  return inline_target();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

static_assert(::cub::inline_overload() == 3);
}

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
