// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-avoid-internal-linkage-in-headers
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: warning:
constexpr int constant = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: namespace-scope constexpr variable 'constant' in a header is not inline [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: 'inline '
// CHECK-MESSAGES-NOT: warning:
static constexpr int static_constant = 2;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: namespace-scope constexpr variable 'static_constant' in a header is not inline [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: inline
// CHECK-MESSAGES-NOT: warning:

namespace named {
template <class T>
constexpr int variable_template = sizeof(T);
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: namespace-scope constexpr variable 'variable_template' in a header is not inline [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: 'inline '
// CHECK-MESSAGES-NOT: warning:
static_assert(variable_template<int> > 0);
static_assert(variable_template<double> > 0);
}

#define CONSTANT(Name) constexpr int Name = 3
CONSTANT(from_macro);
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: namespace-scope constexpr variable 'from_macro' in a header is not inline [cccl-avoid-internal-linkage-in-headers]
// CHECK-MESSAGES-NOT: warning:

inline constexpr int inline_constant = 1;
template <class T>
inline constexpr int inline_template = sizeof(T);
const int ordinary_const = 1;
struct members {
  static constexpr int implicitly_inline = 1;
};
inline void locals()
{
  constexpr int local = 1;
  static constexpr int static_local = 1;
  static_assert(local == static_local);
}
namespace {
constexpr int anonymous_constant = 1;
}
static_assert(constant + static_constant + from_macro + ordinary_const + anonymous_constant == 8);
// clang-format on
