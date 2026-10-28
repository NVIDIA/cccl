// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-avoid-internal-linkage-in-headers
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: warning:
static void declared();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: static function 'declared' in a header has internal linkage [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: inline
// CHECK-MESSAGES-NOT: warning:
static inline void defined() {}
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: static function 'defined' in a header has internal linkage [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-MESSAGES-NOT: warning:

namespace named {
template <class T>
static void template_declared(T);
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: static function 'template_declared' in a header has internal linkage [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-MESSAGES-NOT: warning:
template <class T, int Len>
static T* template_defined()
{
  static T values[Len];
  return values;
}
// CHECK-MESSAGES: :[[@LINE-5]]:{{[0-9]+}}: warning: static function 'template_defined' in a header has internal linkage [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-MESSAGES-NOT: warning:
inline void instantiate_templates()
{
  (void) template_defined<int, 2>();
  (void) template_defined<double, 3>();
}

template <class T>
static constexpr int templated(T) { return 1; }
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: static function 'templated' in a header has internal linkage [cccl-avoid-internal-linkage-in-headers]
// CHECK-FIXES: DiagnosticName: cccl-avoid-internal-linkage-in-headers
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-MESSAGES-NOT: warning:
static_assert(templated(1) == 1);
static_assert(templated(1.0) == 1);
}

#define STATIC_FUNCTION(Name) static void Name() {}
STATIC_FUNCTION(from_macro)
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: static function 'from_macro' in a header has internal linkage [cccl-avoid-internal-linkage-in-headers]
// CHECK-MESSAGES-NOT: warning:

inline void external_inline() {}
constexpr int external_constexpr() { return 1; }
struct members {
  static void declared();
  static void defined() {}
};
namespace {
static void anonymous_function() {}
}
// clang-format on
