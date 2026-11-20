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

namespace cuda::std
{
inline namespace abi
{
template <class T>
constexpr int num_bits() { return sizeof(T) * 8; }

template <class T>
struct complex {};

// We can't resolve the specialization yet, but ADL has no call arguments to inspect.
template <class T>
inline constexpr int num_bits_v = num_bits<T>();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::cuda::std::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <class T>
inline constexpr int num_bits_v<complex<T>> = num_bits<T>() * 2;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
// CHECK-FIXES: FileOffset: [[#CALL:]]
// CHECK-FIXES: Replacements:
// CHECK-FIXES-NEXT: - FilePath:
// CHECK-FIXES-NEXT: Offset: [[#CALL]]
// CHECK-FIXES-NEXT: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '::cuda::std::'
// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

static_assert(num_bits_v<int> == sizeof(int) * 8);
static_assert(num_bits_v<complex<int>> == sizeof(int) * 16);
}
}

namespace other_ns
{
template <class T>
constexpr int imported() { return 1; }
}

namespace cub
{
using ::other_ns::imported;
}

namespace other_ns
{
template <class T, class U = void>
constexpr int imported() { return 2; }
}

namespace cub
{
// We must preserve the using declaration's overload set, even for unresolved calls.
template <class T>
constexpr int imported_call()
{
  return imported<T>();
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
static_assert(::cub::imported_call<int>() == 1);

template <class T>
constexpr int local_using()
{
  using ::cuda::std::num_bits;
  return num_bits<T>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
