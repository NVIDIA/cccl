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
struct local
{
  static void call();
};
using local_alias = local;
}

namespace other_ns
{
using alias = ::cub::local;
template <class T>
using alias_template = ::cub::local;
struct outer
{
  using inner = ::cub::local;
  struct nested : ::cub::local {};
};
}

namespace cub
{
// The underlying type is local, but the written name comes from another namespace.
void external_aliases()
{
  other_ns::alias::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  other_ns::alias_template<int>::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  other_ns::outer::inner::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  other_ns::outer::nested::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
}

void decltype_alias()
{
  decltype(other_ns::outer{})::inner::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
}

namespace detail
{
void parent_alias()
{
  local_alias::call();
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
}

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace other_ns
{
void call();
struct external
{
  static void call();
};
}

namespace cub
{
void local_type_alias()
{
  using traits = ::other_ns::external;
  traits::call();
}

void local_namespace_alias()
{
  namespace local = ::other_ns;
  local::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  local::external::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

// The alias has the same name as a template in the enclosing namespace.
template <class T, int Threads>
struct BlockRakingLayout
{
  static T* PlacementPtr(T*, int);
};

namespace detail
{
template <class T, int Threads>
struct BlockScanRaking
{
  using BlockRakingLayout = BlockRakingLayout<T, Threads>;
  T* scan(T* storage, int thread)
  {
    return BlockRakingLayout::PlacementPtr(storage, thread);
  }
};

template <class T>
struct aliases
{
  using traits = ::other_ns::external;
  typedef ::other_ns::external legacy_traits;
  template <class U>
  using rebound = ::other_ns::external;

  void call()
  {
    traits::call();
    legacy_traits::call();
    rebound<T>::call();
  }
};
}

void local_typedef()
{
  typedef ::other_ns::external traits;
  traits::call();
}
}

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
