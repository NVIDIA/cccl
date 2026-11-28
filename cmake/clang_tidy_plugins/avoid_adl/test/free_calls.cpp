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
int call(int);
template <class T>
int generic(T);
}

namespace cub
{
int local(int);
int overloaded(int);
int overloaded(double);

void ordinary_calls()
{
  local(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  overloaded(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  other_ns::call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::local(0);
  ::other_ns::call(0);
}

void using_declaration()
{
  using ::other_ns::call;
  call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

void using_directive()
{
  using namespace other_ns;
  call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::other_ns::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

namespace relative_alias = ::other_ns;
void namespace_alias()
{
  relative_alias::call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::relative_alias::call(0);
}

template <class T>
void templates(T value)
{
  other_ns::generic<T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  other_ns::generic(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::other_ns::generic<T>(value);
  ::other_ns::generic(value);
}

int recursive(int value)
{
  return value ? recursive(value - 1) : 0;
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
}

// Relative lookup can bind to a nested namespace rather than the global namespace.
namespace shadow_ns
{
int call(int);
}

namespace cub::shadow_ns
{
int call(int);
}

namespace cub
{
void shadowed_namespace()
{
  shadow_ns::call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::shadow_ns::call(0);
  ::cub::shadow_ns::call(0);
}
}

namespace other_ns
{
void target();
}

namespace cub
{
void target();

void parentheses()
{
  (target)();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ((target))();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '_CUB::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (other_ns::target)();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (::cub::target)();
  ((::other_ns::target))();
}

}

namespace other_ns
{
struct argument {};
int pick(argument);

struct hidden
{
  friend int hidden_pick(hidden);
};
}

namespace cub
{
int pick(...);

void adl_selects_external_function(other_ns::argument value)
{
  pick(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::other_ns::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::other_ns::pick(value);
}

void hidden_friend(other_ns::hidden value)
{
  hidden_pick(value);
}

void imported_function(other_ns::argument value)
{
  using ::other_ns::pick;
  pick(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

template <class T>
int unresolved_name(T value)
{
  return unknown_function(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: Replacements: []
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

// Parentheses disable ADL but leave ordinary unqualified lookup exposed.
void parenthesized_lookup()
{
  (pick)(0);
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
}

namespace cuda
{
struct completion {};

template <int Scope>
struct barrier;

// The constructor must find init through ADL before its hidden-friend definition.
template <>
struct barrier<0>
{
  barrier(int count, completion complete = {})
  {
    init(this, count, complete);
  }

  friend void init(barrier*, int, completion = {}) {}
};

template <>
struct barrier<1> : private barrier<0>
{
  using base = barrier<0>;

  friend void init(barrier* value, int count, completion complete = {})
  {
    init(static_cast<base*>(value), count, complete);
  }
};

void initialize(barrier<0>* value)
{
  init(value, 1);
}

struct visible;
void init(visible*, int);

// A namespace declaration makes this friend callable through qualified lookup.
struct visible
{
  friend void init(visible*, int);

  visible(int count)
  {
    init(this, count);
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
};
}

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
