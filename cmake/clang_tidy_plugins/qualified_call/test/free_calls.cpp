// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-qualified-call
// CLANG_TIDY_CONFIG_END

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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  overloaded(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  other_ns::call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::local(0);
  ::other_ns::call(0);
}

void using_declaration()
{
  using ::other_ns::call;
  call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

void using_directive()
{
  using namespace other_ns;
  call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

namespace relative_alias = ::other_ns;
void namespace_alias()
{
  relative_alias::call(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::relative_alias::call(0);
}

template <class T>
void templates(T value)
{
  other_ns::generic<T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  other_ns::generic(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::other_ns::generic<T>(value);
  ::other_ns::generic(value);
}

int recursive(int value)
{
  return value ? recursive(value - 1) : 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ((target))();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (other_ns::target)();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::other_ns::pick(value);
}

template <class T>
int dependent_adl(T value)
{
  return pick(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

void hidden_friend(other_ns::hidden value)
{
  hidden_pick(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

void imported_function(other_ns::argument value)
{
  using ::other_ns::pick;
  pick(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

template <class T>
int unresolved_name(T value)
{
  return unknown_function(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

// Parentheses disable ADL but leave ordinary unqualified lookup exposed.
void parenthesized_lookup()
{
  (pick)(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}

// CHECK-FIXES: DiagnosticName: cccl-qualified-call

// clang-format on
