// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-qualified-call
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace other_ns
{
struct external
{
  static void call();
};
template <class T>
struct external_template
{
  static void call();
};
}

namespace cub
{
struct local
{
  static void call();
};
template <class T>
struct local_template
{
  static void call();
};
using local_alias = local;
using external_alias = ::other_ns::external;
template <class T>
using external_template_alias = ::other_ns::external_template<T>;

void same_namespace()
{
  local::call();
  local_alias::call();
  local_template<int>::call();
  local_template<other_ns::external>::call();
  ::cub::local::call();
}

void other_namespace()
{
  other_ns::external::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  other_ns::external_template<int>::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  external_alias::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  external_template_alias<int>::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::other_ns::external::call();
  ::other_ns::external_template<int>::call();
}

namespace child
{
struct child_type
{
  static void call();
};
void parent_type()
{
  local::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::local::call();
}
}

namespace sibling
{
struct sibling_type
{
  static void call();
};
}

void child_type()
{
  child::child_type::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  sibling::sibling_type::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  ::cub::child::child_type::call();
}

namespace child
{
void sibling_type()
{
  sibling::sibling_type::call();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}
}
}

namespace cub
{
void reopened_namespace()
{
  local::call();
  local_template<int>::call();
}
}

namespace cub
{
struct enclosing
{
  struct nested
  {
    static void call();
  };
  using local_alias = nested;
  using external_alias = ::other_ns::external;

  void operations()
  {
    nested::call();
    local_alias::call();
    external_alias::call();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  }
};

void nested_class_type()
{
  enclosing::nested::call();
}

extern "C"
{
struct linkage_type
{
  static void call();
};

void linkage_type_call()
{
  linkage_type::call();
}
}
}

// CHECK-FIXES: DiagnosticName: cccl-qualified-call

// clang-format on
