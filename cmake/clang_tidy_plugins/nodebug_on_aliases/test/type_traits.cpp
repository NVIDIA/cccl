// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-nodebug-on-aliases
// CLANG_TIDY_CONFIG_END

#include <cuda/std/detail/__config>

// CHECK-MESSAGES-NOT: warning:
template <class T>
struct trait
{
  using type = T;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: The alias of type traits should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
  // CHECK-FIXES: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: ' _CCCL_NODEBUG'
};

// CHECK-MESSAGES-NOT: warning:
template <class T, int N, template <class> class Wrapper>
struct trait_with_helper
{
  using __helper _CCCL_NODEBUG = Wrapper<T>;
  using type = T[N];
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: The alias of type traits should always be marked _CCCL_NODEBUG [cccl-nodebug-on-aliases]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-nodebug-on-aliases
  // CHECK-FIXES: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: ' _CCCL_NODEBUG'
};

// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-NOT: DiagnosticName:
template <class T>
struct annotated_trait
{
  using type _CCCL_NODEBUG = T;
};

struct record_with_data
{
  using type = int;
  int value;
};

struct record_with_method
{
  using type = int;
  void method();
};

struct record_with_public_alias
{
  using type = int;
  using value_type = int;
};

using type = int;
trait<int>::type instance;
// clang-format on
