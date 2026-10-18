// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>
#include <type_traits>

// CHECK-MESSAGES-NOT: warning:
using const_outer = cuda::std::remove_const_t<cuda::std::remove_volatile_t<const volatile int>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 54
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

using volatile_outer = cuda::std::remove_volatile_t<cuda::std::remove_const_t<const volatile int>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 54
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using dependent_const = cuda::std::remove_const_t<cuda::std::remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
using dependent_volatile = cuda::std::remove_volatile_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

namespace traits = cuda::std;
template <typename T>
using alias_const = traits::remove_const_t<traits::remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
using alias_volatile = traits::remove_volatile_t<traits::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

using cuda::std::remove_const_t;
using cuda::std::remove_volatile_t;
template <typename T>
using imported_const = remove_const_t<remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
using imported_volatile = remove_volatile_t<remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

using spaced = cuda::std::remove_const_t < cuda::std::remove_volatile_t < const volatile int > >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 57
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 2
// CHECK-FIXES-NEXT: ReplacementText: ''

// The name lists also match pairs that have no simplification rule.
template <typename T>
using cv_const = cuda::std::remove_cv_t<cuda::std::remove_const_t<T>>;
template <typename T>
using cv_volatile = cuda::std::remove_cv_t<cuda::std::remove_volatile_t<T>>;
template <typename T>
using const_reference = cuda::std::remove_const_t<cuda::std::remove_reference_t<T>>;
template <typename T>
using const_const = cuda::std::remove_const_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_const_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''
template <typename T>
using volatile_reference = cuda::std::remove_volatile_t<cuda::std::remove_reference_t<T>>;
template <typename T>
using volatile_volatile = cuda::std::remove_volatile_t<cuda::std::remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_volatile_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

using host_const = std::remove_const_t<std::remove_volatile_t<const volatile int>>;
using host_volatile = std::remove_volatile_t<std::remove_const_t<const volatile int>>;
template <typename T>
using mixed_const = cuda::std::remove_const_t<std::remove_volatile_t<T>>;
template <typename T>
using mixed_volatile = cuda::std::remove_volatile_t<std::remove_const_t<T>>;
template <typename T>
using host_outer_const = std::remove_const_t<cuda::std::remove_volatile_t<T>>;
template <typename T>
using host_outer_volatile = std::remove_volatile_t<cuda::std::remove_const_t<T>>;

// An alias of a trait must retain its own declaration identity.
template <typename T>
using local_const = cuda::std::remove_const_t<T>;
template <typename T>
using local_volatile = cuda::std::remove_volatile_t<T>;
template <typename T>
using local_outer = local_const<cuda::std::remove_volatile_t<T>>;
template <typename T>
using local_inner = cuda::std::remove_const_t<local_volatile<T>>;

static_assert(cuda::std::is_same_v<const_outer, int>);
static_assert(cuda::std::is_same_v<volatile_outer, int>);
static_assert(cuda::std::is_same_v<dependent_const<const volatile int&>, const volatile int&>);
static_assert(cuda::std::is_same_v<dependent_volatile<const volatile int*>, const volatile int*>);
// clang-format on
