// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <type_traits>

// CHECK-MESSAGES-NOT: warning:
template <typename T>
using const_volatile = cuda::std::add_const_t<cuda::std::add_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using volatile_const = cuda::std::add_volatile_t<cuda::std::add_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using const_cv = cuda::std::remove_const_t<cuda::std::remove_cv_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using volatile_cv = cuda::std::remove_volatile_t<cuda::std::remove_cv_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using cv_cvref = cuda::std::remove_cv_t<cuda::std::remove_cvref_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cvref_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using reference_cvref = cuda::std::remove_reference_t<cuda::std::remove_cvref_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cvref_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using decay_reference = cuda::std::decay_t<cuda::std::remove_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::decay_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using decay_decay = cuda::std::decay_t<cuda::std::decay_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::decay_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using unwrap_decay = cuda::std::unwrap_reference_t<cuda::std::decay_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::unwrap_ref_decay_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using pointer_reference = cuda::std::add_pointer_t<cuda::std::remove_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_pointer_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using lvalue_rvalue = cuda::std::add_lvalue_reference_t<cuda::std::add_rvalue_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_lvalue_reference_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using rvalue_lvalue = cuda::std::add_rvalue_reference_t<cuda::std::add_lvalue_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_lvalue_reference_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace traits = cuda::std;
using concrete_cv = traits::add_const_t<traits::add_volatile_t<int>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::std::decay_t;
using concrete_decay = decay_t<decay_t<const int[3]>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::decay_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

using concrete_unwrap = cuda::std::unwrap_reference_t<cuda::std::decay_t<const cuda::std::reference_wrapper<int>&>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::unwrap_ref_decay_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

// Removing cv-qualifiers before decay changes array element qualification.
template <typename T>
using decay_cv = cuda::std::decay_t<cuda::std::remove_cv_t<T>>;
template <typename T>
using decay_cvref = cuda::std::decay_t<cuda::std::remove_cvref_t<T>>;

// Removing cv-qualifiers before references preserves qualification on the referred type.
template <typename T>
using reference_cv = cuda::std::remove_reference_t<cuda::std::remove_cv_t<T>>;

template <typename T>
using host_cv = std::add_const_t<std::add_volatile_t<T>>;
template <typename T>
using mixed_cv = cuda::std::add_const_t<std::add_volatile_t<T>>;
template <typename T>
using mixed_reference = std::add_pointer_t<cuda::std::remove_reference_t<T>>;

// A local alias must retain its own declaration identity.
template <typename T>
using local_decay = cuda::std::decay_t<T>;
template <typename T>
using local_outer = local_decay<cuda::std::remove_reference_t<T>>;
template <typename T>
using local_inner = cuda::std::decay_t<local_decay<T>>;

static_assert(cuda::std::is_same_v<const_volatile<int>, const volatile int>);
static_assert(cuda::std::is_same_v<volatile_const<void>, const volatile void>);
static_assert(cuda::std::is_same_v<const_volatile<int&>, int&>);
static_assert(cuda::std::is_same_v<volatile_const<int()>, int()>);
static_assert(cuda::std::is_same_v<const_cv<const volatile int[3]>, int[3]>);
static_assert(cuda::std::is_same_v<volatile_cv<const volatile int&>, const volatile int&>);
static_assert(cuda::std::is_same_v<cv_cvref<const volatile int&>, int>);
static_assert(cuda::std::is_same_v<reference_cvref<const volatile int(&)[3]>, int[3]>);
static_assert(cuda::std::is_same_v<decay_reference<const int(&)[3]>, const int*>);
static_assert(cuda::std::is_same_v<decay_reference<int(&)()>, int(*)()>);
static_assert(cuda::std::is_same_v<decay_decay<const int[3]>, const int*>);
static_assert(cuda::std::is_same_v<decay_decay<void>, void>);
static_assert(cuda::std::is_same_v<unwrap_decay<const cuda::std::reference_wrapper<int>&>, int&>);
static_assert(cuda::std::is_same_v<unwrap_decay<const int&>, int>);
static_assert(cuda::std::is_same_v<pointer_reference<const int&>, const int*>);
static_assert(cuda::std::is_same_v<pointer_reference<int(&)[3]>, int(*)[3]>);
static_assert(cuda::std::is_same_v<pointer_reference<void>, void*>);
static_assert(cuda::std::is_same_v<pointer_reference<int() const>, int() const>);
static_assert(cuda::std::is_same_v<lvalue_rvalue<int&&>, int&>);
static_assert(cuda::std::is_same_v<rvalue_lvalue<int>, int&>);
static_assert(cuda::std::is_same_v<lvalue_rvalue<void>, void>);
static_assert(cuda::std::is_same_v<rvalue_lvalue<int() const>, int() const>);
static_assert(cuda::std::is_same_v<concrete_cv, const volatile int>);
static_assert(cuda::std::is_same_v<concrete_decay, const int*>);
static_assert(cuda::std::is_same_v<concrete_unwrap, int&>);
static_assert(cuda::std::is_same_v<decay_cv<const int[3]>, int*>);
static_assert(cuda::std::is_same_v<decay_cvref<const int(&)[3]>, int*>);
static_assert(cuda::std::is_same_v<reference_cv<const int&>, const int>);
// clang-format on
