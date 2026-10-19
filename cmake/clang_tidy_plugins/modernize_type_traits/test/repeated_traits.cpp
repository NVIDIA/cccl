// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>
#include <type_traits>

// CHECK-MESSAGES-NOT: warning:
template <typename T>
using repeated_add_const_t = cuda::std::add_const_t<cuda::std::add_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_const_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_add_volatile_t = cuda::std::add_volatile_t<cuda::std::add_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_volatile_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_add_cv_t = cuda::std::add_cv_t<cuda::std::add_cv_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_remove_const_t = cuda::std::remove_const_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_const_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_remove_volatile_t = cuda::std::remove_volatile_t<cuda::std::remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_volatile_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_remove_cv_t = cuda::std::remove_cv_t<cuda::std::remove_cv_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_remove_reference_t = cuda::std::remove_reference_t<cuda::std::remove_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_reference_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_remove_cvref_t = cuda::std::remove_cvref_t<cuda::std::remove_cvref_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cvref_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_add_lvalue_reference_t = cuda::std::add_lvalue_reference_t<cuda::std::add_lvalue_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_lvalue_reference_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using repeated_add_rvalue_reference_t = cuda::std::add_rvalue_reference_t<cuda::std::add_rvalue_reference_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::add_rvalue_reference_t{{'?}}{{$}}
// CHECK-FIXES: Length: 1
// CHECK-FIXES-NEXT: ReplacementText: ''

// Pointer and extent removal can remove another level on each application.
template <typename T>
using repeated_pointer = cuda::std::remove_pointer_t<cuda::std::remove_pointer_t<T>>;
template <typename T>
using repeated_extent = cuda::std::remove_extent_t<cuda::std::remove_extent_t<T>>;
template <typename T>
using repeated_add_pointer = cuda::std::add_pointer_t<cuda::std::add_pointer_t<T>>;

template <typename T>
using host_repeated = std::remove_cv_t<std::remove_cv_t<T>>;
template <typename T>
using mixed_repeated = cuda::std::remove_cv_t<std::remove_cv_t<T>>;

static_assert(cuda::std::is_same_v<repeated_add_const_t<int>, const int>);
static_assert(cuda::std::is_same_v<repeated_add_const_t<int&>, int&>);
static_assert(cuda::std::is_same_v<repeated_add_volatile_t<int>, volatile int>);
static_assert(cuda::std::is_same_v<repeated_add_volatile_t<int()>, int()>);
static_assert(cuda::std::is_same_v<repeated_add_cv_t<int>, const volatile int>);
static_assert(cuda::std::is_same_v<repeated_add_cv_t<void>, const volatile void>);
static_assert(cuda::std::is_same_v<repeated_remove_const_t<const volatile int>, volatile int>);
static_assert(cuda::std::is_same_v<repeated_remove_volatile_t<const volatile int>, const int>);
static_assert(cuda::std::is_same_v<repeated_remove_cv_t<const volatile int>, int>);
static_assert(cuda::std::is_same_v<repeated_remove_cv_t<const int[3]>, int[3]>);
static_assert(cuda::std::is_same_v<repeated_remove_reference_t<const int&>, const int>);
static_assert(cuda::std::is_same_v<repeated_remove_reference_t<const int&&>, const int>);
static_assert(cuda::std::is_same_v<repeated_remove_cvref_t<const volatile int&>, int>);
static_assert(cuda::std::is_same_v<repeated_remove_cvref_t<const int(&)[3]>, int[3]>);
static_assert(cuda::std::is_same_v<repeated_add_lvalue_reference_t<int&&>, int&>);
static_assert(cuda::std::is_same_v<repeated_add_lvalue_reference_t<void>, void>);
static_assert(cuda::std::is_same_v<repeated_add_rvalue_reference_t<int>, int&&>);
static_assert(cuda::std::is_same_v<repeated_add_rvalue_reference_t<int&>, int&>);
static_assert(cuda::std::is_same_v<repeated_add_rvalue_reference_t<int() const>, int() const>);
static_assert(cuda::std::is_same_v<repeated_pointer<int**>, int>);
static_assert(cuda::std::is_same_v<repeated_extent<int[2][3]>, int>);
static_assert(cuda::std::is_same_v<repeated_add_pointer<int>, int**>);
// clang-format on
