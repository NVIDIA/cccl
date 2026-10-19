// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

// Both namespace levels can contain inline ABI namespaces.
namespace cuda
{
inline namespace abi
{
namespace std
{
inline namespace version
{
template <typename T>
using remove_const_t = T;
template <typename T>
using remove_volatile_t = T;
template <typename T>
using remove_cv_t = T;
}
namespace ext
{
template <typename T>
using remove_const_t = T;
template <typename T>
using remove_volatile_t = T;
}
}
}
}

// CHECK-MESSAGES-NOT: warning:
template <typename T>
using implicit_const = cuda::std::remove_const_t<cuda::std::remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
using implicit_volatile = cuda::std::remove_volatile_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
using explicit_const = cuda::abi::std::version::remove_const_t<cuda::abi::std::version::remove_volatile_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
using explicit_volatile = cuda::abi::std::version::remove_volatile_t<cuda::abi::std::version::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

// Ordinary nested namespaces must not match the standard trait namespace.
template <typename T>
using extension_const = cuda::std::ext::remove_const_t<cuda::std::remove_volatile_t<T>>;
template <typename T>
using extension_volatile = cuda::std::remove_volatile_t<cuda::std::ext::remove_const_t<T>>;
// clang-format on
