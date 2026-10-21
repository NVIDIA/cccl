// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/type_traits>

// CHECK-MESSAGES-NOT: warning:
[[maybe_unused]] constexpr bool floating_point = cuda::is_floating_point<float>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

template <typename T>
constexpr bool dependent = cuda::is_floating_point<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

namespace traits = cuda;
[[maybe_unused]] constexpr bool namespace_alias = traits::is_floating_point<float>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

using cuda::is_floating_point;
template <typename T>
constexpr bool imported = is_floating_point<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

// Aliases of cataloged traits do not themselves belong to the catalog.
template <typename T>
using local_alias = cuda::std::is_integral<T>;
[[maybe_unused]] constexpr bool unlisted_alias = local_alias<int>::value;
template <typename T>
constexpr bool unlisted_dependent_alias = local_alias<T>::value;
template <typename T>
using local_type_alias = cuda::std::remove_const<T>;
using unlisted_type_alias = local_type_alias<const int>::type;
template <typename T>
using unlisted_dependent_type_alias = typename local_type_alias<T>::type;

static_assert(dependent<float>);
static_assert(imported<float>);
static_assert(unlisted_dependent_alias<int>);
// clang-format on
