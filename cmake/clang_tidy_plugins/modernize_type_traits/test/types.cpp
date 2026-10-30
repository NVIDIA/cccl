// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>
#include <type_traits>

// CHECK-MESSAGES-NOT: warning:
using concrete = cuda::std::remove_const<const int>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

template <typename T>
using dependent = typename cuda::std::remove_const<T>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

namespace traits = cuda::std;
template <typename T>
using alias = typename traits::remove_reference<T>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

using cuda::std::remove_reference;
template <typename T>
using imported = typename remove_reference<T>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

using host = std::remove_const<const int>::type;
using modern = cuda::std::remove_const_t<const int>;
namespace unrelated
{
struct trait { using type = int; };
using custom = trait::type;
}
static_assert(cuda::std::is_same_v<dependent<const int>, int>);
static_assert(cuda::std::is_same_v<alias<int&>, int>);
static_assert(cuda::std::is_same_v<imported<int&>, int>);
// clang-format on
