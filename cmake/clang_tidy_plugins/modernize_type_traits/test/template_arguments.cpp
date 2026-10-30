//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/modernize/type-traits.cpp

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>

// CHECK-MESSAGES-NOT: warning:
template <typename T, typename U>
constexpr bool same = cuda::std::is_same<T, U>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_v{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

template <bool Condition>
typename cuda::std::enable_if<Condition>::type dependent_return();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 9
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_t{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

typename cuda::std::enable_if<true>::type concrete_return();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 9
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_t{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

cuda::std::enable_if<true>::type return_without_typename();
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_t{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

using variadic = cuda::std::common_type<int, long, bool>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_t{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

template bool const same<int, int>;
// clang-format on
