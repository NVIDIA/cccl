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
using concrete = cuda::std::enable_if <true>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_t{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

template <bool Condition>
using dependent = typename cuda::std::enable_if <Condition>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 9
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_t{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

[[maybe_unused]] constexpr bool concrete_value = cuda::std::is_const <bool> ::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_v{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
constexpr bool dependent_value = cuda::std::is_const  <T> ::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_v{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

// clang-format on
