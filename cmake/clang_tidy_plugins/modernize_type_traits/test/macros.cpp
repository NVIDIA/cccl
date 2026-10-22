// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional cases adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/modernize/type-traits.cpp

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>

#define INTEGRAL(T) cuda::std::is_integral<T>::value
#define REMOVE_CONST(T) typename cuda::std::remove_const<T>::type

// CHECK-MESSAGES-NOT: warning:
template <typename T>
constexpr bool macro_value = INTEGRAL(T);
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []
template <typename T>
using macro_type = REMOVE_CONST(T);
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []
#define CONCRETE_VALUE cuda::std::is_same<int, int>::value
[[maybe_unused]] constexpr bool concrete_value = CONCRETE_VALUE;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

#define CONCRETE_TYPE typename cuda::std::enable_if<true>::type
using concrete_type = CONCRETE_TYPE;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []
// clang-format on
