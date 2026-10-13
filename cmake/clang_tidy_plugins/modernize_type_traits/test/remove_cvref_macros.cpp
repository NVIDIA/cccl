//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/modernize/type-traits-remove-cvref.cpp

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>

#define REMOVE_CV(T) cuda::std::remove_cv_t<T>
#define REMOVE_REFERENCE(T) cuda::std::remove_reference_t<T>
#define REMOVE_CVREF(T) cuda::std::remove_cv_t<cuda::std::remove_reference_t<T>>

// CHECK-MESSAGES-NOT: warning:
template <typename = REMOVE_CV(cuda::std::remove_reference_t<int>)>
struct outer_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = cuda::std::remove_cv_t<REMOVE_REFERENCE(int)>>
struct inner_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

template <typename = REMOVE_CVREF(int)>
struct complete_macro {};
// CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Replacements: []

// clang-format on
