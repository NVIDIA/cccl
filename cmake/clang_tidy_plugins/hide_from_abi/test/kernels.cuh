// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-hide-from-abi
// CheckOptions: cccl-hide-from-abi.IgnoreTestFiles=false
// CLANG_TIDY_CONFIG_END

#include <cuda/std/detail/__config>

#define __global__ __attribute__((global))

// CHECK-MESSAGES-NOT: warning:
__global__ void kernel() {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use _CCCL_KERNEL_ATTRIBUTES to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_KERNEL_ATTRIBUTES

// CHECK-MESSAGES-NOT: warning:
_CCCL_KERNEL_ATTRIBUTES void annotated_kernel() {}
