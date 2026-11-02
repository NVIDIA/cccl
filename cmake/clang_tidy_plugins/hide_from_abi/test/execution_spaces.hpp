// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-hide-from-abi
// CLANG_TIDY_CONFIG_END

#include <cuda/std/detail/__config>

#undef _CCCL_HOST
#undef _CCCL_DEVICE
#undef _CCCL_TILE
#undef _CCCL_HOST_DEVICE
#undef __host__
#undef __device__
#undef __tile__
// Attribute expansions keep the annotation in the declaration's source range.
#define __host__ __attribute__((annotate("host")))
#define __device__ __attribute__((annotate("device")))
#define __tile__ __attribute__((annotate("tile")))
#define _CCCL_HOST __host__
#define _CCCL_DEVICE __device__
#define _CCCL_TILE __tile__
#define _CCCL_HOST_DEVICE _CCCL_HOST _CCCL_DEVICE

// CHECK-MESSAGES-NOT: warning:
__host__ __device__ inline void host_device() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 11
// CHECK-FIXES-NEXT: ReplacementText: ''

__device__ __host__ inline void device_host() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 9
// CHECK-FIXES-NEXT: ReplacementText: ''

_CCCL_HOST /* Preserve the comment and specifier. */ inline __device__ void mixed_annotations() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 11
// CHECK-FIXES-NEXT: ReplacementText: ''

__tile__ inline void tile_function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_TILE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_TILE_API

_CCCL_DEVICE _CCCL_TILE inline void device_tile() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_TILE _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_TILE _CCCL_DEVICE_API
// CHECK-FIXES: Length: 11
// CHECK-FIXES-NEXT: ReplacementText: ''

_CCCL_TILE _CCCL_HOST_DEVICE inline void all_spaces() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_API
// CHECK-FIXES: Length: 18
// CHECK-FIXES-NEXT: ReplacementText: ''

__host__ __device__ __tile__ inline void separate_spaces() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_API
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: ''

_CCCL_HOST_DEVICE __host__ inline void redundant_host() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 9
// CHECK-FIXES-NEXT: ReplacementText: ''
// clang-format on
