// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-hide-from-abi
// CheckOptions: cccl-hide-from-abi.IgnoreTestFiles=false
// CLANG_TIDY_CONFIG_END

#define _CCCL_HOST __attribute__((annotate("host")))
#define _CCCL_DEVICE __attribute__((annotate("device")))
#define _CCCL_HOST_DEVICE _CCCL_HOST _CCCL_DEVICE

// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST_DEVICE inline void redeclared_function();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST_DEVICE inline void redeclared_function();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST_DEVICE inline void redeclared_function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:

// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API

template <class T>
_CCCL_HOST_DEVICE T redeclared(T value);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
template <class T>
_CCCL_HOST_DEVICE T redeclared(T value);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
template <class T>
_CCCL_HOST_DEVICE T redeclared(T value) { return value; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:

// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// clang-format on
