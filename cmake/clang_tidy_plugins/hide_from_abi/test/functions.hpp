// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-hide-from-abi
// CheckOptions: cccl-hide-from-abi.IgnoreTestFiles=false
// CLANG_TIDY_CONFIG_END

#include <cuda/std/detail/__config>

#undef _CCCL_HOST
#undef _CCCL_DEVICE
#undef _CCCL_HOST_DEVICE
// Attribute expansions keep the annotation in the declaration's source range.
#define _CCCL_HOST __attribute__((annotate("host")))
#define _CCCL_DEVICE __attribute__((annotate("device")))
#define _CCCL_HOST_DEVICE _CCCL_HOST _CCCL_DEVICE

// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST inline void host_function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_DEVICE constexpr int device_function() { return 1; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST_DEVICE inline void host_device_function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
inline void unannotated() {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function needs an API macro (one of _CCCL_API, _CCCL_DEVICE_API, or _CCCL_HOST_API) to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST void declaration_only();
_CCCL_DEVICE void deleted_function() = delete;
_CCCL_DEVICE inline int variable = 0;
inline auto lambda = []() _CCCL_HOST { return 1; };
inline int lambda_result = lambda();

// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_DEVICE_API
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Replacements: []

template <class T>
_CCCL_HOST_DEVICE T identity(T value) { return value; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
inline int instantiated = identity(1);
inline double also_instantiated = identity(1.0);
template long identity(long);
template <> _CCCL_HOST inline char identity(char value) { return value; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:

// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API

_CCCL_HOST_API inline void host_api() {}
_CCCL_DEVICE_API inline void device_api() {}
_CCCL_API inline void host_device_api() {}
_CCCL_PUBLIC_API inline void public_api() {}
_CCCL_TRIVIAL_API void trivial_api() {}
_CCCL_NODEBUG_API void nodebug_api() {}
_CCCL_HOST __attribute__((visibility("hidden"))) inline void hidden_attribute() {}
_CCCL_HOST __attribute__((visibility("default"))) inline void exported_attribute() {}
__attribute__((visibility("hidden"))) void inherited();
_CCCL_HOST inline void inherited() {}

// Exclusion from explicit instantiation alone does not establish visibility.
_CCCL_HOST __attribute__((exclude_from_explicit_instantiation)) inline void excluded() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:

// NOLINTNEXTLINE(cccl-hide-from-abi)
_CCCL_HOST inline void suppressed() {}

// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API

_CCCL_HOST [[nodiscard]] constexpr int attributed() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API

_CCCL_HOST /* Keep this comment and inline. */ inline int explicit_inline() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API

_CCCL_DEVICE _CCCL_FORCEINLINE void forceinline_function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_DEVICE_API

inline _CCCL_HOST_DEVICE void middle_annotation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API

#define DEFINE_FUNCTION(name) inline void name() {}
DEFINE_FUNCTION(macro_function)
// CHECK-MESSAGES: :[[@LINE-2]]:43: warning: function needs an API macro (one of _CCCL_API, _CCCL_DEVICE_API, or _CCCL_HOST_API) to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Replacements: []

// clang-format on
