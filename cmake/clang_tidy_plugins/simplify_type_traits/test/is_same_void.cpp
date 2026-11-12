// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>
#include <type_traits>

// CHECK-MESSAGES-NOT: warning:
template <typename T>
inline constexpr bool void_second = cuda::std::is_same_v<T, void>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
inline constexpr bool void_first = cuda::std::is_same_v<void, T>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

inline constexpr bool concrete_second = cuda::std::is_same_v<int, void>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

inline constexpr bool concrete_first = cuda::std::is_same_v<void, int>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

// The fix must remove the written void, not an alias with the same canonical type.
using void_alias = void;
inline constexpr bool alias_first = cuda::std::is_same_v<void_alias, void>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

inline constexpr bool const_void = cuda::std::is_same_v<int, const void>;

inline constexpr bool volatile_void = cuda::std::is_same_v<volatile void, int>;

inline constexpr bool cv_void = cuda::std::is_same_v<const volatile void, const volatile void>;

// Qualified void arguments remain unchanged.
inline constexpr bool qualified_next = cuda::std::is_same_v<void volatile, const int>;

inline constexpr bool qualified_previous = cuda::std::is_same_v<int volatile, void const>;

// Commas inside retained template arguments are not removal boundaries.
inline constexpr bool nested_next = cuda::std::is_same_v<void, cuda::std::is_same<int, int>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

inline constexpr bool nested_previous = cuda::std::is_same_v<cuda::std::is_same<int, int>, void>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::is_void_v{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''
// CHECK-FIXES-NOT: DiagnosticName:

inline constexpr bool aliased_void = cuda::std::is_same_v<void_alias, void_alias>;

inline constexpr bool non_void = cuda::std::is_same_v<int, int>;
inline constexpr bool void_pointer = cuda::std::is_same_v<void*, void*>;
inline constexpr bool host_first = std::is_same_v<void, int>;
inline constexpr bool host_second = std::is_same_v<int, void>;

// Instantiations must not rewrite a source argument that names a template parameter.
template <typename T, typename U>
inline constexpr bool generic = cuda::std::is_same_v<T, U>;

static_assert(generic<void, void>);
static_assert(!generic<void, int>);
static_assert(!generic<int, void>);
static_assert(void_first<void>);
static_assert(void_second<void>);
static_assert(!void_first<int>);
static_assert(!void_second<int>);
static_assert(!concrete_first);
static_assert(!concrete_second);
// clang-format on
