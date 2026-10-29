// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>
#include <type_traits>

namespace bool_constant_tests
{
// CHECK-MESSAGES-NOT: warning:
using literal_true = cuda::std::bool_constant<true>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 30
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::true_type{{'?}}{{$}}

using literal_false = cuda::std::bool_constant<false>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 31
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::false_type{{'?}}{{$}}

namespace traits = cuda::std;
using qualified = traits::bool_constant<true>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::true_type{{'?}}{{$}}

using cuda::std::bool_constant;
using imported = bool_constant<false>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::false_type{{'?}}{{$}}

using spaced = cuda::std::bool_constant < false >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 34
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::false_type{{'?}}{{$}}

// Written literals remain eligible inside variable-template initializers.
template <typename T>
inline constexpr auto literal_variable = cuda::std::bool_constant<true>{};
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::true_type{{'?}}{{$}}
// CHECK-FIXES-NOT: DiagnosticName:

// Only literal arguments qualify, even when an expression has a constant value.
constexpr bool enabled = true;
using named_value = cuda::std::bool_constant<enabled>;
using negated = cuda::std::bool_constant<!true>;
using comparison = cuda::std::bool_constant<(1 == 1)>;
template <bool B>
using dependent = cuda::std::bool_constant<B>;
using host_true = std::bool_constant<true>;
using host_false = std::bool_constant<false>;

template <bool B>
using local_constant = cuda::std::bool_constant<B>;
using local_true = local_constant<true>;

// Substitution creates literal AST nodes, but the source still contains Value.
template <bool Value>
inline constexpr auto bool_constant_v = cuda::std::bool_constant<Value>{};

static_assert(bool_constant_v<true>);
static_assert(!bool_constant_v<false>);
static_assert(literal_variable<int>);

static_assert(cuda::std::is_same_v<literal_true, cuda::std::true_type>);
static_assert(cuda::std::is_same_v<literal_false, cuda::std::false_type>);
}

namespace integral_constant_tests
{
// CHECK-MESSAGES-NOT: warning:
template <bool B>
using dependent = cuda::std::integral_constant<bool, B>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 28
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

using literal_true = cuda::std::integral_constant<bool, true>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

using literal_false = cuda::std::integral_constant<bool, false>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace traits = cuda::std;
template <bool B>
using qualified = traits::integral_constant<bool, B>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 25
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::std::integral_constant;
template <bool B>
using imported = integral_constant<bool, B>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

template <bool B>
using spaced = cuda::std::integral_constant < bool ,  !B >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 29
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

using integer = cuda::std::integral_constant<int, 1>;
template <typename T, T V>
using dependent_type = cuda::std::integral_constant<T, V>;
using host_bool = std::integral_constant<bool, true>;
template <bool B>
using local_constant = cuda::std::integral_constant<int, B>;
using local_true = local_constant<true>;

static_assert(dependent<true>::value);
static_assert(!spaced<true>::value);
}
// clang-format on
