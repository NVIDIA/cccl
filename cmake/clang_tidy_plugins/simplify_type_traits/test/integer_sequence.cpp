// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cstddef>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <utility>

namespace integer_sequence_tests
{
// CHECK-MESSAGES-NOT: warning:
using singleton = cuda::std::integer_sequence<size_t, 0>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

using values = cuda::std::integer_sequence<::size_t, 3, 1, 3>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: ''

using host_size = cuda::std::integer_sequence<std::size_t, 2>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 13
// CHECK-FIXES-NEXT: ReplacementText: ''

template <size_t... Is>
using dependent_values = cuda::std::integer_sequence<cuda::std::size_t, Is...>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace traits = cuda::std;
template <size_t... Is>
using namespace_alias = traits::integer_sequence<traits::size_t, Is...>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 24
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 16
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::std::integer_sequence;
template <size_t... Is>
using imported = integer_sequence<size_t, Is...>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 16
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

template <size_t N, size_t... Is>
using spaced = cuda::std::integer_sequence < size_t ,  N + 1, Is... >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 28
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: ''

// Matching size_t must not match its underlying type or separately named aliases.
using unsigned_index = cuda::std::integer_sequence<unsigned long, 0, 1>;
using signed_index = cuda::std::integer_sequence<int, -1, 0>;
using index_type = size_t;
using aliased_index = cuda::std::integer_sequence<index_type, 0>;
typedef size_t index_typedef;
using typedef_index = cuda::std::integer_sequence<index_typedef, 0>;
template <typename T, T... Is>
using dependent_index = cuda::std::integer_sequence<T, Is...>;
using already_simplified = cuda::std::index_sequence<0, 1>;
using host_sequence = std::integer_sequence<size_t, 0, 1>;

namespace unrelated
{
using size_t = unsigned int;
using different_size = cuda::std::integer_sequence<size_t, 0>;
template <typename T, T... Is>
struct integer_sequence {};
using different_template = integer_sequence<::size_t, 0>;
}

static_assert(cuda::std::is_same_v<singleton, cuda::std::index_sequence<0>>);
static_assert(cuda::std::is_same_v<values, cuda::std::index_sequence<3, 1, 3>>);
static_assert(cuda::std::is_same_v<dependent_values<>, cuda::std::index_sequence<>>);
static_assert(cuda::std::is_same_v<dependent_values<2, 0>, cuda::std::index_sequence<2, 0>>);
static_assert(cuda::std::is_same_v<spaced<2, 0, 1>, cuda::std::index_sequence<3, 0, 1>>);
}

namespace integer_sequence_empty_tests
{
// An empty sequence has no comma or second argument to remove.
// CHECK-MESSAGES-NOT: warning:
using empty = cuda::std::integer_sequence<cuda::std::size_t>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: ''

static_assert(cuda::std::is_same_v<empty, cuda::std::index_sequence<>>);
}

namespace make_integer_sequence_tests
{
// CHECK-MESSAGES-NOT: warning:
using empty = cuda::std::make_integer_sequence<size_t, 0>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 32
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

using singleton = cuda::std::make_integer_sequence<::size_t, 1>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 32
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: ''

using host_size = cuda::std::make_integer_sequence<std::size_t, 3>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 13
// CHECK-FIXES-NEXT: ReplacementText: ''

template <size_t N>
using dependent_count = cuda::std::make_integer_sequence<cuda::std::size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace traits = cuda::std;
template <size_t N>
using namespace_alias = traits::make_integer_sequence<traits::size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 29
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 16
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::std::make_integer_sequence;
template <size_t N>
using imported = make_integer_sequence<size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 21
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

template <size_t N>
using spaced = cuda::std::make_integer_sequence < size_t ,  N + 1 >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 33
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: ''

// Matching size_t must not match its underlying type or separately named aliases.
using unsigned_index = cuda::std::make_integer_sequence<unsigned long, 2>;
using signed_index = cuda::std::make_integer_sequence<int, 2>;
using index_type = size_t;
using aliased_index = cuda::std::make_integer_sequence<index_type, 2>;
typedef size_t index_typedef;
using typedef_index = cuda::std::make_integer_sequence<index_typedef, 2>;
template <typename T, T N>
using dependent_index = cuda::std::make_integer_sequence<T, N>;
using already_simplified = cuda::std::make_index_sequence<2>;
using host_sequence = std::make_integer_sequence<size_t, 2>;

namespace unrelated
{
using size_t = unsigned int;
using different_size = cuda::std::make_integer_sequence<size_t, 2>;
template <typename T, T N>
struct make_integer_sequence {};
using different_template = make_integer_sequence<::size_t, 2>;
}

static_assert(cuda::std::is_same_v<empty, cuda::std::index_sequence<>>);
static_assert(cuda::std::is_same_v<singleton, cuda::std::index_sequence<0>>);
static_assert(cuda::std::is_same_v<host_size, cuda::std::index_sequence<0, 1, 2>>);
static_assert(cuda::std::is_same_v<dependent_count<3>, cuda::std::index_sequence<0, 1, 2>>);
static_assert(cuda::std::is_same_v<namespace_alias<2>, cuda::std::index_sequence<0, 1>>);
static_assert(cuda::std::is_same_v<imported<2>, cuda::std::index_sequence<0, 1>>);
static_assert(cuda::std::is_same_v<spaced<2>, cuda::std::index_sequence<0, 1, 2>>);
}

namespace make_index_sequence_tests
{
// CHECK-MESSAGES-NOT: warning:
template <typename... Ts>
using types = cuda::std::make_index_sequence<sizeof...(Ts)>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 30
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence_for{{'?}}{{$}}
// CHECK-FIXES: Length: 13
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}Ts...{{'?}}{{$}}

namespace traits = cuda::std;
template <class... Args>
using namespace_alias = traits::make_index_sequence<sizeof...(Args)>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence_for{{'?}}{{$}}
// CHECK-FIXES: Length: 15
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}Args...{{'?}}{{$}}

using cuda::std::make_index_sequence;
template <typename... Elements>
using imported = make_index_sequence<sizeof...(Elements)>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence_for{{'?}}{{$}}
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}Elements...{{'?}}{{$}}

template <typename... Ts>
using spaced = cuda::std::make_index_sequence < sizeof ... ( Ts ) >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 31
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence_for{{'?}}{{$}}
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}Ts...{{'?}}{{$}}

// index_sequence_for accepts types, not values, templates, or function parameters.
template <int... Is>
using values = cuda::std::make_index_sequence<sizeof...(Is)>;
template <template <typename> class... Templates>
using templates = cuda::std::make_index_sequence<sizeof...(Templates)>;
template <typename... Ts>
auto function_parameters(Ts... args) -> cuda::std::make_index_sequence<sizeof...(args)>;

template <typename... Ts>
using different_count = cuda::std::make_index_sequence<sizeof...(Ts) + 1>;
template <typename T>
using type_size = cuda::std::make_index_sequence<sizeof(T)>;
template <cuda::std::size_t N>
using dependent_count = cuda::std::make_index_sequence<N>;
using constant_count = cuda::std::make_index_sequence<3>;
template <typename... Ts>
using already_simplified = cuda::std::index_sequence_for<Ts...>;
template <typename... Ts>
using host_sequence = std::make_index_sequence<sizeof...(Ts)>;

namespace unrelated
{
template <cuda::std::size_t N>
struct make_index_sequence {};
template <typename... Ts>
using different_template = make_index_sequence<sizeof...(Ts)>;
}

static_assert(cuda::std::is_same_v<types<>, cuda::std::index_sequence<>>);
static_assert(cuda::std::is_same_v<types<int>, cuda::std::index_sequence<0>>);
static_assert(cuda::std::is_same_v<types<int, void, int&>, cuda::std::index_sequence<0, 1, 2>>);
static_assert(cuda::std::is_same_v<namespace_alias<int, void>, cuda::std::index_sequence<0, 1>>);
static_assert(cuda::std::is_same_v<imported<int, void>, cuda::std::index_sequence<0, 1>>);
static_assert(cuda::std::is_same_v<spaced<int, void>, cuda::std::index_sequence<0, 1>>);
}
// clang-format on
