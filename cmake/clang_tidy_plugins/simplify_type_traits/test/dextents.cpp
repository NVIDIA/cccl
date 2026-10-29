// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cstddef>
#include <cuda/mdspan>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

namespace dextents_tests
{
// CHECK-MESSAGES-NOT: warning:
using scalar = cuda::std::dextents<size_t, 0>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

using matrix = cuda::std::dextents<::size_t, 2>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: ''

using host_size = cuda::std::dextents<std::size_t, 3>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 13
// CHECK-FIXES-NEXT: ReplacementText: ''

template <size_t N>
using dependent_rank = cuda::std::dextents<cuda::std::size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace traits = cuda::std;
template <size_t N>
using namespace_alias = traits::dextents<traits::size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 16
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 16
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::std::dextents;
template <size_t N>
using imported = dextents<size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''

template <size_t N>
using spaced = cuda::std::dextents < size_t ,  N + 1 >;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 20
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: ''

// Matching size_t must not match its underlying type or separately named aliases.
using unsigned_index = cuda::std::dextents<unsigned long, 2>;
using signed_index = cuda::std::dextents<int, 2>;
using index_type = size_t;
using aliased_index = cuda::std::dextents<index_type, 2>;
typedef size_t index_typedef;
using typedef_index = cuda::std::dextents<index_typedef, 2>;
template <typename T>
using dependent_index = cuda::std::dextents<T, 2>;
using already_simplified = cuda::std::dims<2>;

namespace unrelated
{
using size_t = unsigned int;
using different_size = cuda::std::dextents<size_t, 2>;
template <typename T, ::size_t N>
struct dextents {};
using different_template = dextents<::size_t, 2>;
}

static_assert(cuda::std::is_same_v<scalar, cuda::std::dims<0>>);
static_assert(cuda::std::is_same_v<matrix, cuda::std::dims<2>>);
static_assert(cuda::std::is_same_v<dependent_rank<3>, cuda::std::dims<3>>);
static_assert(cuda::std::is_same_v<spaced<2>, cuda::std::dims<3>>);
}

namespace dstrides_tests
{
// CHECK-MESSAGES-NOT: warning:
using empty = cuda::dstrides<ptrdiff_t, 0>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 14
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 11
// CHECK-FIXES-NEXT: ReplacementText: ''

using global_offset = cuda::dstrides<::ptrdiff_t, 2>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 13
// CHECK-FIXES-NEXT: ReplacementText: ''

using host_offset = cuda::dstrides<std::ptrdiff_t, 3>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 16
// CHECK-FIXES-NEXT: ReplacementText: ''

template <cuda::std::size_t N>
using dependent_rank = cuda::dstrides<cuda::std::ptrdiff_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 22
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace gpu = cuda;
template <cuda::std::size_t N>
using namespace_alias = gpu::dstrides<gpu::std::ptrdiff_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 21
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::dstrides;
using imported = dstrides<ptrdiff_t, 1>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 11
// CHECK-FIXES-NEXT: ReplacementText: ''

// Matching ptrdiff_t must not match its underlying type or separately named aliases.
using signed_offset = cuda::dstrides<long, 2>;
using unsigned_offset = cuda::dstrides<cuda::std::size_t, 2>;
using offset_type = ptrdiff_t;
using aliased_offset = cuda::dstrides<offset_type, 2>;
typedef ptrdiff_t offset_typedef;
using typedef_offset = cuda::dstrides<offset_typedef, 2>;
template <typename T>
using dependent_offset = cuda::dstrides<T, 2>;
using already_simplified = cuda::steps<2>;

namespace unrelated
{
using ptrdiff_t = unsigned int;
using different_offset = cuda::dstrides<ptrdiff_t, 2>;
template <typename T, ::size_t N>
struct dstrides {};
using different_template = dstrides<::ptrdiff_t, 2>;
}

static_assert(cuda::std::is_same_v<empty, cuda::steps<0>>);
static_assert(cuda::std::is_same_v<global_offset, cuda::steps<2>>);
static_assert(cuda::std::is_same_v<host_offset, cuda::steps<3>>);
static_assert(cuda::std::is_same_v<dependent_rank<3>, cuda::steps<3>>);
static_assert(cuda::std::is_same_v<namespace_alias<2>, cuda::steps<2>>);
static_assert(cuda::std::is_same_v<imported, cuda::steps<1>>);
}
// clang-format on
