// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>

// These declarations isolate the rewrite from CUDA runtime dependencies.
// CHECK-MESSAGES-NOT: warning:
namespace cuda
{
inline namespace abi
{
namespace mr
{
struct device_accessible {};
struct host_accessible {};
}
template <typename T, typename... Properties>
struct buffer {};

// Replacing these specializations would make the aliases refer to themselves.
template <typename T>
using device_buffer = buffer<T, mr::device_accessible>;
template <typename T>
using host_buffer = buffer<T, mr::host_accessible>;
}
}

using device = cuda::buffer<int, cuda::mr::device_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::device_buffer{{'?}}{{$}}
// CHECK-FIXES: Length: 29
// CHECK-FIXES-NEXT: ReplacementText: ''

using host = cuda::buffer<int, cuda::mr::host_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::host_buffer{{'?}}{{$}}
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: ''

namespace gpu = cuda;
template <typename T>
using dependent_device = gpu::buffer<T, gpu::mr::device_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::device_buffer{{'?}}{{$}}
// CHECK-FIXES: Length: 28
// CHECK-FIXES-NEXT: ReplacementText: ''

using cuda::buffer;
using cuda::mr::host_accessible;
template <typename T>
using dependent_host = buffer<T, host_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::host_buffer{{'?}}{{$}}
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: ''

// A matching name in another namespace must not suppress the diagnostic.
namespace unrelated
{
template <typename T>
using device_buffer = cuda::buffer<T, cuda::mr::device_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::device_buffer{{'?}}{{$}}
// CHECK-FIXES: Length: 29
// CHECK-FIXES-NEXT: ReplacementText: ''

template <typename T>
using host_buffer = cuda::buffer<T, cuda::mr::host_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::host_buffer{{'?}}{{$}}
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: ''
}

// Removing a property must not discard additional properties or change their order.
using both = cuda::buffer<int, cuda::mr::device_accessible, cuda::mr::host_accessible>;
using reversed = cuda::buffer<int, cuda::mr::host_accessible, cuda::mr::device_accessible>;
using no_properties = cuda::buffer<int>;
struct custom_property {};
using extra_property = cuda::buffer<int, cuda::mr::device_accessible, custom_property>;
using other_property = cuda::buffer<int, custom_property>;
template <typename Property>
using dependent_property = cuda::buffer<int, Property>;
template <typename... Properties>
using property_pack = cuda::buffer<int, Properties...>;
using already_device = cuda::device_buffer<int>;
using already_host = cuda::host_buffer<int>;

namespace unrelated
{
struct device_accessible {};
struct host_accessible {};
using other_device = cuda::buffer<int, device_accessible>;
using other_host = cuda::buffer<int, host_accessible>;
template <typename T, typename... Properties>
struct buffer {};
using other_template = buffer<int, cuda::mr::device_accessible>;
}

static_assert(cuda::std::is_same_v<device, cuda::device_buffer<int>>);
static_assert(cuda::std::is_same_v<host, cuda::host_buffer<int>>);
static_assert(cuda::std::is_same_v<dependent_device<int>, cuda::device_buffer<int>>);
static_assert(cuda::std::is_same_v<dependent_host<int>, cuda::host_buffer<int>>);
static_assert(!cuda::std::is_same_v<both, cuda::device_buffer<int>>);
// clang-format on
