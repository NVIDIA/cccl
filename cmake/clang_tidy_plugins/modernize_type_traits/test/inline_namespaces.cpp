// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional cases adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/modernize/type-traits.cpp

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

// Both namespace levels can contain inline ABI namespaces.
namespace cuda
{
inline namespace abi
{
namespace std
{
inline namespace version
{
template <typename T>
struct remove_const { using type = T; };
template <typename T>
using remove_const_t = T;
template <typename T>
struct is_integral { static constexpr bool value = true; };
template <typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;
}
namespace ext
{
template <typename T>
struct remove_const { using type = T; };
}
}
}
}

// CHECK-MESSAGES-NOT: warning:
using concrete = cuda::std::remove_const<int>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

template <typename T>
using dependent = typename cuda::std::remove_const<T>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

constexpr bool value = cuda::std::is_integral<int>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}
static_assert(value);
template <typename T>
constexpr bool dependent_value = cuda::std::is_integral<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}
using explicit_inline = cuda::abi::std::version::remove_const<int>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++14 style type templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_t{{'?}}{{$}}

// An ordinary nested namespace must not match the standard trait namespace.
using extension = cuda::std::ext::remove_const<int>::type;

// A matching variable name in another namespace must still produce a warning.
namespace unrelated
{
template <typename T>
inline constexpr bool is_integral_v = cuda::std::is_integral<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}
}
// clang-format on
