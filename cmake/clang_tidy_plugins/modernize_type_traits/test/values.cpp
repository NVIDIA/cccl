// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>
#include <type_traits>

// CHECK-MESSAGES-NOT: warning:
[[maybe_unused]] constexpr bool concrete = cuda::std::is_integral<int>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

template <typename T>
constexpr bool dependent = cuda::std::is_integral<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

namespace traits = cuda::std;
template <typename T>
constexpr bool alias = traits::is_integral<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

using cuda::std::is_integral;
template <typename T>
constexpr bool imported = is_integral<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

[[maybe_unused]] constexpr bool host = std::is_integral<int>::value;
[[maybe_unused]] constexpr bool modern = cuda::std::is_integral_v<int>;

namespace unrelated
{
template <typename T>
struct is_integral { static constexpr bool value = true; };
[[maybe_unused]] constexpr bool custom = is_integral<int>::value;
}
namespace nested
{
namespace cuda
{
namespace std
{
template <typename T>
struct is_integral { static constexpr bool value = true; };
[[maybe_unused]] constexpr bool custom = is_integral<int>::value;
}
}
}

template bool const dependent<int>;
template bool const alias<int>;
template bool const imported<int>;
// clang-format on
