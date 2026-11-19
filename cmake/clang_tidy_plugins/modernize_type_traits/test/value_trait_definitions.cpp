// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: warning:
namespace cuda::std
{
inline namespace abi
{
template <typename T>
struct tuple_size { static constexpr unsigned long value = 1; };

template <typename T>
inline constexpr unsigned long tuple_size_v = tuple_size<T>::value;

template <typename T>
inline constexpr unsigned long tuple_size_v<T*> = tuple_size<T*>::value;

template <>
inline constexpr unsigned long tuple_size_v<void> = tuple_size<void>::value;

// A nondependent initializer still belongs to an ordinary variable template definition.
template <typename T>
struct is_integral { static constexpr bool value = true; };
template <typename T>
inline constexpr bool is_integral_v = is_integral<int>::value;
}
}

// Instantiation must not reintroduce diagnostics at the template definition.
static_assert(cuda::std::tuple_size_v<int> == 1);
static_assert(cuda::std::tuple_size_v<int*> == 1);
static_assert(cuda::std::tuple_size_v<void> == 1);
static_assert(cuda::std::is_integral_v<int>);

// The guard must retain diagnostics for consumers, including matching unqualified names.
namespace unrelated
{
template <typename T>
inline constexpr unsigned long tuple_size_v = cuda::std::tuple_size<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}

inline constexpr unsigned long concrete = cuda::std::tuple_size<int>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}_v{{'?}}{{$}}
}
// clang-format on
