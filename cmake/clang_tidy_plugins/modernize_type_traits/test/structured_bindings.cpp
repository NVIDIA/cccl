//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/modernize/type-traits.cpp

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-modernize-type-traits
// CLANG_TIDY_CONFIG_END

// Also covers:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/modernize/type-traits-GH153649.cpp

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <tuple>

// CHECK-MESSAGES-NOT: warning:
[[maybe_unused]] constexpr auto explicit_size = cuda::std::tuple_size<cuda::std::tuple<int>>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use c++17 style variable templates [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}_v{{'?}}{{$}}
// CHECK-FIXES: Length: 7
// CHECK-FIXES-NEXT: ReplacementText: ''

struct aggregate {};

namespace std
{
template <>
struct tuple_size<aggregate> : integral_constant<size_t, 1> {};
template <>
struct tuple_element<0, aggregate> { using type = int; };
}

template <int>
int get(const aggregate&);

auto [element] = aggregate();
auto [cuda_element] = cuda::std::tuple<int>{};

template <typename T>
struct member_type
{
  typename cuda::std::remove_reference<T>::type member; // NOLINT(cccl-modernize-type-traits)
};

#if _CCCL_STD_VER >= 2020
int capture_binding()
{
  auto [member] = member_type<int>{};
  return [&] { return member; }();
}
#endif
// clang-format on
