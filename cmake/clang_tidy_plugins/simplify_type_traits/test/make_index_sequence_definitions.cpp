// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cstddef>

// CHECK-MESSAGES-NOT: warning:
namespace cuda::std
{
inline namespace abi
{
using ::size_t;
template <size_t N>
struct sequence {};
template <size_t N>
using make_index_sequence = sequence<N>;

// Replacing this specialization would make index_sequence_for refer to itself.
template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
}
}

// A matching name in another namespace must not suppress the diagnostic.
namespace unrelated
{
template <typename... Ts>
using index_sequence_for = cuda::std::make_index_sequence<sizeof...(Ts)>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 30
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence_for{{'?}}{{$}}
// CHECK-FIXES: Length: 13
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}Ts...{{'?}}{{$}}
}
// clang-format on
