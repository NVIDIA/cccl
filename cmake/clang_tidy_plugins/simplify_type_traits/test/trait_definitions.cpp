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
template <typename T> using remove_const_t = T;
template <typename T> using remove_volatile_t = T;
template <typename T> using remove_reference_t = T;
template <typename T> using decay_t = T;
template <typename T> using unwrap_reference_t = T;

// Both the class member and the public alias implement the trait being suggested.
template <typename T>
struct remove_cv
{
  using type = remove_volatile_t<remove_const_t<T>>;
};
template <typename T>
using remove_cv_t = remove_volatile_t<remove_const_t<T>>;

template <typename T>
struct remove_cvref
{
  using type = remove_cv_t<remove_reference_t<T>>;
};
template <typename T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

template <typename T>
struct unwrap_ref_decay
{
  using type = unwrap_reference_t<decay_t<T>>;
};
template <typename T>
using unwrap_ref_decay_t = unwrap_reference_t<decay_t<T>>;

// Class specializations must receive the same protection as the primary definition.
template <typename T>
struct remove_cv<T*>
{
  using type = remove_volatile_t<remove_const_t<T*>>;
};
template <>
struct remove_cv<void>
{
  using type = remove_volatile_t<remove_const_t<void>>;
};
}
}

static_assert(sizeof(cuda::std::remove_cv<int>) > 0);
static_assert(sizeof(cuda::std::remove_cv<int*>) > 0);
static_assert(sizeof(cuda::std::remove_cv<void>) > 0);
static_assert(sizeof(cuda::std::remove_cvref<int>) > 0);
static_assert(sizeof(cuda::std::unwrap_ref_decay<int>) > 0);
static_assert(sizeof(cuda::std::remove_cv_t<int>) == sizeof(int));
static_assert(sizeof(cuda::std::remove_cvref_t<int>) == sizeof(int));
static_assert(sizeof(cuda::std::unwrap_ref_decay_t<int>) == sizeof(int));

// Matching names in other namespaces must not suppress valid simplifications.
namespace unrelated
{
template <typename T>
using remove_cv_t = cuda::std::remove_volatile_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
struct remove_cv
{
  using type = cuda::std::remove_volatile_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
};
}

// CHECK-MESSAGES-NOT: warning:
namespace cuda::std
{
inline namespace abi
{
template <typename T, T V>
struct integral_constant {};

// Rewriting these definitions would make the aliases refer to themselves.
template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;
}
}

// The definition guard must compare the full name, including the namespace.
namespace unrelated
{
template <bool B>
using bool_constant = cuda::std::integral_constant<bool, B>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::bool_constant{{'?}}{{$}}
// CHECK-FIXES: Length: 6
// CHECK-FIXES-NEXT: ReplacementText: ''

using true_type = cuda::std::bool_constant<true>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 30
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::true_type{{'?}}{{$}}

using false_type = cuda::std::bool_constant<false>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 31
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::false_type{{'?}}{{$}}
}

// CHECK-MESSAGES-NOT: warning:
namespace cuda::std
{
inline namespace abi
{
using ::size_t;
template <typename T, T... Is>
struct integer_sequence {};

// Replacing this specialization would make index_sequence refer to itself.
template <size_t... Is>
using index_sequence = integer_sequence<size_t, Is...>;
}
}

// A matching name in another namespace must not suppress the diagnostic.
namespace unrelated
{
template <size_t... Is>
using index_sequence = cuda::std::integer_sequence<size_t, Is...>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 27
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''
}

// CHECK-MESSAGES-NOT: warning:
namespace cuda::std
{
inline namespace abi
{
using ::size_t;
template <typename T, T N>
struct sequence {};
template <typename T, T N>
using make_integer_sequence = sequence<T, N>;

// Replacing this specialization would make make_index_sequence refer to itself.
template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;
}
}

// A matching name in another namespace must not suppress the diagnostic.
namespace unrelated
{
template <size_t N>
using make_index_sequence = cuda::std::make_integer_sequence<size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 32
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::make_index_sequence{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''
}

// CHECK-MESSAGES-NOT: warning:
namespace cuda::std
{
inline namespace abi
{
using ::size_t;
template <typename T, size_t N>
struct extents {};
template <typename T, size_t N>
using dextents = extents<T, N>;

// Replacing this specialization would make dims refer to itself.
template <size_t N>
using dims = dextents<size_t, N>;
}
}

// A matching name in another namespace must not suppress the diagnostic.
namespace unrelated
{
template <size_t N>
using dims = cuda::std::dextents<size_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 19
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::dims{{'?}}{{$}}
// CHECK-FIXES: Length: 8
// CHECK-FIXES-NEXT: ReplacementText: ''
}

// CHECK-MESSAGES-NOT: warning:
namespace cuda
{
inline namespace abi
{
template <typename T, size_t N>
struct strides {};
template <typename T, size_t N>
using dstrides = strides<T, N>;

// Replacing this specialization would make steps refer to itself.
template <size_t N>
using steps = dstrides<ptrdiff_t, N>;
}
}

// A matching name in another namespace must not suppress the diagnostic.
namespace unrelated
{
template <size_t N>
using steps = cuda::dstrides<ptrdiff_t, N>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: Length: 14
// CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::steps{{'?}}{{$}}
// CHECK-FIXES: Length: 11
// CHECK-FIXES-NEXT: ReplacementText: ''
}
// clang-format on
