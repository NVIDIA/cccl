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
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}

template <typename T>
struct remove_cv
{
  using type = cuda::std::remove_volatile_t<cuda::std::remove_const_t<T>>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-modernize-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-modernize-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::std::remove_cv_t{{'?}}{{$}}
};
}
// clang-format on
