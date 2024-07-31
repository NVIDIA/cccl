//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// iter_common_reference_t

#include <cuda/std/concepts>
#include <cuda/std/iterator>

struct X
{};

// value_type and dereferencing are the same
struct T1
{
  using value_type = X;
  __host__ __device__ X operator*() const;
};
static_assert(cuda::std::same_as<cuda::std::iter_common_reference_t<T1>, X>);

// value_type and dereferencing are the same (modulo qualifiers)
struct T2
{
  using value_type = X;
  __host__ __device__ X& operator*() const;
};
static_assert(cuda::std::same_as<cuda::std::iter_common_reference_t<T2>, X&>);

// There's a custom common reference between value_type and the type of dereferencing
struct A
{};
struct B
{};
struct Common
{
  __host__ __device__ Common(A);
  __host__ __device__ Common(B);
};
template <template <class> class TQual, template <class> class QQual>
struct cuda::std::basic_common_reference<A, B, TQual, QQual>
{
  using type = Common;
};
template <template <class> class TQual, template <class> class QQual>
struct cuda::std::basic_common_reference<B, A, TQual, QQual> : cuda::std::basic_common_reference<A, B, TQual, QQual>
{};

struct T3
{
  using value_type = A;
  __host__ __device__ B&& operator*() const;
};
static_assert(cuda::std::same_as<cuda::std::iter_common_reference_t<T3>, Common>);

// Make sure we're SFINAE-friendly
#if TEST_STD_VER > 2017
template <class T>
constexpr bool has_common_reference = requires { typename cuda::std::iter_common_reference_t<T>; };
#else
template <class T>
_LIBCUDACXX_CONCEPT_FRAGMENT(has_common_reference_, requires()(typename(cuda::std::iter_common_reference_t<T>)));

template <class T>
_LIBCUDACXX_CONCEPT has_common_reference = _LIBCUDACXX_FRAGMENT(has_common_reference_, T);
#endif
struct NotIndirectlyReadable
{};
static_assert(!has_common_reference<NotIndirectlyReadable>);

int main(int, char**)
{
  return 0;
}
