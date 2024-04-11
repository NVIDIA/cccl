//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::borrowed_subrange_t;

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/span>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#endif
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  include <cuda/std/string_view>
#endif
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif

#if defined(_LIBCUDACXX_HAS_STRING)
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string>, cuda::std::ranges::dangling>);
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string&&>, cuda::std::ranges::dangling>);
#endif
#if defined(_LIBCUDACXX_HAS_VECTOR)
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::vector<int>>, cuda::std::ranges::dangling>);
#endif

#if defined(_LIBCUDACXX_HAS_STRING)
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string&>,
                                 cuda::std::ranges::subrange<cuda::std::string::iterator>>);
#endif
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::span<int>>,
                                 cuda::std::ranges::subrange<cuda::std::span<int>::iterator>>);

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string_view>,
                                 cuda::std::ranges::subrange<cuda::std::string_view::iterator>>);
#endif

#if TEST_STD_VER > 2017
template <class T>
constexpr bool has_type = requires { typename cuda::std::ranges::borrowed_subrange_t<T>; };
#else
template <class T, class = void>
constexpr bool has_type = false;

template <class T>
constexpr bool has_type<T, cuda::std::void_t<cuda::std::ranges::borrowed_subrange_t<T>>> = false;
#endif

static_assert(!has_type<int>);

struct S
{};
static_assert(!has_type<S>);

int main(int, char**)
{
  return 0;
}
