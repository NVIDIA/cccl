//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_CONVERTIBLE_H
#define SUPPORT_TEST_CONVERTIBLE_H

// "test_convertible<Tp, Args...>()" is a metafunction used to check if 'Tp'
// is implicitly convertible from 'Args...' for any number of arguments,
// Unlike 'std::is_convertible' which only allows checking for single argument
// conversions.

#include <cuda/std/type_traits>

#include "test_macros.h"

namespace detail
{
template <class Tp>
__host__ __device__ void eat_type(Tp);

template <class Tp, class... Args>
__host__ __device__ constexpr auto
test_convertible_imp(int) -> decltype(eat_type<Tp>({cuda::std::declval<Args>()...}), true)
{
  return true;
}

template <class Tp, class... Args>
__host__ __device__ constexpr auto test_convertible_imp(long) -> bool
{
  return false;
}
} // namespace detail

template <class Tp, class... Args>
__host__ __device__ constexpr bool test_convertible()
{
  return detail::test_convertible_imp<Tp, Args...>(0);
}

#endif // SUPPORT_TEST_CONVERTIBLE_H
