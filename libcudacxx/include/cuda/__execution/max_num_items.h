//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_MAX_NUM_ITEMS_H
#define __CUDA___EXECUTION_MAX_NUM_ITEMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__execution/guarantee.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

struct __get_max_num_items_t
{};

//! @brief Promises that the number of input items is at most @p _MaxItems.
//! @details Pass this guarantee through @c cuda::execution::guarantee. The caller must ensure that the input
//! does not exceed the bound. Algorithms may use it to select a narrower offset type at compile time.
template <::cuda::std::size_t _MaxItems>
struct max_num_items : __guarantee
{
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(__get_max_num_items_t) noexcept
  {
    return ::cuda::std::integral_constant<::cuda::std::size_t, _MaxItems>{};
  }
};

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_MAX_NUM_ITEMS_H
