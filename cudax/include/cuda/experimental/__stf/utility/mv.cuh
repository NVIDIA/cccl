//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 * @brief Widely used artifacts used by most of the library.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>

namespace cuda::experimental::stf
{
#ifndef _CCCL_DOXYGEN_INVOKED // FIXME Doxygen is lost with decltype(auto)
/**
 * @brief Custom move function that performs checks on the argument type.
 *
 * @tparam T Type of the object being moved. The type should satisfy certain conditions for the move to be performed.
 * @param obj The object to be moved.
 * @return The moved object, ready to be passed to another owner.
 *
 * @pre The argument `obj` must be an lvalue, i.e., the function will fail to compile for rvalues.
 * @pre The argument `obj` must not be `const`, i.e., the function will fail to compile for `const` lvalues.
 */
template <typename T>
_CCCL_HOST_DEVICE constexpr decltype(auto) mv(T&& obj)
{
  static_assert(::cuda::std::is_lvalue_reference_v<T>, "Useless move from rvalue.");
  static_assert(!::cuda::std::is_const_v<::cuda::std::remove_reference_t<T>>, "Misleading move from const lvalue.");
  return ::cuda::std::move(obj);
}
#endif // _CCCL_DOXYGEN_INVOKED
} // namespace cuda::experimental::stf
