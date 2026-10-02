// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/******************************************************************************
 * Typedef-detection
 ******************************************************************************/

/**
 * \brief Determine whether or not BinaryOp's functor is of the form <tt>bool operator()(const T& a, const T&b)</tt> or
 * <tt>bool operator()(const T& a, const T&b, unsigned int idx)</tt>
 */
template <typename T, typename BinaryOp, typename = void>
struct BinaryOpHasIdxParam : ::cuda::std::false_type
{};

template <typename T, typename BinaryOp>
struct BinaryOpHasIdxParam<T,
                           BinaryOp,
                           ::cuda::std::void_t<decltype(::cuda::std::declval<BinaryOp>()(
                             ::cuda::std::declval<T>(), ::cuda::std::declval<T>(), int{}))>> : ::cuda::std::true_type
{};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
