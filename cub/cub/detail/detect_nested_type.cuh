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

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/******************************************************************************
 * Typedef-detection
 ******************************************************************************/

/**
 * \brief Defines a structure \p detector_name that is templated on type \p T.  The \p detector_name struct exposes a
 * constant member \p value indicating whether or not parameter \p T exposes a nested type \p nested_type_name
 */
#  define CUB_DEFINE_DETECT_NESTED_TYPE(detector_name, nested_type_name)                                \
    template <typename T, typename = void>                                                              \
    struct detector_name : ::cuda::std::false_type                                                      \
    {};                                                                                                 \
    template <typename T>                                                                               \
    struct detector_name<T, ::cuda::std::void_t<typename T::nested_type_name>> : ::cuda::std::true_type \
    {};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
