// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file thrust/system_error.h
 *  \brief System diagnostics
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN

/*! \addtogroup system
 *  \{
 */

/*! \namespace thrust::system
 *  \brief \p thrust::system is the namespace which contains specific Thrust
 *         backend systems. It also contains functionality for reporting error
 *         conditions originating from the operating system or other low-level
 *         application program interfaces such as the CUDA runtime. They are
 *         provided in a separate namespace for import convenience but are
 *         also aliased in the top-level \p thrust namespace for easy access.
 */
namespace system
{
} // namespace system

/*! \} // end system
 */

THRUST_NAMESPACE_END

#include <thrust/system/error_code.h>
#include <thrust/system/system_error.h>
