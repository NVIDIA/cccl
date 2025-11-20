// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/*! \file version.cuh
 *  \brief Compile-time macros encoding CUB release version
 *
 *         <cub/version.h> is the only CUB header that is guaranteed to
 *         change with every CUB release.
 *
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/version>

/*! \def CUB_VERSION
 *  \brief The preprocessor macro \p CUB_VERSION encodes the version
 *         number of the CUB library as MMMmmmpp.
 *
 *  \note CUB_VERSION is formatted as `MMMmmmpp`, which differs from `CCCL_VERSION` that uses `MMMmmmppp`.
 *
 *         <tt>CUB_VERSION % 100</tt> is the sub-minor version.
 *         <tt>CUB_VERSION / 100 % 1000</tt> is the minor version.
 *         <tt>CUB_VERSION / 100000</tt> is the major version.
 */
#define CUB_VERSION 300300 // macro expansion with ## requires this to be a single value

/*! \def CUB_MAJOR_VERSION
 *  \brief The preprocessor macro \p CUB_MAJOR_VERSION encodes the
 *         major version number of the CUB library.
 */
#define CUB_MAJOR_VERSION (CUB_VERSION / 100000)

/*! \def CUB_MINOR_VERSION
 *  \brief The preprocessor macro \p CUB_MINOR_VERSION encodes the
 *         minor version number of the CUB library.
 */
#define CUB_MINOR_VERSION (CUB_VERSION / 100 % 1000)

/*! \def CUB_SUBMINOR_VERSION
 *  \brief The preprocessor macro \p CUB_SUBMINOR_VERSION encodes the
 *         sub-minor version number of the CUB library.
 */
#define CUB_SUBMINOR_VERSION (CUB_VERSION % 100)

/*! \def CUB_PATCH_NUMBER
 *  \brief The preprocessor macro \p CUB_PATCH_NUMBER encodes the
 *         patch number of the CUB library.
 */
#define CUB_PATCH_NUMBER 0

static_assert(CUB_MAJOR_VERSION == CCCL_MAJOR_VERSION, "");
static_assert(CUB_MINOR_VERSION == CCCL_MINOR_VERSION, "");
static_assert(CUB_SUBMINOR_VERSION == CCCL_PATCH_VERSION, "");
