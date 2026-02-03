// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file simple_defines.h
 *  \brief Primitive macros without dependencies.
 */

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! Deprecated [Since 3.0]
#define THRUST_UNKNOWN 0
//! Deprecated [Since 3.0]
#define THRUST_FALSE 0
//! Deprecated [Since 3.0]
#define THRUST_TRUE 1

//! Deprecated [Since 3.0]
#define THRUST_UNUSED_VAR(expr) \
  do                            \
  {                             \
    (void) (expr);              \
  } while (0)

//! Deprecated [Since 3.0]
#define THRUST_PREVENT_MACRO_SUBSTITUTION
