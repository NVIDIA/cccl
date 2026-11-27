// SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * Static configuration header for the CUB project.
 *
 * This header is deprecated and should no longer be included. Include the headers that provide the functionality you
 * use instead. The deprecation warning can be disabled by defining `CCCL_DISABLE_CUB_CONFIG_HEADER_WARNING`.
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config> // IWYU pragma: export

// Note: the warning must be issued before this header is marked as a system header
#ifndef CCCL_DISABLE_CUB_CONFIG_HEADER_WARNING
#  if _CCCL_COMPILER(MSVC)
#    pragma message( \
      "warning: <cub/config.cuh> is deprecated and should no longer be included. Include the headers that provide the functionality you use instead (e.g., <cub/util_namespace.cuh>, <cub/util_macro.cuh>, or <cub/util_arch.cuh>). Define CCCL_DISABLE_CUB_CONFIG_HEADER_WARNING to disable this warning.")
#  else
#    warning \
      "<cub/config.cuh> is deprecated and should no longer be included. Include the headers that provide the functionality you use instead (e.g., <cub/util_namespace.cuh>, <cub/util_macro.cuh>, or <cub/util_arch.cuh>). Define CCCL_DISABLE_CUB_CONFIG_HEADER_WARNING to disable this warning."
#  endif // _CCCL_COMPILER(MSVC)
#endif // CCCL_DISABLE_CUB_CONFIG_HEADER_WARNING

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_arch.cuh> // IWYU pragma: export
#include <cub/util_cpp_dialect.cuh> // IWYU pragma: export
#include <cub/util_macro.cuh> // IWYU pragma: export
#include <cub/util_namespace.cuh> // IWYU pragma: export
