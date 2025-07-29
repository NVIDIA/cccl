//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NVTX_NVTX_H
#define _CUDA___NVTX_NVTX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:
//! When this macro is defined, no NVTX ranges are emitted by CCCL
#  define CCCL_DISABLE_NVTX
#endif // _CCCL_DOXYGEN_INVOKED

// Enable the functionality of this header if:
// * The NVTX3 C API is available in CTK
// * NVTX is not explicitly disabled (via CCCL_DISABLE_NVTX or NVTX_DISABLE)
// * NVTX3 uses module as an identifier, which trips up NVHPC
#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && !defined(CCCL_DISABLE_NVTX) && !defined(NVTX_DISABLE) \
                      && (!_CCCL_COMPILER(NVHPC))                                                    \
                      && !_CCCL_COMPILER(NVRTC)
// Include our NVTX3 C++ wrapper if not available from the CTK or not provided by the user
// Note: NVTX3 is available in the CTK since 12.9, so we can drop our copy once this is the minimum supported version
#  if _CCCL_HAS_INCLUDE(<nvtx3/nvtx3.hpp>)
#    include <nvtx3/nvtx3.hpp>
#  else // _CCCL_HAS_INCLUDE(<nvtx3/nvtx3.hpp>)
#    include <cuda/__nvtx/nvtx3.h>
#  endif // _CCCL_HAS_INCLUDE(<nvtx3/nvtx3.hpp>)

// We expect the NVTX3 V1 C++ API to be available when nvtx3.hpp is available. This should work, because newer versions
// of NVTX3 will continue to declare previous API versions. See also:
// https://github.com/NVIDIA/NVTX/blob/release-v3/c/include/nvtx3/nvtx3.hpp#L2835-L2841.
#  ifdef NVTX3_CPP_DEFINITIONS_V1_0
#    include <cuda/std/optional>

#    include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

namespace detail
{
struct NVTXCCCLDomain
{
  static constexpr const char* name{"CCCL"};
};
} // namespace detail

// Hook for the NestedNVTXRangeGuard from the unit tests
#    ifndef _CCCL_BEFORE_NVTX_RANGE_SCOPE
#      define _CCCL_BEFORE_NVTX_RANGE_SCOPE(name)
#    endif // !CCCL_DETAIL_BEFORE_NVTX_RANGE_SCOPE

// Conditionally inserts a NVTX range starting here until the end of the current function scope in host code. Does
// nothing in device code.
// The optional is needed to defer the construction of an NVTX range (host-only code) and message string registration
// into a dispatch region running only on the host, while preserving the semantic scope where the range is declared.
#    define _CCCL_NVTX_RANGE_SCOPE_IF(condition, name)                                                               \
      _CCCL_BEFORE_NVTX_RANGE_SCOPE(name)                                                                            \
      _CUDA_VSTD::optional<::nvtx3::v1::scoped_range_in<::cuda::detail::NVTXCCCLDomain>> __cuda_nvtx3_range;         \
      NV_IF_TARGET(                                                                                                  \
        NV_IS_HOST,                                                                                                  \
        static const ::nvtx3::v1::registered_string_in<::cuda::detail::NVTXCCCLDomain> __cuda_nvtx3_func_name{name}; \
        static const ::nvtx3::v1::event_attributes __cuda_nvtx3_func_attr{__cuda_nvtx3_func_name};                   \
        if (condition) __cuda_nvtx3_range.emplace(__cuda_nvtx3_func_attr);                                           \
        (void) __cuda_nvtx3_range;)

#    define _CCCL_NVTX_RANGE_SCOPE(name) _CCCL_NVTX_RANGE_SCOPE_IF(true, name)
#  else // NVTX3_CPP_DEFINITIONS_V1_0
#    if _CCCL_COMPILER(MSVC)
#      pragma message( \
        "warning: nvtx3.h is available but does not define the V1 API. This is odd. Please open a GitHub issue at: https://github.com/NVIDIA/cccl/issues.")
#    else
#      warning nvtx3.h is available but does not define the V1 API. This is odd. Please open a GitHub issue at: https://github.com/NVIDIA/cccl/issues.
#    endif
#    define _CCCL_NVTX_RANGE_SCOPE_IF(condition, name)
#    define _CCCL_NVTX_RANGE_SCOPE(name)
#  endif // NVTX3_CPP_DEFINITIONS_V1_0

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#else // _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h> ) && !defined(CCCL_DISABLE_NVTX) && !defined(NVTX_DISABLE)
#  define _CCCL_NVTX_RANGE_SCOPE_IF(condition, name)
#  define _CCCL_NVTX_RANGE_SCOPE(name)
#endif // _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h> ) && !defined(CCCL_DISABLE_NVTX) && !defined(NVTX_DISABLE)

#endif // _CUDA___NVTX_NVTX_H
