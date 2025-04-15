//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_EXCEPTION
#define __CUDAX_ASYNC_DETAIL_EXCEPTION

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__detail/config.cuh>

#include <exception> // IWYU pragma: export

#if defined(__CUDACC__)
#  include <nv/target>
#  define _CUDAX_CATCH(...)
#  define _CUDAX_TRY(_TRY, _CATCH) \
    NV_IF_TARGET(                  \
      NV_IS_HOST, (try { _CCCL_PP_EXPAND _TRY } catch (...){_CCCL_PP_EXPAND _CATCH}), ({_CCCL_PP_EXPAND _TRY}))
#else
#  define _CUDAX_CATCH(...)
#  define _CUDAX_TRY(_TRY, _CATCH) _NV_EVAL(try { _CCCL_PP_EXPAND _TRY } catch (...){_CCCL_PP_EXPAND _CATCH})
#endif

#if defined(__CUDA_ARCH__)
// Treat everything as no-throw in device code
#  define _CUDAX_NOEXCEPT_EXPR(...) true
#else
// This is the default behavior for host code, and for nvc++
#  define _CUDAX_NOEXCEPT_EXPR(...) noexcept(__VA_ARGS__)
#endif

#endif
