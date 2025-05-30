//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_EXCEPTION
#define __CUDAX_EXECUTION_EXCEPTION

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <exception> // IWYU pragma: export

#if _CCCL_CUDA_COMPILATION()
#  include <nv/target>
#  define _CUDAX_CATCH(...)
#  define _CUDAX_TRY(_TRY, _CATCH) \
    NV_IF_TARGET(                  \
      NV_IS_HOST, (try { _CCCL_PP_EXPAND _TRY } catch (...){_CCCL_PP_EXPAND _CATCH}), ({_CCCL_PP_EXPAND _TRY}))
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
#  define _CUDAX_CATCH(...)
#  define _CUDAX_TRY(_TRY, _CATCH) _CCCL_PP_EXPAND(try { _CCCL_PP_EXPAND _TRY } catch (...){_CCCL_PP_EXPAND _CATCH})
#endif // ^^^ !_CCCL_CUDA_COMPILATION() ^^^

#if _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)
// Treat everything as no-throw in device code
#  define _CUDAX_NOEXCEPT_EXPR(...) true
#else // ^^^ _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC) ^^^ /
      // vvv !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) vvv
// This is the default behavior for host code, and for nvc++
#  define _CUDAX_NOEXCEPT_EXPR(...) noexcept(__VA_ARGS__)
#endif // ^^^ !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) ^^^

#endif // __CUDAX_EXECUTION_EXCEPTION
