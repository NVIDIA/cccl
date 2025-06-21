//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BINUTILS_DEMANGLE_H
#define _CUDA___BINUTILS_DEMANGLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/cstdlib>
#  include <cuda/std/string_view>

#  include <string>

#  include <nv_decode.h>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Demangles a CUDA C++ mangled name.
//!
//! @param __name The mangled name to demangle.
//!
//! @return A `std::string` containing the demangled name.
//!
//! @throws cuda::cuda_error if memory allocation fails, or if the input name is invalid.
[[nodiscard]] ::std::string demangle(_CUDA_VSTD::string_view __name)
{
  // input must be zero-terminated, so we convert string_view to std::string
  ::std::string __name_in{__name.begin(), __name.end()};

  int __status{};
  char* __dname_ptr = ::__cu_demangle(__name_in.c_str(), nullptr, nullptr, &__status);

  _CCCL_TRY
  {
    switch (__status)
    {
      case 0: {
        ::std::string __ret{__dname_ptr};
        _CUDA_VSTD::free(__dname_ptr);
        return __ret;
      }
      case -1:
        ::cuda::__throw_cuda_error(::cudaErrorMemoryAllocation, "failed to allocate memory for demangled name");
      case -2:
        ::cuda::__throw_cuda_error(::cudaErrorInvalidSymbol, "invalid mangled name passed to demangle function");
      case -3:
        ::cuda::__throw_cuda_error(::cudaErrorInvalidValue, "invalid value passed to demangle function");
      default:
        // Should not happen, but if it does, we throw an unknown error
        ::cuda::__throw_cuda_error(::cudaErrorUnknown, "unknown error during demangling occurred");
    }
  }
  _CCCL_CATCH_ALL
  {
    // If an exception is thrown, free the allocated memory and rethrow the exception
    _CUDA_VSTD::free(__dname_ptr);
    throw;
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___BINUTILS_DEMANGLE_H
