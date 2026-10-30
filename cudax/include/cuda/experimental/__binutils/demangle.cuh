//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___BINUTILS_DEMANGLE_CUH
#define _CUDAX___BINUTILS_DEMANGLE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED() && __has_include(<nv_decode.h>)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/new>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/cstdlib>
#  include <cuda/std/string_view>

#  include <string>

#  include <nv_decode.h>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// todo: make this function take cuda::std::cstring_view once P3655 is merged to C++29 and implemented in libcu++

//! @brief Demangles a CUDA C++ mangled name.
//!
//! @note Use of this function introduces a dependency on the cu++filt library which is part of the cuxxfilt package
//!       from the CUDA Toolkit and must be linked to the program.
//!
//! @param __name The mangled name to demangle.
//!
//! @return A \c std::string containing the demangled name.
//!
//! @throws \c std::bad_alloc if memory allocation fails.
//! @throws \c std::runtime_error if the passed \c __name is not a valid mangled symbol or an unknown error happens.
template <class _Dummy = void>
[[nodiscard]] _CCCL_HOST_API ::std::string demangle(::cuda::std::string_view __name)
{
  // input must be zero-terminated, so we convert string_view to std::string
  ::std::string __name_in{__name.data(), __name.size()};

  int __status{};
  char* __dname = ::__cu_demangle(__name_in.c_str(), nullptr, nullptr, &__status);

  _CCCL_TRY
  {
    switch (__status)
    {
      case 0: {
        ::std::string __ret{__dname};
        ::cuda::std::free(__dname);
        return __ret;
      }
      case -1:
        _CCCL_THROW(::std::bad_alloc);
      case -2:
        _CCCL_THROW(::std::runtime_error, "invalid mangled name passed to cuda::demangle function");
      case -3:
        _CCCL_VERIFY(false, "cccl internal error - invalid argument passed to __cu_demangle");
      default:
        _CCCL_THROW(::std::runtime_error, "an unknown error occurred during demangling operation");
    }
  }
  _CCCL_CATCH_ALL
  {
    // If an exception is thrown, free the allocated memory and rethrow the exception
    ::cuda::std::free(__dname);
    _CCCL_RETHROW;
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HOSTED() && __has_include(<nv_decode.h>)

#endif // _CUDAX___BINUTILS_DEMANGLE_CUH
