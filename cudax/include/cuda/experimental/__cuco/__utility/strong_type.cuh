//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO__UTILITY__STRONG_TYPE_CUH
#define _CUDAX__CUCO__UTILITY__STRONG_TYPE_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! A strong type wrapper
//!
//! Template parameter:
//! - `_T`: Type of the underlying value
template <class _T>
struct __strong_type
{
  //! Constructs a strong type
  //!
  //! Parameter:
  //! - `v`: Value to be wrapped as a strong type
  __host__ __device__ explicit constexpr __strong_type(_T __v)
      : __value{__v}
  {}

  //! Implicit conversion operator to the underlying value.
  //!
  //! Returns: Underlying value
  __host__ __device__ constexpr operator _T() const noexcept
  {
    return __value;
  }

  _T __value; //!< Underlying data value
};
} // namespace cuda::experimental::cuco

//! Convenience wrapper for defining a strong type
#define CUDAX_CUCO_DEFINE_STRONG_TYPE(Name, Type)                      \
  struct Name : public ::cuda::experimental::cuco::__strong_type<Type> \
  {                                                                    \
    __host__ __device__ explicit constexpr Name(Type __value)          \
        : ::cuda::experimental::cuco::__strong_type<Type>(__value)     \
    {}                                                                 \
  };

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO__UTILITY__STRONG_TYPE_CUH
