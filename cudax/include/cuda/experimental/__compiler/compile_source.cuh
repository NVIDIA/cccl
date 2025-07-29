//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_COMPILE_SOURCE_CUH
#define _CUDAX___COMPILER_COMPILE_SOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/string_view>

#include <cuda/experimental/__detail/utility.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief A class representing a source code to be compiled by the CUDA compiler.
class cuda_compile_source
{
  friend class cuda_compiler;

  _CUDA_VSTD::string_view __name_;
  _CUDA_VSTD::string_view __code_;
  ::std::vector<_CUDA_VSTD::string_view> __name_exprs_;

public:
  cuda_compile_source() = delete;

  //! @brief Constructor that initializes the object to move-from state.
  cuda_compile_source(no_init_t) noexcept {}

  //! @brief Constructor that initializes the object with the given name and code.
  //!
  //! @param __name The name of the source code.
  //! @param __code The source code to be compiled.
  cuda_compile_source(_CUDA_VSTD::string_view __name, _CUDA_VSTD::string_view __code) noexcept
      : __name_{__name}
      , __code_{__code}
      , __name_exprs_{}
  {}

  cuda_compile_source(const cuda_compile_source&) = delete;

  //! @brief Move constructor.
  cuda_compile_source(cuda_compile_source&&) noexcept = default;

  cuda_compile_source& operator=(const cuda_compile_source&) = delete;

  //! @brief Move assignment operator.
  cuda_compile_source& operator=(cuda_compile_source&&) noexcept = default;

  //! @brief Adds a name expression to the source code.
  //!
  //! @param __name_expr The name expression to be added.
  void add_name_expression(_CUDA_VSTD::string_view __name_expr)
  {
    __name_exprs_.push_back(__name_expr);
  }
};

//! @brief A class representing a PTX source code to be compiled by the CUDA compiler.
class ptx_compile_source
{
  friend class ptx_compiler;

  _CUDA_VSTD::string_view __name_;
  _CUDA_VSTD::string_view __code_;
  ::std::vector<_CUDA_VSTD::string_view> __symbols_;

public:
  ptx_compile_source() = delete;

  //! @brief Constructor that initializes the object to move-from state.
  ptx_compile_source(no_init_t) noexcept {}

  //! @brief Constructor that initializes the object with the given name and code.
  //!
  //! @param __name The name of the source code.
  //! @param __code The source code to be compiled.
  ptx_compile_source(_CUDA_VSTD::string_view __name, _CUDA_VSTD::string_view __code) noexcept
      : __name_{__name}
      , __code_{__code}
      , __symbols_{}
  {}

  ptx_compile_source(const ptx_compile_source&) = delete;

  //! @brief Move constructor.
  ptx_compile_source(ptx_compile_source&&) noexcept = default;

  ptx_compile_source& operator=(const ptx_compile_source&) = delete;

  //! @brief Move assignment operator.
  ptx_compile_source& operator=(ptx_compile_source&&) noexcept = default;

  //! @brief Adds a kernel symbol to be kept in the PTX source code.
  //!
  //! @param __symbol The kernel symbol to be added.
  void add_kernel_symbol(_CUDA_VSTD::string_view __symbol)
  {
    __symbols_.push_back(__symbol);
  }

  //! @brief Adds a function symbol to be kept in the PTX source code.
  //!
  //! @param __symbol The function symbol to be added.
  void add_function_symbol(_CUDA_VSTD::string_view __symbol)
  {
    __symbols_.push_back(__symbol);
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_COMPILE_SOURCE_CUH
