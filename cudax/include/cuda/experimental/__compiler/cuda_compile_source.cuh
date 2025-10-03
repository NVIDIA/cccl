//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_CUDA_COMPILE_SOURCE_CUH
#define _CUDAX___COMPILER_CUDA_COMPILE_SOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief An identifier for a CUDA source code.
enum class __cuda_compile_source_id : _CUDA_VSTD::uint64_t
{
  __invalid = 0, //!< Invalid source ID
};

//! @brief An identifier for a name expression within a CUDA source code.
struct __cuda_name_expression_id
{
  __cuda_compile_source_id __src_id_; //!< The source code identifier.
  _CUDA_VSTD::size_t __expr_idx_; //!< The index of the name expression in the name expression vector.
};

//! @brief A class representing a source code to be compiled by the CUDA compiler.
class cuda_compile_source
{
  friend class cuda_compiler;

  ::std::string __name_{}; //!< The name of the source code.
  ::std::string __code_{}; //!< The source code to be compiled.
  ::std::vector<::std::string> __name_exprs_{}; //!< The name expressions in the source code.
  ::std::vector<::std::string> __pchs_{}; //!< The precompiled headers for the source code.
  __cuda_compile_source_id __id_{}; //!< The unique identifier for the source code.

  //! @brief Makes an unique id.
  //!
  //! @return An unique identifier for the CUDA source code.
  [[nodiscard]] static __cuda_compile_source_id __make_id() noexcept
  {
    using _IdCounter               = _CUDA_VSTD::underlying_type_t<__cuda_compile_source_id>;
    static _IdCounter __id_counter = 1;
    return __cuda_compile_source_id{__id_counter++};
  }

public:
  cuda_compile_source() = delete;

  //! @brief Constructor that initializes the object to move-from state.
  cuda_compile_source(no_init_t) noexcept {}

  //! @brief Constructor that initializes the object with the given name and code.
  //!
  //! @param __name The name of the source code.
  //! @param __code The source code to be compiled.
  cuda_compile_source(::std::string __name, ::std::string __code) noexcept
      : __name_{_CUDA_VSTD::move(__name)}
      , __code_{_CUDA_VSTD::move(__code)}
      , __name_exprs_{}
      , __id_{__make_id()}
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
  //!
  //! @return An identifier for the added name expression of an undefined type.
  [[nodiscard]] __cuda_name_expression_id add_name_expression(::std::string __name_expr)
  {
    __name_exprs_.push_back(_CUDA_VSTD::move(__name_expr));
    return __cuda_name_expression_id{__id_, __name_exprs_.size() - 1};
  }

  //! @brief Adds a precompiled header.
  //!
  //! @param __pch The precompiled header to be added.
  void add_pch(::std::string __pch)
  {
    __pchs_.push_back(_CUDA_VSTD::move(__pch));
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_CUDA_COMPILE_SOURCE_CUH
