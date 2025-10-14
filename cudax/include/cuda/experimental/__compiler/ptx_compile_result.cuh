//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_PTX_COMPILE_RESULT_CUH
#define _CUDAX___COMPILER_PTX_COMPILE_RESULT_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/exchange.h>

#include <cuda/experimental/__compiler/ptx_compile_source.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <nvPTXCompiler.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Result of compiling PTX to CUBIN.
class compile_ptx_to_cubin_result
{
  friend class ptx_compiler;

  ::nvPTXCompilerHandle __handle_{}; //!< The NVPTX compiler handle.
  bool __success_{}; //!< The compilation success flag.

  //! @brief Constructor for the PTX to CUBIN compile result.
  //!
  //! @param __handle The NVPTX compiler handle.
  //! @param __success The compilation success flag.
  compile_ptx_to_cubin_result(::nvPTXCompilerHandle __handle, bool __success) noexcept
      : __handle_{__handle}
      , __success_{__success}
  {}

  //! @brief Destroy the NVPTX compiler handle.
  void __destroy() noexcept
  {
    if (__handle_ != nullptr)
    {
      [[maybe_unused]] auto __status = ::nvPTXCompilerDestroy(&__handle_);
    }
  }

public:
  compile_ptx_to_cubin_result() = delete;

  //! @brief Constructor for an uninitialized result.
  //!
  //! @param __uninit An uninitialized tag.
  compile_ptx_to_cubin_result(no_init_t) noexcept {}

  compile_ptx_to_cubin_result(const compile_ptx_to_cubin_result&) = delete;

  //! @brief Move constructor.
  //!
  //! @param __other The other result to move from.
  compile_ptx_to_cubin_result(compile_ptx_to_cubin_result&& __other) noexcept
      : __handle_{_CUDA_VSTD::exchange(__other.__handle_, nullptr)}
      , __success_{_CUDA_VSTD::exchange(__other.__success_, false)}
  {}

  //! @brief Destructor.
  ~compile_ptx_to_cubin_result() noexcept
  {
    __destroy();
  }

  compile_ptx_to_cubin_result& operator=(const compile_ptx_to_cubin_result&) = delete;

  //! @brief Move assignment operator.
  //!
  //! @param __other The other result to move from.
  //!
  //! @return A reference to this result.
  compile_ptx_to_cubin_result& operator=(compile_ptx_to_cubin_result&& __other) noexcept
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      __destroy();
      __handle_  = _CUDA_VSTD::exchange(__other.__handle_, nullptr);
      __success_ = _CUDA_VSTD::exchange(__other.__success_, false);
    }
    return *this;
  }

  //! @brief Was the compilation successful?
  //!
  //! @return `true` if the compilation was successful, `false` otherwise.
  [[nodiscard]] bool success() const noexcept
  {
    return __success_;
  }

  //! @brief Convert the result to a boolean value.
  //!
  //! @return `true` if the compilation was successful, `false` otherwise.
  explicit operator bool() const noexcept
  {
    return __success_;
  }

  //! @brief Get the log.
  //!
  //! @return A string containing the log.
  [[nodiscard]] ::std::string log() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvPTXCompilerGetErrorLogSize(__handle_, &__size) != ::NVPTXCOMPILE_SUCCESS)
    {
      // todo: throw
    }
    ::std::string __log(__size, '\0');
    if (::nvPTXCompilerGetErrorLog(__handle_, __log.data()) != ::NVPTXCOMPILE_SUCCESS)
    {
      // todo: throw
    }
    return __log;
  }

  //! @brief Get the compiled CUBIN.
  //!
  //! @return A vector containing the compiled CUBIN.
  [[nodiscard]] ::std::vector<_CUDA_VSTD_NOVERSION::byte> cubin() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvPTXCompilerGetCompiledProgramSize(__handle_, &__size) != ::NVPTXCOMPILE_SUCCESS)
    {
      // todo: throw
    }
    ::std::vector<_CUDA_VSTD_NOVERSION::byte> __code(__size);
    if (::nvPTXCompilerGetCompiledProgram(__handle_, __code.data()) != ::NVPTXCOMPILE_SUCCESS)
    {
      // todo: throw
    }
    return __code;
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_PTX_COMPILE_RESULT_CUH
