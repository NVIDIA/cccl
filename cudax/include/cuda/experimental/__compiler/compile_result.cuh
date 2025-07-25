//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_COMPILE_RESULT_CUH
#define _CUDAX___COMPILER_COMPILE_RESULT_CUH

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
#include <cuda/std/string_view>

#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <nvPTXCompiler.h>
#include <nvrtc.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

class __nvrtc_compile_result_base
{
protected:
  ::nvrtcProgram __program_;
  bool __success_;

  __nvrtc_compile_result_base(::nvrtcProgram __program, bool __success) noexcept
      : __program_{__program}
      , __success_{__success}
  {}

  void __destroy() noexcept
  {
    if (__program_ != nullptr)
    {
      [[maybe_unused]] auto __status = ::nvrtcDestroyProgram(&__program_);
    }
  }

public:
  __nvrtc_compile_result_base() = delete;

  //! @brief Constructor for an uninitialized result.
  //!
  //! @param __uninit An uninitialized tag.
  __nvrtc_compile_result_base(no_init_t) noexcept
      : __program_{nullptr}
      , __success_{false}
  {}

  __nvrtc_compile_result_base(const __nvrtc_compile_result_base&) = delete;

  //! @brief Move constructor.
  //!
  //! @param __other The other result to move from.
  __nvrtc_compile_result_base(__nvrtc_compile_result_base&& __other) noexcept
      : __program_{_CUDA_VSTD::exchange(__other.__program_, nullptr)}
      , __success_{_CUDA_VSTD::exchange(__other.__success_, false)}
  {}

  //! @brief Destructor.
  ~__nvrtc_compile_result_base() noexcept
  {
    __destroy();
  }

  __nvrtc_compile_result_base& operator=(const __nvrtc_compile_result_base&) = delete;

  //! @brief Move assignment operator.
  //!
  //! @param __other The other result to move from.
  //!
  //! @return A reference to this result.
  __nvrtc_compile_result_base& operator=(__nvrtc_compile_result_base&& __other) noexcept
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      __destroy();
      __program_ = _CUDA_VSTD::exchange(__other.__program_, nullptr);
      __success_ = _CUDA_VSTD::exchange(__other.__success_, false);
    }
    return *this;
  }

  //! @brief Get the log.
  //!
  //! @return A string containing the info log.
  [[nodiscard]] ::std::string log() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvrtcGetProgramLogSize(__program_, &__size) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    ::std::string __log(__size, '\0');
    if (::nvrtcGetProgramLog(__program_, __log.data()) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    __log.resize(__size - 1);
    return __log;
  }

  //! @brief Get the lowered name for a name expression.
  //!
  //! @param __name The name expression to get the lowered name for.
  //!
  //! @return A string view containing the lowered name.
  //!
  //! @note Only expressions that were specified during compilation can be lowered.
  [[nodiscard]] _CUDA_VSTD::string_view lowered_name(_CUDA_VSTD::string_view __name) const
  {
    const char* __lowered_name{};
    ::std::string __tmp(__name.begin(), __name.end());
    if (::nvrtcGetLoweredName(__program_, __tmp.c_str(), &__lowered_name) != ::NVRTC_SUCCESS)
    {
      // todo: throw an exception if the lowered name could not be retrieved
    }
    return _CUDA_VSTD::string_view{__lowered_name};
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
};

//! @brief Result of compiling CUDA source code to PTX.
struct compile_cuda_to_ptx_result : public __nvrtc_compile_result_base
{
  friend class cuda_compiler;

  using _Base = __nvrtc_compile_result_base;

  using _Base::_Base;
  using _Base::operator=;
  using _Base::operator bool;

  //! @brief Get the compiled PTX.
  //!
  //! @brief A string containing the compiled PTX code.
  [[nodiscard]] ::std::string ptx() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvrtcGetPTXSize(__program_, &__size) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    ::std::string __code(__size, '\0');
    if (::nvrtcGetPTX(__program_, __code.data()) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    return __code;
  }
};

//! @brief Result of compiling CUDA source code to CUBIN.
struct compile_cuda_to_cubin_result : public __nvrtc_compile_result_base
{
  friend class cuda_compiler;

  using _Base = __nvrtc_compile_result_base;

  using _Base::_Base;
  using _Base::operator=;
  using _Base::operator bool;

  //! @brief Get the compiled PTX.
  //!
  //! @brief A string containing the compiled PTX code.
  [[nodiscard]] ::std::string ptx() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvrtcGetPTXSize(__program_, &__size) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    ::std::string __code(__size, '\0');
    if (::nvrtcGetPTX(__program_, __code.data()) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    return __code;
  }

  //! @brief Get the compiled CUBIN.
  //!
  //! @return A vector containing the compiled CUBIN code.
  [[nodiscard]] ::std::vector<_CUDA_VSTD_NOVERSION::byte> cubin() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvrtcGetCUBINSize(__program_, &__size) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    ::std::vector<_CUDA_VSTD_NOVERSION::byte> __code(__size);
    if (::nvrtcGetCUBIN(__program_, reinterpret_cast<char*>(__code.data())) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    return __code;
  }
};

//! @brief Result of compiling CUDA source code to LTOIR.
struct compile_cuda_to_ltoir_result : public __nvrtc_compile_result_base
{
  friend class cuda_compiler;

  using _Base = __nvrtc_compile_result_base;

  using _Base::_Base;
  using _Base::operator=;
  using _Base::operator bool;

  //! @brief Get the compiled LTOIR.
  //!
  //! @return A vector containing the compiled LTOIR code.
  [[nodiscard]] ::std::vector<_CUDA_VSTD_NOVERSION::byte> ltoir() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvrtcGetLTOIRSize(__program_, &__size) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    ::std::vector<_CUDA_VSTD_NOVERSION::byte> __code(__size);
    if (::nvrtcGetLTOIR(__program_, reinterpret_cast<char*>(__code.data())) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    return __code;
  }
};

//! @brief Result of compiling PTX to CUBIN.
class compile_ptx_to_cubin_result
{
  friend class ptx_compiler;

  ::nvPTXCompilerHandle __handle_;
  bool __success_;

  compile_ptx_to_cubin_result(::nvPTXCompilerHandle __handle, bool __success) noexcept
      : __handle_{__handle}
      , __success_{__success}
  {}

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
  compile_ptx_to_cubin_result(no_init_t) noexcept
      : __handle_{nullptr}
      , __success_{false}
  {}

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

#endif // _CUDAX___COMPILER_COMPILE_CUH
