//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_CUDA_COMPILE_RESULT_CUH
#define _CUDAX___COMPILER_CUDA_COMPILE_RESULT_CUH

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
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/experimental/__compiler/cuda_compile_source.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <nvrtc.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Enumeration representing the status of the PCH (precompiled header) creation.
enum class cuda_pch_create_status
{
  success, //!< The PCH was created successfully
  not_attempted, //!< The PCH creation was not attempted or not allowed
  heap_exhausted, //!< The PCH heap was exhausted
  error, //!< An error occurred during PCH creation
};

class __nvrtc_compile_result_base
{
protected:
  __cuda_compile_source_id __src_id_{}; //!< The ID of the cuda compile source object
  ::nvrtcProgram __program_{}; //!< The NVRTC program object
  bool __success_{}; //!< The compilation success flag
  ::std::vector<_CUDA_VSTD::string_view> __lowered_names_{}; //!< The vector of lowered names, ptx and cubin only

  //! @brief Constructor for the NVRTC compile result base.
  //!
  //! @param __src_id The ID of the cuda compile source object.
  //! @param __program The NVRTC program object.
  //! @param __success The compilation success flag.
  //! @param __lowered_names The vector of lowered names (optional).
  __nvrtc_compile_result_base(__cuda_compile_source_id __src_id,
                              ::nvrtcProgram __program,
                              bool __success,
                              ::std::vector<_CUDA_VSTD::string_view> __lowered_names = {}) noexcept
      : __src_id_{__src_id}
      , __program_{__program}
      , __success_{__success}
      , __lowered_names_{_CUDA_VSTD::move(__lowered_names)}
  {}

  //! @brief Destroy the NVRTC program.
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
  __nvrtc_compile_result_base(no_init_t) noexcept {}

  __nvrtc_compile_result_base(const __nvrtc_compile_result_base&) = delete;

  //! @brief Move constructor.
  //!
  //! @param __other The other result to move from.
  __nvrtc_compile_result_base(__nvrtc_compile_result_base&& __other) noexcept
      : __src_id_{_CUDA_VSTD::exchange(__other.__src_id_, __cuda_compile_source_id{})}
      , __program_{_CUDA_VSTD::exchange(__other.__program_, nullptr)}
      , __success_{_CUDA_VSTD::exchange(__other.__success_, false)}
      , __lowered_names_{_CUDA_VSTD::move(__other.__lowered_names_)}
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
      __src_id_        = _CUDA_VSTD::exchange(__other.__src_id_, __cuda_compile_source_id{});
      __program_       = _CUDA_VSTD::exchange(__other.__program_, nullptr);
      __success_       = _CUDA_VSTD::exchange(__other.__success_, false);
      __lowered_names_ = _CUDA_VSTD::move(__other.__lowered_names_);
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

  //! @brief Was the compilation successful?
  //!
  //! @return `true` if the compilation was successful, `false` otherwise.
  [[nodiscard]] bool success() const noexcept
  {
    return __success_;
  }

  //! @brief Get the PCH create status.
  //!
  //! @return The PCH create status.
  [[nodiscard]] cuda_pch_create_status pch_create_status() const noexcept
  {
    switch (::nvrtcGetPCHCreateStatus(__program_))
    {
      case ::NVRTC_SUCCESS:
        return cuda_pch_create_status::success;
      case ::NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED:
        return cuda_pch_create_status::heap_exhausted;
      case ::NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED:
        return cuda_pch_create_status::not_attempted;
      case ::NVRTC_ERROR_PCH_CREATE:
        return cuda_pch_create_status::error;
      default:
        _CCCL_UNREACHABLE();
    }
  }

  //! @brief Get the PCH heap required size. Only valid if `cuda_pch_create_status::heap_exhausted` is returned by
  //!        `pch_create_status()`.
  //!
  //! @return The PCH required heap size.
  [[nodiscard]] _CUDA_VSTD::size_t pch_heap_required_size() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvrtcGetPCHHeapSizeRequired(__program_, &__size) != ::NVRTC_SUCCESS)
    {
      // todo: throw
    }
    return __size;
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

  //! @brief Get the lowered name for a name expression.
  //!
  //! @param __id The name expression id to get the lowered name for.
  //!
  //! @return A string view containing the lowered name.
  //!
  //! @note Only expressions that were specified during compilation can be lowered.
  [[nodiscard]] _CUDA_VSTD::string_view lowered_name(__cuda_name_expression_id __id_) const
  {
    _CCCL_ASSERT(__id_.__src_id_ == __src_id_, "Invalid name expression id");
    return (success()) ? __lowered_names_[__id_.__expr_idx_] : _CUDA_VSTD::string_view{};
  }

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

  //! @brief Get the lowered name for a name expression.
  //!
  //! @param __id The name expression id to get the lowered name for.
  //!
  //! @return A string view containing the lowered name.
  //!
  //! @note Only expressions that were specified during compilation can be lowered.
  [[nodiscard]] _CUDA_VSTD::string_view lowered_name(__cuda_name_expression_id __id_) const
  {
    _CCCL_ASSERT(__id_.__src_id_ == __src_id_, "Invalid name expression id");
    return (success()) ? __lowered_names_[__id_.__expr_idx_] : _CUDA_VSTD::string_view{};
  }

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

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_CUDA_COMPILE_RESULT_CUH
