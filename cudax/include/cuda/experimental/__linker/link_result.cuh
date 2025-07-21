//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LINKER_LINK_RESULT_CUH
#define _CUDAX___LINKER_LINK_RESULT_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/exchange.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__linker/nvjitlink.cuh>

#include <string>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

class __link_result_base
{
protected:
  ::nvJitLinkHandle __handle_;
  bool __success_;

  __link_result_base(::nvJitLinkHandle handle, bool success) noexcept
      : __handle_(handle)
      , __success_(success)
  {}

  void __destroy() noexcept
  {
    if (__handle_ != nullptr)
    {
      [[maybe_unused]] auto __status = ::nvJitLinkDestroy(&__handle_);
    }
  }

public:
  __link_result_base() = delete;

  //! @brief Constructs the object to uninitialized state.
  __link_result_base(no_init_t) noexcept
      : __handle_(nullptr)
      , __success_(false)
  {}

  __link_result_base(const __link_result_base&) = delete;

  //! @brief Move constructor.
  //!
  //! @param __other The object to move from.
  __link_result_base(__link_result_base&& __other) noexcept
      : __handle_(_CUDA_VSTD::exchange(__other.__handle_, nullptr))
      , __success_(_CUDA_VSTD::exchange(__other.__success_, false))
  {}

  //! @brief Destructor.
  ~__link_result_base() noexcept
  {
    __destroy();
  }

  __link_result_base& operator=(const __link_result_base&) = delete;

  //! @brief Move assignment operator.
  //!
  //! @param __other The object to move from.
  __link_result_base& operator=(__link_result_base&& __other) noexcept
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      __destroy();
      __handle_  = _CUDA_VSTD::exchange(__other.__handle_, nullptr);
      __success_ = _CUDA_VSTD::exchange(__other.__success_, false);
    }
    return *this;
  }

  //! @brief Get the log.
  //!
  //! @return A string containing the log.
  [[nodiscard]] ::std::string log() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvJitLinkGetErrorLogSize(__handle_, &__size) != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    ::std::string __log(__size, '\0');
    if (::nvJitLinkGetErrorLog(__handle_, __log.data()) != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    return __log;
  }

  //! @brief Was the link operation successful?
  //!
  //! @return `true` if the link operation was successful, `false` otherwise.
  [[nodiscard]] bool success() const noexcept
  {
    return __success_;
  }

  //! @brief Check if the link operation was successful.
  //!
  //! @return `true` if the link operation was successful, `false` otherwise.
  explicit operator bool() const noexcept
  {
    return __success_;
  }
};

//! @brief Class representing the result of a PTX link operation.
struct link_to_ptx_result : public __link_result_base
{
  friend class ptx_linker;

  using _Base = __link_result_base;

  using _Base::_Base;
  using _Base::operator=;
  using _Base::operator bool;

  //! @brief Get the linked PTX.
  //!
  //! @return A string containing the linked PTX.
  [[nodiscard]] ::std::string ptx() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvJitLinkGetLinkedPtxSize(__handle_, &__size) != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    ::std::string __ptx(__size, '\0');
    if (::nvJitLinkGetLinkedPtx(__handle_, __ptx.data()) != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    return __ptx;
  }
};

//! @brief Class representing the result of a CUBIN link operation.
struct link_to_cubin_result : public __link_result_base
{
  friend class cubin_linker;

  using _Base = __link_result_base;

  using _Base::_Base;
  using _Base::operator=;
  using _Base::operator bool;

  //! @brief Get the linked CUBIN.
  //!
  //! @return A string containing the linked CUBIN.
  [[nodiscard]] ::std::vector<_CUDA_VSTD_NOVERSION::byte> cubin() const
  {
    _CUDA_VSTD::size_t __size{};
    if (::nvJitLinkGetLinkedCubinSize(__handle_, &__size) != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    ::std::vector<_CUDA_VSTD_NOVERSION::byte> __cubin(__size);
    if (::nvJitLinkGetLinkedCubin(__handle_, __cubin.data()) != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    return __cubin;
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LINKER_LINK_RESULT_CUH
