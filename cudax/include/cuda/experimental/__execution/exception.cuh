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

#include <cuda/std/__cccl/exceptions.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__utility/move.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <exception> // IWYU pragma: keep
#endif // !_CCCL_COMPILER(NVRTC)

namespace cuda::experimental::execution
{
// Since there are no exceptions in device code, we provide a stub implementation of
// std::exception_ptr and related functions.
#if _CCCL_COMPILER(NVRTC) || !_CCCL_HOST_COMPILATION()

struct exception_ptr
{
private:
  struct __nullptr_t
  {};

  //! In libstdc++ and libc++, std::exception_ptr is the size of a pointer, but in MSVC it
  //! is the size of two pointers. We must match that size here to avoid breaking the ABI
  //! of any types that contain an exception_ptr.
  void* __ptrs[1 + _CCCL_COMPILER(MSVC)] = {};

public:
  _CCCL_HIDE_FROM_ABI exception_ptr() noexcept = default;

  //! For conversion from nullptr so that code like:
  //!
  //! @code
  //! std::exception_ptr eptr = nullptr;
  //! @endcode
  //!
  //! and
  //!
  //! @code
  //! eptr == nullptr
  //! @endcode
  //!
  //! works as expected.
  _CCCL_API constexpr exception_ptr(const __nullptr_t* __ptr) noexcept
      : exception_ptr()
  {
    _CCCL_ASSERT(__ptr == nullptr, "Can only construct exception_ptr from nullptr");
  }

  [[nodiscard]] _CCCL_API explicit constexpr operator bool() const noexcept
  {
    return false;
  }

  [[nodiscard]] _CCCL_API constexpr bool operator!() const noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const exception_ptr&, const exception_ptr&) noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const exception_ptr&, const exception_ptr&) noexcept
  {
    return false;
  }
};

[[nodiscard]] _CCCL_API inline exception_ptr current_exception() noexcept
{
  return exception_ptr{};
}

[[noreturn]] _CCCL_API inline void rethrow_exception(const exception_ptr&)
{
  ::cuda::__throw_cuda_error(cudaErrorUnknown, "unknown exception");
}

// ^^^ _CCCL_COMPILER(NVRTC) || !_CCCL_HOST_COMPILATION() ^^^
#else
// vvv !_CCCL_COMPILER(NVRTC) && _CCCL_HOST_COMPILATION() vvv

using ::std::current_exception;
using ::std::exception_ptr;
using ::std::rethrow_exception;

#endif // !_CCCL_COMPILER(NVRTC) && _CCCL_HOST_COMPILATION()
} // namespace cuda::experimental::execution

#endif // __CUDAX_EXECUTION_EXCEPTION
