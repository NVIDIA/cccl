//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LIBRARY_LIBRARY_REF_CUH
#define _CUDAX___LIBRARY_LIBRARY_REF_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/cuda_error.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

template <class _Signature>
class kernel_ref;

//! @brief Information about a symbol in a CUDA library
struct library_symbol_info
{
  void* ptr; //!< Address of the symbol
  ::cuda::std::size_t size; //!< Size of the symbol in bytes
};

//! @brief A non-owning wrapper for a CUDA library handle
class library_ref
{
public:
#if _CCCL_CTK_BELOW(12, 7)
  using value_type = ::CUlibrary;
#else // ^^^ _CCCL_CTK_BELOW(12, 7) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 7) vvv
  using value_type = ::cudaLibrary_t;
#endif // _CCCL_CTK_BELOW(12, 7)

  //! @brief Disallow construction from a null pointer
  library_ref(::cuda::std::nullptr_t) = delete;

  //! @brief Constructs a `library_ref` from a CUlibrary handle
  //!
  //! @param __library The CUlibrary handle to wrap
  constexpr library_ref(value_type __library) noexcept
      : __library_((::CUlibrary) __library)
  {}

  library_ref(const library_ref&) = default;

  //! @brief Checks if the library contains a kernel with the given name
  //!
  //! @param __name The name of the kernel to check for
  //!
  //! @return true if the library contains a kernel with the given name, false otherwise
  //!
  //! @throws cuda_error if the library could not be queried for the kernel
  [[nodiscard]] bool has_kernel(const char* __name) const
  {
    ::CUkernel __kernel{};
    switch (const auto __res = _CUDA_DRIVER::__libraryGetKernelNoThrow(__kernel, __library_, __name))
    {
      case ::cudaSuccess:
        return true;
      case ::cudaErrorSymbolNotFound:
        return false;
      default:
        ::cuda::__throw_cuda_error(__res, "Failed to get the kernel from library");
    }
  }

  //! @brief Gets a reference to a kernel from the library
  //!
  //! @tparam _Signature The signature of the kernel to retrieve in form of `void(Args...)`
  //!
  //! @param __name The name of the kernel to retrieve
  //!
  //! @return A `kernel_ref` that refers to the kernel with the given name
  //!
  //! @throws cuda_error if the kernel could not be found in the library
  template <class _Signature>
  [[nodiscard]] kernel_ref<_Signature> kernel(const char* __name) const
  {
    ::CUkernel __kernel{};
    if (const auto __res = _CUDA_DRIVER::__libraryGetKernelNoThrow(__kernel, __library_, __name);
        __res != ::cudaSuccess)
    {
      ::cuda::__throw_cuda_error(__res, "Failed to get the kernel from the library");
    }
    return kernel_ref<_Signature>{__kernel};
  }

  //! @brief Checks if the library contains a global symbol with the given name on a device
  //!
  //! @param __name The name of the global symbol to check for
  //! @param __device The device on which to check for the global symbol
  //!
  //! @return true if the library contains a global symbol with the given name, false otherwise
  //!
  //! @throws cuda_error if the library could not be queried for the global symbol
  [[nodiscard]] bool has_global(const char* __name, ::cuda::device_ref __device) const
  {
    ::cuda::__ensure_current_context __ctx_guard(__device);

    ::CUdeviceptr __dptr{};
    ::cuda::std::size_t __size{};
    switch (const auto __res = _CUDA_DRIVER::__libraryGetGlobalNoThrow(__dptr, __size, __library_, __name))
    {
      case ::cudaSuccess:
        return true;
      case ::cudaErrorSymbolNotFound:
        return false;
      default:
        ::cuda::__throw_cuda_error(__res, "Failed to get the global symbol from library");
    }
  }

  //! @brief Gets a pointer and size of a global symbol from the library on a device
  //!
  //! @param __name The name of the global symbol to retrieve
  //! @param __device The device on which to retrieve the global symbol
  //!
  //! @return A pair containing a device pointer to the global symbol and its size
  //!
  //! @throws cuda_error if the global symbol could not be found in the library
  [[nodiscard]] library_symbol_info global(const char* __name, ::cuda::device_ref __device) const
  {
    ::cuda::__ensure_current_context __ctx_guard(__device);

    ::CUdeviceptr __dptr{};
    ::cuda::std::size_t __size{};
    if (const auto __res = _CUDA_DRIVER::__libraryGetGlobalNoThrow(__dptr, __size, __library_, __name);
        __res != ::cudaSuccess)
    {
      ::cuda::__throw_cuda_error(__res, "Failed to get the global symbol from the library");
    }
    return library_symbol_info{reinterpret_cast<void*>(__dptr), __size};
  }

  //! @brief Checks if the library contains a managed symbol with the given name
  //!
  //! @param __name The name of the managed symbol to check for
  //!
  //! @return true if the library contains a managed symbol with the given name, false otherwise
  //!
  //! @throws cuda_error if the library could not be queried for the managed symbol
  //!
  //! @note Managed memory is shared across devices
  [[nodiscard]] bool has_managed(const char* __name) const
  {
    ::CUdeviceptr __dptr{};
    ::cuda::std::size_t __size{};
    switch (const auto __res = _CUDA_DRIVER::__libraryGetManagedNoThrow(__dptr, __size, __library_, __name))
    {
      case ::cudaSuccess:
        return true;
      case ::cudaErrorSymbolNotFound:
        return false;
      default:
        ::cuda::__throw_cuda_error(__res, "Failed to get the managed symbol from library");
    }
  }

  //! @brief Gets a pointer and size of a managed symbol from the library
  //!
  //! @param __name The name of the managed symbol to retrieve
  //!
  //! @return A pair containing a pointer to the managed symbol and its size
  //!
  //! @throws cuda_error if the managed symbol could not be found in the library
  //!
  //! @note Managed memory is shared across devices
  [[nodiscard]] library_symbol_info managed(const char* __name) const
  {
    ::CUdeviceptr __dptr{};
    ::cuda::std::size_t __size{};
    if (const auto __res = _CUDA_DRIVER::__libraryGetManagedNoThrow(__dptr, __size, __library_, __name);
        __res != ::cudaSuccess)
    {
      ::cuda::__throw_cuda_error(__res, "Failed to get the managed symbol from the library");
    }
    return library_symbol_info{reinterpret_cast<void*>(__dptr), __size};
  }

  //! @brief Gets the CUlibrary handle
  //!
  //! @return The CUlibrary handle wrapped by this `library_ref`
  [[nodiscard]] constexpr value_type get() const noexcept
  {
    return (value_type) __library_;
  }

  //! @brief Compares two `library_ref` for equality
  //!
  //! @param __lhs The first `library_ref` to compare
  //! @param __rhs The second `library_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same library
  [[nodiscard]] friend constexpr bool operator==(library_ref __lhs, library_ref __rhs) noexcept
  {
    return __lhs.__library_ == __rhs.__library_;
  }

  //! @brief Compares two `library_ref` for inequality
  //!
  //! @param __lhs The first `library_ref` to compare
  //! @param __rhs The second `library_ref` to compare
  //! @return true if `lhs` and `rhs` refer to a different library
  [[nodiscard]] friend constexpr bool operator!=(library_ref __lhs, library_ref __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

protected:
  ::CUlibrary __library_;
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LIBRARY_LIBRARY_REF_CUH
