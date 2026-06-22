//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___NCCL_SHARED_LIBRARY_H
#define _CUDA_EXPERIMENTAL___NCCL_SHARED_LIBRARY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__host_stdlib/memory>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/cstdint>

#include <stdexcept>

#if _CCCL_OS(WINDOWS)
#  include <windows.h>
#else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
#  include <dlfcn.h>
#endif // ^^^ !_CCCL_OS(WINDOWS) ^^^

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class __shared_library
{
#if _CCCL_OS(WINDOWS)
  static constexpr ::cuda::std::int32_t __platform_default_flags = LOAD_LIBRARY_SEARCH_SYSTEM32;

  struct __platform_deleter
  {
    using pointer = HMODULE;

    _CCCL_HOST_API void operator()(HMODULE __mod) const noexcept
    {
      static_cast<void>(::FreeLibrary(__mod));
    }
  };

  using __platform_handle_t = ::std::unique_ptr<::cuda::std::remove_pointer_t<HMODULE>, __platform_deleter>;

  [[nodiscard]] static __platform_handle_t
  __load_lib_platform(const char* const __lib_name, const ::cuda::std::int32_t __flags)
  {
    return __platform_handle_t{::LoadLibraryExA(__lib_name, /*hFile=*/nullptr, static_cast<DWORD>(__flags))};
  }

  [[nodiscard]] void* __load_symbol_platform(const char* const __sym_name) const
  {
    void* const __sym = ::GetProcAddress(handle(), __sym_name);

    if (__sym == nullptr)
    {
      _CCCL_THROW(::std::invalid_argument, "Failed to locate the symbol in the shared library");
    }
    return __sym;
  }
#else
  static constexpr ::cuda::std::int32_t __platform_default_flags = RTLD_LAZY | RTLD_LOCAL;

  struct __platform_deleter
  {
    _CCCL_HOST_API void operator()(void* const __mod) const noexcept
    {
      static_cast<void>(::dlclose(__mod));
    }
  };

  using __platform_handle_t = ::std::unique_ptr<void, __platform_deleter>;

  [[nodiscard]] static __platform_handle_t
  __load_lib_platform(const char* const __lib_name, const ::cuda::std::int32_t __flags)
  {
    static_cast<void>(::dlerror());

    return __platform_handle_t{::dlopen(__lib_name, __flags)};
  }

  [[nodiscard]] void* __load_symbol_platform(const char* const __sym_name) const
  {
    static_cast<void>(::dlerror());

    auto* const __sym = ::dlsym(handle(), __sym_name);

    if (const char* const __error = ::dlerror(); __error || !__sym)
    {
      _CCCL_THROW(::std::invalid_argument, "Failed to locate the symbol in the shared library");
    }
    return __sym;
  }
#endif

public:
  __shared_library() = delete;

  explicit __shared_library(const char* __lib_path, ::cuda::std::int32_t __flags = __platform_default_flags)
      : __handle_{__load_lib_platform(__lib_path, __flags)}
  {
    if (!__handle_)
    {
      _CCCL_THROW(::std::runtime_error, "Failed to load dynamic shared object");
    }
  }

  /**
   * @return Pointer to the platform-specific library handle.
   */
  [[nodiscard]] typename __platform_handle_t::pointer handle() const
  {
    return __handle_.get();
  }

  template <class _Tp>
  [[nodiscard]] ::cuda::std::decay_t<_Tp> load_symbol(const char* __symbol_name) const
  {
    return reinterpret_cast<::cuda::std::decay_t<_Tp>>(__load_symbol_platform(__symbol_name));
  }

private:
  __platform_handle_t __handle_;
};
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___NCCL_SHARED_LIBRARY_H
