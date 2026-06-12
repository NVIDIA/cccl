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
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/cstdint>

#if _CCCL_OS(WINDOWS)
#  include <windows.h>
#else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
#  include <dlfcn.h>
#endif // ^^^ !_CCCL_OS(WINDOWS) ^^^

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class __shared_library
{
#  if _CCCL_OS(WINDOWS)
  static constexpr ::cuda::std::int32_t __platform_default_flags = LOAD_LIBRARY_SEARCH_SYSTEM32;

  struct __platform_deleter
  {
    using pointer _CCCL_NODEBUG_ALIAS = HMODULE;

    _CCCL_HOST_API void operator()(HMODULE __mod) const noexcept
    {
      static_cast<void>(::FreeLibrary(__mod));
    }
  };

  using __platform_handle_t _CCCL_NODEBUG_ALIAS =
    ::std::unique_ptr<::cuda::std::remove_pointer_t<HMODULE>, __platform_deleter>;

  [[nodiscard]] _CCCL_HOST_API static __platform_handle_t
  __load_lib_platform(const char* const __lib_name, const ::cuda::std::int32_t __flags)
  {
    return __platform_handle_t{::LoadLibraryExA(__lib_name, /*hFile=*/nullptr, static_cast<DWORD>(__flags))};
  }

  [[nodiscard]] _CCCL_HOST_API void* __load_symbol_platform(const char* const __sym_name, bool __can_fail) const
  {
    void* const __sym = ::GetProcAddress(handle(), __sym_name);

    if (__sym == nullptr && !__can_fail)
    {
      _CCCL_THROW(::std::invalid_argument, "Failed to locate the symbol in the shared library");
    }
    return __sym;
  }
#  else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
  static constexpr ::cuda::std::int32_t __platform_default_flags = RTLD_LAZY | RTLD_LOCAL;

  struct __platform_deleter
  {
    _CCCL_HOST_API void operator()(void* const __mod) const noexcept
    {
      static_cast<void>(::dlclose(__mod));
    }
  };

  using __platform_handle_t _CCCL_NODEBUG_ALIAS = ::std::unique_ptr<void, __platform_deleter>;

  [[nodiscard]] _CCCL_HOST_API static __platform_handle_t
  __load_lib_platform(const char* const __lib_name, const ::cuda::std::int32_t __flags)
  {
    static_cast<void>(::dlerror());

    return __platform_handle_t{::dlopen(__lib_name, __flags)};
  }

  [[nodiscard]] _CCCL_HOST_API void* __load_symbol_platform(const char* const __sym_name, bool __can_fail) const
  {
    static_cast<void>(::dlerror());

    auto* __sym = ::dlsym(handle(), __sym_name);

    if (const char* const __error = ::dlerror(); __error || !__sym)
    {
      if (__can_fail)
      {
        // Ensure it is null
        __sym = nullptr;
      }
      else
      {
        _CCCL_THROW(::std::runtime_error, "Failed to locate the symbol in the shared library");
      }
    }
    return __sym;
  }
#  endif // ^^^ !_CCCL_OS(WINDOWS) ^^^

public:
  using native_handle_type = typename __platform_handle_t::pointer;

  __shared_library() = delete;

  /**
   * @brief Construct and load a named shared library with the provided flags
   *
   * @param __lib_path The path used to find the shared library object. If absolute, then the
   * specified path will be attempted to be loaded. If relative, or without path prefixes, then
   * loading is platform dependent.
   *
   * @param __flags Platform dependent flags to pass to the module loading functions. At
   * present should probably be left unspecified.
   *
   * @throw std::runtime_error if the library could not be loaded.
   */
  _CCCL_HOST_API explicit __shared_library(const char* __lib_path,
                                           ::cuda::std::int32_t __flags = __platform_default_flags)
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
  [[nodiscard]] _CCCL_HOST_API native_handle_type handle() const
  {
    return __handle_.get();
  }

  /**
   * @brief Load a named symbol from the library.
   *
   * @tparam _Tp The type of the symbol to load.
   *
   * @param __symbol_name The name of the symbol to load.
   * @param __can_fail `true` if the function is allowed to fail, false otherwise.
   *
   * @return A pointer to the loaded symbol casted to `_Tp`.
   */
  template <class _Tp>
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::decay_t<_Tp>
  load_symbol(const char* __symbol_name, bool __can_fail = false) const
  {
    return reinterpret_cast<::cuda::std::decay_t<_Tp>>(__load_symbol_platform(__symbol_name, __can_fail));
  }

private:
  __platform_handle_t __handle_;
};
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___NCCL_SHARED_LIBRARY_H
