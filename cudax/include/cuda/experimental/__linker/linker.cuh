//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LINKER_LINKER_CUH
#define _CUDAX___LINKER_LINKER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/string_view>

#include <cuda/experimental/__compiler/compile_options.cuh>
#include <cuda/experimental/__linker/link_options.cuh>
#include <cuda/experimental/__linker/link_result.cuh>
#include <cuda/experimental/__linker/link_sources.cuh>
#include <cuda/experimental/__linker/nvjitlink.cuh>

#include <string>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

class __linker_base
{
  bool __enable_internal_cache_{true};
  unsigned __thread_limit_{1};

protected:
  static void __add_link_sources(::nvJitLinkHandle __handle, _CUDA_VSTD::span<const __link_src> __sources)
  {
    for (const auto& __source : __sources)
    {
      ::std::string __name(__source.__name_.begin(), __source.__name_.end());
      if (::nvJitLinkAddData(__handle,
                             __source.__type_,
                             __name.c_str(),
                             __source.__data_.size(),
                             reinterpret_cast<const char*>(__source.__data_.data()))
          != ::NVJITLINK_SUCCESS)
      {
        // todo: throw
      }
    }
  }

public:
  //! @brief Get the version of the nvJitLink linker.
  //!
  //! @return The version of the nvJitLink linker as an integer, where the major version is multiplied by 100 and added
  //!         to the minor version.
  [[nodiscard]] static int version()
  {
    unsigned __major{};
    unsigned __minor{};
    if (::nvJitLinkVersion(&__major, &__minor) != ::NVJITLINK_SUCCESS)
    {
      // Handle error
    }
    return static_cast<int>(__major * 100 + __minor);
  }

  //! @brief Set whether to enable the internal cache.
  //!
  //! @param __enable If `true`, the internal cache is enabled; otherwise, it is disabled.
  void enable_internal_cache(bool __enable = true) noexcept
  {
    __enable_internal_cache_ = __enable;
  }

  //! @brief Set the thread limit for compilation.
  //!
  //! @param __limit The maximum number of threads to use for compilation. 0 means no limit.
  void set_thread_limit(unsigned __limit) noexcept
  {
    __thread_limit_ = __limit;
  }
};

//! @brief A linker for PTX sources.
class ptx_linker : public __linker_base
{
public:
  //! @brief Link PTX sources to a single PTX output.
  //!
  //! @param __sources The PTX sources to link.
  [[nodiscard]] link_to_ptx_result link_to_ptx(const ptx_link_sources& __sources, const ptx_link_options& __ptx_opts)
  {
    ::std::vector<const char*> __opt_ptrs{"-lto", "-ptx"};

    // todo: process options
    (void) __ptx_opts; // suppress unused variable warning

    // todo: process symbols

    ::nvJitLinkHandle __handle{};
    if (::nvJitLinkCreate(&__handle, static_cast<_CUDA_VSTD::uint32_t>(__opt_ptrs.size()), __opt_ptrs.data())
        != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    __add_link_sources(__handle, __sources.__sources_);
    return link_to_ptx_result{__handle, ::nvJitLinkComplete(__handle) == ::NVJITLINK_SUCCESS};
  }
};

//! @brief A linker for CUBIN sources.
class cubin_linker : public __linker_base
{
public:
  //! @brief Link CUBIN sources to a single CUBIN output.
  //!
  //! @param __sources The CUBIN sources to link.
  //! @param __ptx_opts The PTX compilation options to use.
  [[nodiscard]] link_to_cubin_result link_to_cubin(
    const cubin_link_sources& __sources, const ptx_compile_options& __ptx_opts, const cubin_link_options& __cubin_opts)
  {
    ::std::vector<const char*> __opt_ptrs{};

    // todo: process options
    (void) __ptx_opts; // suppress unused variable warning
    (void) __cubin_opts; // suppress unused variable warning

    // todo: process symbols

    ::nvJitLinkHandle __handle{};
    if (::nvJitLinkCreate(&__handle, static_cast<_CUDA_VSTD::uint32_t>(__opt_ptrs.size()), __opt_ptrs.data())
        != ::NVJITLINK_SUCCESS)
    {
      // todo: throw
    }
    __add_link_sources(__handle, __sources.__sources_);
    return link_to_cubin_result{__handle, ::nvJitLinkComplete(__handle) == ::NVJITLINK_SUCCESS};
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LINKER_LINKER_CUH
