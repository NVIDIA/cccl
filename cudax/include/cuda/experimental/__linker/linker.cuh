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

void __add_link_sources(::nvJitLinkHandle __handle, _CUDA_VSTD::span<const __link_src> __sources)
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

//! @brief A linker for PTX sources.
class ptx_linker
{
public:
  //! @brief Link PTX sources to a single PTX output.
  //!
  //! @param __sources The PTX sources to link.
  [[nodiscard]] link_to_ptx_result link_to_ptx(const ptx_link_sources& __sources, const ptx_link_opts& __ptx_opts)
  {
    ::std::vector<const char*> __opt_ptrs{"-lto", "-ptx"};

    // todo: process options
    (void) __ptx_opts; // suppress unused variable warning

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
class cubin_linker
{
public:
  //! @brief Link CUBIN sources to a single CUBIN output.
  //!
  //! @param __sources The CUBIN sources to link.
  //! @param __ptx_opts The PTX compilation options to use.
  [[nodiscard]] link_to_cubin_result link_to_cubin(
    const cubin_link_sources& __sources, const ptx_compile_opts& __ptx_opts, const cubin_link_opts& __cubin_opts)
  {
    ::std::vector<const char*> __opt_ptrs{};

    // todo: process options
    (void) __ptx_opts; // suppress unused variable warning
    (void) __cubin_opts; // suppress unused variable warning

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
