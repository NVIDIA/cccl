//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LINKER_LINK_SOURCES_CUH
#define _CUDAX___LINKER_LINK_SOURCES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/span>
#include <cuda/std/string_view>

#include <cuda/experimental/__linker/nvjitlink.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

struct __link_src
{
  ::nvJitLinkInputType __type_;
  _CUDA_VSTD::string_view __name_;
  _CUDA_VSTD::span<const _CUDA_VSTD_NOVERSION::byte> __data_;
};

struct __link_symbol
{
  enum class __type
  {
    __kernel,
    __var,
  };
  __type __type_;
  _CUDA_VSTD::string_view __name_;
};

//! @brief A collection of PTX link sources.
class ptx_link_sources
{
  friend class ptx_linker;

  ::std::vector<__link_src> __sources_;
  ::std::vector<__link_symbol> __symbols_;

public:
  //! @brief Adds a PTX source to the collection.
  //!
  //! @param __name The name of the PTX source.
  //! @param __data The PTX source data.
  void add_ptx(_CUDA_VSTD::string_view __name, _CUDA_VSTD::string_view __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_PTX,
                          __name,
                          {reinterpret_cast<const _CUDA_VSTD_NOVERSION::byte*>(__data.data()), __data.size()}});
  }

  //! @brief Adds a CUBIN source to the collection.
  //!
  //! @param __name The name of the CUBIN source.
  //! @param __data The CUBIN source data.
  void add_ltoir(_CUDA_VSTD::string_view __name, _CUDA_VSTD::span<const _CUDA_VSTD_NOVERSION::byte> __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_LTOIR, __name, __data});
  }

  //! @brief Adds a CUBIN source to the collection.
  //!
  //! @param __name The name of the CUBIN source.
  //! @param __data The CUBIN source data.
  void add_fatbin(_CUDA_VSTD::string_view __name, _CUDA_VSTD::span<const _CUDA_VSTD_NOVERSION::byte> __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_FATBIN, __name, __data});
  }

  //! @brief Adds a kernel symbol to be kept in the PTX source code.
  //!
  //! @param __name The name of the kernel symbol.
  void add_kernel_symbol(_CUDA_VSTD::string_view __name)
  {
    __symbols_.push_back({__link_symbol::__type::__kernel, __name});
  }

  //! @brief Adds a variable symbol to be kept in the PTX source code.
  //!
  //! @param __name The name of the variable symbol.
  void add_variable_symbol(_CUDA_VSTD::string_view __name)
  {
    __symbols_.push_back({__link_symbol::__type::__var, __name});
  }
};

//! @brief A collection of cubin link sources.
class cubin_link_sources
{
  friend class cubin_linker;

  ::std::vector<__link_src> __sources_;
  ::std::vector<__link_symbol> __symbols_;

public:
  //! @brief Adds a PTX source to the collection.
  //!
  //! @param __name The name of the PTX source.
  //! @param __data The PTX source data.
  void add_ptx(_CUDA_VSTD::string_view __name, _CUDA_VSTD::string_view __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_PTX,
                          __name,
                          {reinterpret_cast<const _CUDA_VSTD_NOVERSION::byte*>(__data.data()), __data.size()}});
  }

  //! @brief Adds a LTOIR source to the collection.
  //!
  //! @param __name The name of the LTOIR source.
  //! @param __data The LTOIR source data.
  void add_ltoir(_CUDA_VSTD::string_view __name, _CUDA_VSTD::span<const _CUDA_VSTD_NOVERSION::byte> __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_LTOIR, __name, __data});
  }

  //! @brief Adds a CUBIN source to the collection.
  //!
  //! @param __name The name of the CUBIN source.
  //! @param __data The CUBIN source data.
  void add_cubin(_CUDA_VSTD::string_view __name, _CUDA_VSTD::span<const _CUDA_VSTD_NOVERSION::byte> __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_CUBIN, __name, __data});
  }

  //! @brief Adds a FATBIN source to the collection.
  //!
  //! @param __name The name of the FATBIN source.
  //! @param __data The FATBIN source data.
  void add_fatbin(_CUDA_VSTD::string_view __name, _CUDA_VSTD::span<const _CUDA_VSTD_NOVERSION::byte> __data)
  {
    __sources_.push_back({::NVJITLINK_INPUT_FATBIN, __name, __data});
  }

  //! @brief Adds a kernel symbol to be kept in the PTX source code.
  //!
  //! @param __name The name of the kernel symbol.
  void add_kernel_symbol(_CUDA_VSTD::string_view __name)
  {
    __symbols_.push_back({__link_symbol::__type::__kernel, __name});
  }

  //! @brief Adds a variable symbol to be kept in the PTX source code.
  //!
  //! @param __name The name of the variable symbol.
  void add_variable_symbol(_CUDA_VSTD::string_view __name)
  {
    __symbols_.push_back({__link_symbol::__type::__var, __name});
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LINKER_LINK_SOURCES_CUH
