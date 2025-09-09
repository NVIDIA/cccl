//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_PTX_COMPILE_OPTIONS_CUH
#define _CUDAX___COMPILER_PTX_COMPILE_OPTIONS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/arch_traits.h>
#include <cuda/std/__utility/to_underlying.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Option to specify the optimization level for PTX compilation.
enum class ptx_optimization_level
{
  O0,
  O1,
  O2,
  O3,
};

//! @brief Class to hold PTX compilation options.
class ptx_compile_options
{
  friend class cuda_compiler;
  friend class ptx_compiler;

  int __max_reg_count_; //!< The maximum number of registers per thread.
  int __binary_arch_; //!< The binary architecture ID.
  unsigned __optimization_level_ : 2; //!< The optimization level.
  unsigned __device_debug_       : 1; //!< Whether device debugging is enabled.
  unsigned __line_info_          : 1; //!< Whether line information is enabled.
  unsigned __fmad_               : 1; //!< Whether fused multiply-add (FMA) is enabled.
  unsigned __pic_                : 1; //!< Whether position-independent code (PIC) is enabled.

public:
  //! @brief Default constructor for PTX compilation options.
  ptx_compile_options() noexcept
      : __max_reg_count_{-1}
      , __binary_arch_{_CUDA_VSTD::to_underlying(cuda::arch::id::sm_75)}
      , __optimization_level_{_CUDA_VSTD::to_underlying(ptx_optimization_level::O3)}
      , __device_debug_{false}
      , __line_info_{false}
      , __fmad_{false}
      , __pic_{false}
  {}

  //! @brief Enable device debugging information.
  //!
  //! @param __enable If true, enables device debugging information; otherwise, disables it.
  void enable_device_debug(bool __enable = true) noexcept
  {
    __device_debug_ = __enable;
  }

  //! @brief Enable line information.
  //!
  //! @param __enable If true, enables line information; otherwise, disables it.
  void enable_line_info(bool __enable = true) noexcept
  {
    __line_info_ = __enable;
  }

  //! @brief Enable fused multiply-add (FMA) operations.
  //!
  //! @param __enable If true, enables FMA operations; otherwise, disables it.
  void enable_fmad(bool __enable = true) noexcept
  {
    __fmad_ = __enable;
  }

  //! @brief Set the maximum number of registers per thread.
  //!
  //! @param __max_reg_count The maximum number of registers per thread.
  void set_max_reg_count(int __max_reg_count) noexcept
  {
    __max_reg_count_ = __max_reg_count;
  }

  //! @brief Set the optimization level for PTX compilation.
  //!
  //! @param __opt_level The optimization level to use.
  void set_optimization_level(ptx_optimization_level __opt_level) noexcept
  {
    __optimization_level_ = _CUDA_VSTD::to_underlying(__opt_level);
  }

  //! @brief Enable position-independent code (PIC).
  //!
  //! @param __enable If true, enables PIC; otherwise, disables it.
  void enable_pic(bool __enable = true) noexcept
  {
    __pic_ = __enable;
  }

  //! @brief Set the binary architecture for PTX compilation.
  //!
  //! @param __arch_id The binary architecture ID to use. Default is `cuda::arch::id::sm_75`.
  void set_binary_arch(cuda::arch::id __arch_id) noexcept
  {
    __binary_arch_ = _CUDA_VSTD::to_underlying(__arch_id);
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_PTX_COMPILE_OPTIONS_CUH
