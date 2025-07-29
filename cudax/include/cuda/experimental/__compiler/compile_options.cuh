//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_COMPILE_OPTIONS_CUH
#define _CUDAX___COMPILER_COMPILE_OPTIONS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/arch_traits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/string_view>

#include <string>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Enum class to specify the C++ standard version for the CUDA compilation.
enum class cuda_std_version
{
  cxx03 = 03, //!< C++03 standard version.
  cxx11 = 11, //!< C++11 standard version.
  cxx14 = 14, //!< C++14 standard version.
  cxx17 = 17, //!< C++17 standard version.
  cxx20 = 20, //!< C++20 standard version.
};

//! @brief Class to hold CUDA compilation options.
class cuda_compile_options
{
  friend class cuda_compiler;

  ::std::vector<::std::string> __dyn_opts_;
  unsigned __std_version_ : 8;
  int __virtual_arch_;

public:
  //! @brief Default constructor for CUDA compilation options.
  cuda_compile_options() noexcept
      : __dyn_opts_{}
      , __std_version_{_CUDA_VSTD::to_underlying(cuda_std_version::cxx17)}
      , __virtual_arch_{_CUDA_VSTD::to_underlying(cuda::arch::id::sm_75)}
  {}

  //! @brief Adds a macro definition to the list of options.
  //!
  //! @param __name The name of the macro to define. If the `__name` is empty, no action is taken.
  void add_macro_definition(_CUDA_VSTD::string_view __name)
  {
    constexpr _CUDA_VSTD::string_view __prefix{"-D"};

    if (__name.empty())
    {
      return;
    }
    ::std::string __str{};
    __str.reserve(__name.size() + __prefix.size());
    __str.append(__prefix.begin(), __prefix.end());
    __str.append(__name.begin(), __name.end());
    __dyn_opts_.push_back(_CUDA_VSTD::move(__str));
  }

  //! @brief Adds a macro definition to the list of options.
  //!
  //! @param __name The name of the macro to define. If the `__name` is empty, no action is taken. If the `__value` is
  //! empty, the macro is defined without a value.
  void add_macro_definition(_CUDA_VSTD::string_view __name, _CUDA_VSTD::string_view __value)
  {
    constexpr _CUDA_VSTD::string_view __prefix{"-D"};

    if (__name.empty())
    {
      return;
    }
    if (__value.empty())
    {
      return add_macro_definition(__name);
    }
    ::std::string __str{};
    __str.reserve(__name.size() + __value.size() + __prefix.size() + 1); // +1 for '='
    __str.append(__prefix.begin(), __prefix.end());
    __str.append(__name.begin(), __name.end());
    __str.append("=");
    __str.append(__value.begin(), __value.end());
    __dyn_opts_.push_back(_CUDA_VSTD::move(__str));
  }

  //! @brief Adds a macro undefinition to the list of options.
  //!
  //! @param __name The name of the macro to undefine. If the `__name` is empty, no action is taken.
  void add_macro_undefinition(_CUDA_VSTD::string_view __name)
  {
    constexpr _CUDA_VSTD::string_view __prefix{"-U"};

    if (__name.empty())
    {
      return;
    }
    ::std::string __str{};
    __str.reserve(__name.size() + __prefix.size());
    __str.append(__prefix.begin(), __prefix.end());
    __str.append(__name.begin(), __name.end());
    __dyn_opts_.push_back(_CUDA_VSTD::move(__str));
  }

  //! @brief Adds an include path to the list of options.
  //!
  //! @param __path The include path to add. If the `__path` is empty, no action is taken.
  void add_include_path(_CUDA_VSTD::string_view __path)
  {
    constexpr _CUDA_VSTD::string_view __prefix{"-I"};

    ::std::string __str{};
    __str.reserve(__path.size() + __prefix.size());
    __str.append(__prefix.begin(), __prefix.end());
    __str.append(__path.begin(), __path.end());
    __dyn_opts_.push_back(_CUDA_VSTD::move(__str));
  }

  //! @brief Adds a file to be force-included in the compilation.
  //!
  //! @param __file_name The file to force include. If the `__file_name` is empty, no action is taken.
  void add_force_include(_CUDA_VSTD::string_view __file_name)
  {
    constexpr _CUDA_VSTD::string_view __prefix{"-include"};

    if (__file_name.empty())
    {
      return;
    }
    __dyn_opts_.push_back(::std::string{__prefix.begin(), __prefix.end()});
    __dyn_opts_.push_back(::std::string{__file_name.begin(), __file_name.end()});
  }

  //! @brief Sets the C++ standard version for the compilation.
  //!
  //! @param __std_version The C++ standard version to use. Default is C++17.
  void set_std_version(cuda_std_version __std_version)
  {
    __std_version_ = _CUDA_VSTD::to_underlying(__std_version);
  }

  //! @brief Sets the virtual architecture ID for the compilation.
  //!
  //! @param __arch_id The virtual architecture ID to use. Default is `cuda::arch::id::sm_75`.
  void set_virtual_arch(cuda::arch::id __arch_id)
  {
    __virtual_arch_ = _CUDA_VSTD::to_underlying(__arch_id);
  }
};

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

  int __max_reg_count_;
  int __binary_arch_;
  unsigned __optimization_level_ : 2;
  unsigned __device_debug_       : 1;
  unsigned __line_info_          : 1;
  unsigned __fmad_               : 1;
  unsigned __pic_                : 1;

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

#endif // _CUDAX___COMPILER_COMPILE_OPTIONS_CUH
