//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_CUDA_COMPILE_OPTIONS_CUH
#define _CUDAX___COMPILER_CUDA_COMPILE_OPTIONS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/arch_traits.h>
#include <cuda/std/__utility/move.h>
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

  enum class _DynOptKind
  {
    __macro_def,
    __macro_undef,
    __include_path,
    __force_include,
  };

  struct _DynOpt
  {
    _DynOptKind __kind_;
    ::std::string __value_;
  };

  ::std::vector<_DynOpt> __dyn_opts_; //!< Dynamic compilation options.
  unsigned __std_version_ : 8; //!< C++ standard version.
  int __virtual_arch_; //!< Virtual architecture.
  ::std::string __pch_file_name_; //!< Name of the generated precompiled header file.
  ::std::string __pch_dir_; //!< Directory for precompiled headers.

public:
  //! @brief Default constructor for CUDA compilation options.
  cuda_compile_options() noexcept
      : __dyn_opts_{}
      , __std_version_{_CUDA_VSTD::to_underlying(cuda_std_version::cxx17)}
      , __virtual_arch_{_CUDA_VSTD::to_underlying(cuda::arch::id::sm_75)}
      , __pch_file_name_{}
      , __pch_dir_{}
  {}

  //! @brief Adds a macro definition to the list of options.
  //!
  //! @param __name The name of the macro to define. If the `__name` is empty, no action is taken.
  void add_macro_definition(::std::string __name)
  {
    if (__name.empty())
    {
      return;
    }
    __dyn_opts_.push_back({_DynOptKind::__macro_def, _CUDA_VSTD::move(__name)});
  }

  //! @brief Adds a macro definition to the list of options.
  //!
  //! @param __name The name of the macro to define. If the `__name` is empty, no action is taken. If the `__value` is
  //! empty, the macro is defined without a value.
  void add_macro_definition(::std::string __name, _CUDA_VSTD::string_view __value)
  {
    if (__name.empty())
    {
      return;
    }
    if (__value.empty())
    {
      return add_macro_definition(__name);
    }
    __name.append("=");
    __name.append(__value.begin(), __value.end());
    __dyn_opts_.push_back({_DynOptKind::__macro_def, _CUDA_VSTD::move(__name)});
  }

  //! @brief Adds a macro undefinition to the list of options.
  //!
  //! @param __name The name of the macro to undefine. If the `__name` is empty, no action is taken.
  void add_macro_undefinition(::std::string __name)
  {
    if (__name.empty())
    {
      return;
    }
    __dyn_opts_.push_back({_DynOptKind::__macro_undef, _CUDA_VSTD::move(__name)});
  }

  //! @brief Adds an include path to the list of options.
  //!
  //! @param __path The include path to add. If the `__path` is empty, no action is taken.
  void add_include_path(::std::string __path)
  {
    if (__path.empty())
    {
      return;
    }
    __dyn_opts_.push_back({_DynOptKind::__include_path, _CUDA_VSTD::move(__path)});
  }

  //! @brief Adds a file to be force-included in the compilation.
  //!
  //! @param __file_name The file to force include. If the `__file_name` is empty, no action is taken.
  void add_force_include(::std::string __file_name)
  {
    if (__file_name.empty())
    {
      return;
    }
    __dyn_opts_.push_back({_DynOptKind::__force_include, _CUDA_VSTD::move(__file_name)});
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

  //! @brief Sets the precompiled header output file.
  //!
  //! @param __file_name The name of the precompiled header file.
  void set_pch_output_file(::std::string __file_name)
  {
    if (__file_name.empty())
    {
      return;
    }
    __pch_file_name_ = _CUDA_VSTD::move(__file_name);
  }

  //! @brief Sets the precompiled headers directory.
  //!
  //! @param __dir_name The name of the precompiled headers directory.
  void set_pch_dir(::std::string __dir_name)
  {
    if (__dir_name.empty())
    {
      return;
    }
    __pch_dir_ = _CUDA_VSTD::move(__dir_name);
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_CUDA_COMPILE_OPTIONS_CUH
