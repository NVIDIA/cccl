//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_COMPILER_CUH
#define _CUDAX___COMPILER_COMPILER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>
#include <cuda/std/string_view>

#include <cuda/experimental/__compiler/compile_options.cuh>
#include <cuda/experimental/__compiler/compile_result.cuh>
#include <cuda/experimental/__compiler/compile_source.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <nvPTXCompiler.h>
#include <nvrtc.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

struct __compile_options
{
  ::std::vector<const char*> __opt_ptrs;
  ::std::vector<::std::string> __opt_strs;
};

/// @brief A class representing a CUDA compiler.
class cuda_compiler
{
  [[nodiscard]] static ::nvrtcProgram __make_program(const cuda_compile_source& __src)
  {
    ::std::string __name{__src.__name_.begin(), __src.__name_.end()};
    ::std::string __code{__src.__code_.begin(), __src.__code_.end()};
    ::nvrtcProgram __program{};
    if (::nvrtcCreateProgram(&__program, __code.c_str(), __name.c_str(), 0, nullptr, nullptr) != ::NVRTC_SUCCESS)
    {
      // todo: throw an exception if the program creation failed
    }
    return __program;
  }

  [[nodiscard]] static const char*
  __make_dyn_opt(cuda_compile_opts::_DynOpt __dopt, ::std::vector<::std::string>& __opt_strs)
  {
    switch (__dopt.__type_)
    {
      case cuda_compile_opts::_DynOptType::__define_macro: {
        auto& __str = __opt_strs.emplace_back("-D");
        __str.append(__dopt.__value_.__string_view2_.__first_.begin(), __dopt.__value_.__string_view2_.__first_.end());
        if (!__dopt.__value_.__string_view2_.__second_.empty())
        {
          __str.append("=");
          __str.append(__dopt.__value_.__string_view2_.__second_.begin(),
                       __dopt.__value_.__string_view2_.__second_.end());
        }
        return __str.c_str();
      }
      case cuda_compile_opts::_DynOptType::__undefine_macro:
        return __opt_strs.emplace_back(_CUDA_VSTD::move("-U"))
          .append(__dopt.__value_.__string_view_.begin(), __dopt.__value_.__string_view_.end())
          .c_str();
      case cuda_compile_opts::_DynOptType::__include_path:
        return __opt_strs.emplace_back(_CUDA_VSTD::move("-I"))
          .append(__dopt.__value_.__string_view_.begin(), __dopt.__value_.__string_view_.end())
          .c_str();
      case cuda_compile_opts::_DynOptType::__force_include:
        return __opt_strs.emplace_back(_CUDA_VSTD::move("-include "))
          .append(__dopt.__value_.__string_view_.begin(), __dopt.__value_.__string_view_.end())
          .c_str();
      default:
        _CCCL_UNREACHABLE();
    }
  }

  [[nodiscard]] static __compile_options __make_options(const cuda_compile_opts& __cuda_opts)
  {
    using namespace cuda_compile_options;

    __compile_options __ret{};

    // disable automatic addition of source's directory to the include path
    __ret.__opt_ptrs.push_back("-no-source-include");

    // C++ standard version
    switch (__cuda_opts.__std_version_)
    {
      case _CUDA_VSTD::to_underlying(std_version_opt::cxx03):
        __ret.__opt_ptrs.push_back("-std=c++03");
        break;
      case _CUDA_VSTD::to_underlying(std_version_opt::cxx11):
        __ret.__opt_ptrs.push_back("-std=c++11");
        break;
      case _CUDA_VSTD::to_underlying(std_version_opt::cxx14):
        __ret.__opt_ptrs.push_back("-std=c++14");
        break;
      case _CUDA_VSTD::to_underlying(std_version_opt::cxx17):
        __ret.__opt_ptrs.push_back("-std=c++17");
        break;
      case _CUDA_VSTD::to_underlying(std_version_opt::cxx20):
        __ret.__opt_ptrs.push_back("-std=c++20");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // virtual architecture
    switch (__cuda_opts.__virtual_arch_)
    {
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_75):
        __ret.__opt_ptrs.push_back("-arch=compute_75");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_80):
        __ret.__opt_ptrs.push_back("-arch=compute_80");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_86):
        __ret.__opt_ptrs.push_back("-arch=compute_86");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_89):
        __ret.__opt_ptrs.push_back("-arch=compute_89");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90):
        __ret.__opt_ptrs.push_back("-arch=compute_90");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100):
        __ret.__opt_ptrs.push_back("-arch=compute_100");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103):
        __ret.__opt_ptrs.push_back("-arch=compute_103");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120):
        __ret.__opt_ptrs.push_back("-arch=compute_120");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90a):
        __ret.__opt_ptrs.push_back("-arch=compute_90a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100a):
        __ret.__opt_ptrs.push_back("-arch=compute_100a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103a):
        __ret.__opt_ptrs.push_back("-arch=compute_103a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120a):
        __ret.__opt_ptrs.push_back("-arch=compute_120a");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // process dynamic options
    for (const auto& __dopt : __cuda_opts.__dyn_opts_)
    {
      const char* __opt_ptr = __make_dyn_opt(__dopt, __ret.__opt_strs);
      if (__opt_ptr != nullptr)
      {
        __ret.__opt_ptrs.push_back(__opt_ptr);
      }
    }

    return __ret;
  }

  [[nodiscard]] static __compile_options
  __make_options(const cuda_compile_opts& __cuda_opts, const ptx_compile_opts& __ptx_opts)
  {
    using namespace ptx_compile_options;

    auto __ret = __make_options(__cuda_opts);

    // device debug flag
    if (__ptx_opts.__device_debug_)
    {
      __ret.__opt_ptrs.push_back("--device-debug");
    }

    // line info flag
    if (__ptx_opts.__line_info_)
    {
      __ret.__opt_ptrs.push_back("-line-info");
    }

    // fmad flag
    __ret.__opt_ptrs.push_back((__ptx_opts.__fmad_) ? "--fmad=true" : "--fmad=false");

    // max register count
    switch (__ptx_opts.__max_reg_count_)
    {
      case _CUDA_VSTD::to_underlying(max_reg_count_opt::__unspecified):
        break;
      case _CUDA_VSTD::to_underlying(max_reg_count_opt::arch_min):
        __ret.__opt_ptrs.push_back("--maxrregcount=archmin");
        break;
      case _CUDA_VSTD::to_underlying(max_reg_count_opt::arch_max):
        __ret.__opt_ptrs.push_back("--maxrregcount=archmax");
        break;
      default: {
        auto& __str = __ret.__opt_strs.emplace_back("--maxrregcount=");
        __str.append(::std::to_string(__ptx_opts.__max_reg_count_));
        __ret.__opt_ptrs.push_back(__str.c_str());
        break;
      }
    }

    // optimization level
    switch (__ptx_opts.__optimization_level_)
    {
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O0):
        __ret.__opt_ptrs.push_back("-Xptxas");
        __ret.__opt_ptrs.push_back("-O0");
        break;
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O1):
        __ret.__opt_ptrs.push_back("-Xptxas");
        __ret.__opt_ptrs.push_back("-O1");
        break;
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O2):
        __ret.__opt_ptrs.push_back("-Xptxas");
        __ret.__opt_ptrs.push_back("-O2");
        break;
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O3):
        __ret.__opt_ptrs.push_back("-Xptxas");
        __ret.__opt_ptrs.push_back("-O3");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // position independent code flag
    __ret.__opt_ptrs.push_back("-Xptxas");
    __ret.__opt_ptrs.push_back(
      __ptx_opts.__pic_ ? "--position-independent-code=true" : "--position-independent-code=false");

    // binary architecture
    switch (__ptx_opts.__binary_arch_)
    {
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_75):
        __ret.__opt_ptrs.push_back("-arch=sm_75");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_80):
        __ret.__opt_ptrs.push_back("-arch=sm_80");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_86):
        __ret.__opt_ptrs.push_back("-arch=sm_86");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_89):
        __ret.__opt_ptrs.push_back("-arch=sm_89");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90):
        __ret.__opt_ptrs.push_back("-arch=sm_90");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100):
        __ret.__opt_ptrs.push_back("-arch=sm_100");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103):
        __ret.__opt_ptrs.push_back("-arch=sm_103");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120):
        __ret.__opt_ptrs.push_back("-arch=sm_120");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90a):
        __ret.__opt_ptrs.push_back("-arch=sm_90a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100a):
        __ret.__opt_ptrs.push_back("-arch=sm_100a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103a):
        __ret.__opt_ptrs.push_back("-arch=sm_103a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120a):
        __ret.__opt_ptrs.push_back("-arch=sm_120a");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    return __ret;
  }

  static void
  __add_name_expressions(::nvrtcProgram __program, _CUDA_VSTD::span<const _CUDA_VSTD::string_view> __name_exprs)
  {
    ::std::string __tmp{};
    for (const auto& __name_expr : __name_exprs)
    {
      __tmp.assign(__name_expr.begin(), __name_expr.end());
      if (::nvrtcAddNameExpression(__program, __tmp.c_str()) != ::NVRTC_SUCCESS)
      {
        // todo: throw an exception if the name expression could not be added
      }
    }
  }

  [[nodiscard]] static bool __compile(::nvrtcProgram __program, _CUDA_VSTD::span<const char*> __opt_ptrs)
  {
    const auto __result = ::nvrtcCompileProgram(__program, static_cast<int>(__opt_ptrs.size()), __opt_ptrs.data());
    return __result == ::NVRTC_SUCCESS;
  }

public:
  //! @brief Get the version of the NVRTC compiler.
  //!
  //! @return The version of the NVRTC compiler as an integer, where the major version is multiplied by 100 and added to
  //!         the minor version.
  [[nodiscard]] static int version()
  {
    int __major{};
    int __minor{};
    if (::nvrtcVersion(&__major, &__minor) != ::NVRTC_SUCCESS)
    {
      // Handle error
    }
    return __major * 100 + __minor;
  }

  //! @brief Compile CUDA source code to PTX.
  //!
  //! @param __cuda_src The CUDA source code to compile.
  //! @param __cuda_opts The CUDA compilation options to use.
  //!
  //! @return A compile_cuda_to_ptx_result object.
  [[nodiscard]] compile_cuda_to_ptx_result
  compile_to_ptx(const cuda_compile_source& __cuda_src, const cuda_compile_opts& __cuda_opts)
  {
    auto __program = __make_program(__cuda_src);
    __add_name_expressions(__program, __cuda_src.__name_exprs_);

    [[maybe_unused]] auto [__opt_ptrs, __opt_strs] = __make_options(__cuda_opts);

    return compile_cuda_to_ptx_result{__program, __compile(__program, __opt_ptrs)};
  }

  //! @brief Compile CUDA source code to CUBIN.
  //!
  //! @param __cuda_src The CUDA source code to compile.
  //! @param __cuda_opts The CUDA compilation options to use.
  //! @param __ptx_opts The PTX compilation options to use.
  //!
  //! @return A compile_cuda_to_cubin_result object.
  [[nodiscard]] compile_cuda_to_cubin_result compile_to_cubin(
    const cuda_compile_source& __cuda_src, const cuda_compile_opts& __cuda_opts, const ptx_compile_opts& __ptx_opts)
  {
    auto __program = __make_program(__cuda_src);
    __add_name_expressions(__program, __cuda_src.__name_exprs_);

    [[maybe_unused]] auto [__opt_ptrs, __opt_strs] = __make_options(__cuda_opts, __ptx_opts);

    return compile_cuda_to_cubin_result{__program, __compile(__program, __opt_ptrs)};
  }

  //! @brief Compile CUDA source code to LTOIR.
  //!
  //! @param __cuda_src The CUDA source code to compile.
  //! @param __cuda_opts The CUDA compilation options to use.
  //!
  //! @return A compile_cuda_to_ltoir_result object.
  [[nodiscard]] compile_cuda_to_ltoir_result
  compile_to_ltoir(const cuda_compile_source& __cuda_src, const cuda_compile_opts& __cuda_opts)
  {
    auto __program = __make_program(__cuda_src);
    __add_name_expressions(__program, __cuda_src.__name_exprs_);

    [[maybe_unused]] auto [__opt_ptrs, __opt_strs] = __make_options(__cuda_opts);
    __opt_ptrs.push_back("-dlto");

    return compile_cuda_to_ltoir_result{__program, __compile(__program, __opt_ptrs)};
  }
};

//! @brief A class representing a PTX compiler.
class ptx_compiler
{
  [[nodiscard]] static ::nvPTXCompilerHandle __make_handle(const ptx_compile_source& __ptx_src)
  {
    ::nvPTXCompilerHandle __handle{};
    if (::nvPTXCompilerCreate(&__handle, __ptx_src.__code_.size(), __ptx_src.__code_.data()) != ::NVPTXCOMPILE_SUCCESS)
    {
      // todo: throw an exception if the program creation failed
    }
    return __handle;
  }

  [[nodiscard]] static __compile_options __make_options(const ptx_compile_opts& __ptx_opts)
  {
    using namespace ptx_compile_options;

    __compile_options __ret{};

    // device debug flag
    if (__ptx_opts.__device_debug_)
    {
      __ret.__opt_ptrs.push_back("--device-debug");
    }

    // line info flag
    if (__ptx_opts.__line_info_)
    {
      __ret.__opt_ptrs.push_back("-line-info");
    }

    // fmad flag
    __ret.__opt_ptrs.push_back(__ptx_opts.__fmad_ ? "--fmad=true" : "--fmad=false");

    // max register count
    switch (__ptx_opts.__max_reg_count_)
    {
      case _CUDA_VSTD::to_underlying(max_reg_count_opt::__unspecified):
        break;
      case _CUDA_VSTD::to_underlying(max_reg_count_opt::arch_min):
        __ret.__opt_ptrs.push_back("--maxrregcount=archmin");
        break;
      case _CUDA_VSTD::to_underlying(max_reg_count_opt::arch_max):
        __ret.__opt_ptrs.push_back("--maxrregcount=archmax");
        break;
      default: {
        auto& __str = __ret.__opt_strs.emplace_back("--maxrregcount=");
        __str.append(::std::to_string(__ptx_opts.__max_reg_count_));
        __ret.__opt_ptrs.push_back(__str.c_str());
        break;
      }
    }

    // optimization level
    switch (__ptx_opts.__optimization_level_)
    {
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O0):
        __ret.__opt_ptrs.push_back("-O0");
        break;
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O1):
        __ret.__opt_ptrs.push_back("-O1");
        break;
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O2):
        __ret.__opt_ptrs.push_back("-O2");
        break;
      case _CUDA_VSTD::to_underlying(optimization_level_opt::O3):
        __ret.__opt_ptrs.push_back("-O3");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // position independent code flag
    __ret.__opt_ptrs.push_back(
      __ptx_opts.__pic_ ? "--position-independent-code=true" : "--position-independent-code=false");

    // binary architecture
    switch (__ptx_opts.__binary_arch_)
    {
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_75):
        __ret.__opt_ptrs.push_back("-arch=sm_75");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_80):
        __ret.__opt_ptrs.push_back("-arch=sm_80");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_86):
        __ret.__opt_ptrs.push_back("-arch=sm_86");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_89):
        __ret.__opt_ptrs.push_back("-arch=sm_89");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90):
        __ret.__opt_ptrs.push_back("-arch=sm_90");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100):
        __ret.__opt_ptrs.push_back("-arch=sm_100");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103):
        __ret.__opt_ptrs.push_back("-arch=sm_103");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120):
        __ret.__opt_ptrs.push_back("-arch=sm_120");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90a):
        __ret.__opt_ptrs.push_back("-arch=sm_90a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100a):
        __ret.__opt_ptrs.push_back("-arch=sm_100a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103a):
        __ret.__opt_ptrs.push_back("-arch=sm_103a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120a):
        __ret.__opt_ptrs.push_back("-arch=sm_120a");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    return __ret;
  }

  [[nodiscard]] static bool __compile(::nvPTXCompilerHandle __handle, _CUDA_VSTD::span<const char*> __opt_ptrs)
  {
    const auto __result = ::nvPTXCompilerCompile(__handle, static_cast<int>(__opt_ptrs.size()), __opt_ptrs.data());
    return __result == ::NVPTXCOMPILE_SUCCESS;
  }

public:
  //! @brief Get the version of the nvPTX compiler.
  //!
  //! @return The version of the nvPTX compiler as an integer, where the major version is multiplied by 100 and added to
  //!         the minor version.
  [[nodiscard]] static int version()
  {
    unsigned __major{};
    unsigned __minor{};
    if (::nvPTXCompilerGetVersion(&__major, &__minor) != ::NVPTXCOMPILE_SUCCESS)
    {
      // Handle error
    }
    return static_cast<int>(__major * 100 + __minor);
  }

  //! @brief Compile PTX source code to CUBIN.
  //!
  //! @param __ptx_src The PTX source code to compile.
  //! @param __ptx_opts The PTX compilation options to use.
  //! @param __lowered_names Optional names to lower.
  //!
  //! @return A compile_ptx_to_cubin_result object.
  [[nodiscard]] compile_ptx_to_cubin_result
  compile_to_cubin(const ptx_compile_source& __ptx_src, const ptx_compile_opts& __ptx_opts)
  {
    auto __handle = __make_handle(__ptx_src);

    auto [__opt_ptrs, __opt_strings] = __make_options(__ptx_opts);

    ::std::string __tmp{"--entry=\""};
    for (_CUDA_VSTD::size_t __i = 0; __i < __ptx_src.__symbols_.size(); ++__i)
    {
      if (__i > 0)
      {
        __tmp.append(",");
      }
      __tmp.append(__ptx_src.__symbols_[__i].begin(), __ptx_src.__symbols_[__i].end());
    }
    __tmp.append("\"");
    __opt_strings.emplace_back(_CUDA_VSTD::move(__tmp));
    __opt_ptrs.push_back(__opt_strings.back().c_str());

    return compile_ptx_to_cubin_result{__handle, __compile(__handle, __opt_ptrs)};
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_COMPILER_CUH
