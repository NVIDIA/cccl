//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILATION_COMPILE_CUH
#define _CUDAX___COMPILATION_COMPILE_CUH

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
#include <cuda/std/span>
#include <cuda/std/string_view>

#include <cuda/experimental/__compilation/compile_options.cuh>
#include <cuda/experimental/__compilation/compile_result.cuh>
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

[[nodiscard]] const char* __make_nvrtc_option(cuda_compile_opts::_Opt __opt, ::std::vector<::std::string>& __opt_strs)
{
  switch (__opt.__type_)
  {
    case cuda_compile_opts::_OptType::__define_macro: {
      auto& __str = __opt_strs.emplace_back("-D");
      __str.append(__opt.__value_.__string_view2_.__first_.begin(), __opt.__value_.__string_view2_.__first_.end());
      if (!__opt.__value_.__string_view2_.__second_.empty())
      {
        __str.append("=");
        __str.append(__opt.__value_.__string_view2_.__second_.begin(), __opt.__value_.__string_view2_.__second_.end());
      }
      return __str.c_str();
    }
    case cuda_compile_opts::_OptType::__undefine_macro:
      return __opt_strs.emplace_back(_CUDA_VSTD::move("-U"))
        .append(__opt.__value_.__string_view_.begin(), __opt.__value_.__string_view_.end())
        .c_str();
    case cuda_compile_opts::_OptType::__include_path:
      return __opt_strs.emplace_back(_CUDA_VSTD::move("-I"))
        .append(__opt.__value_.__string_view_.begin(), __opt.__value_.__string_view_.end())
        .c_str();
    case cuda_compile_opts::_OptType::__force_include:
      return __opt_strs.emplace_back(_CUDA_VSTD::move("-include "))
        .append(__opt.__value_.__string_view_.begin(), __opt.__value_.__string_view_.end())
        .c_str();
    case cuda_compile_opts::_OptType::__std_ver:
      return (__opt.__value_.__int_ == 17) ? "-std=c++17" : "-std=c++20";
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] const char* __make_nvrtc_option(ptx_compile_opts::_Opt __opt, ::std::vector<::std::string>& __opt_strs)
{
  switch (__opt.__type_)
  {
    case ptx_compile_opts::_OptType::__device_debug:
      return __opt.__value_.__boolean_ ? "--device-debug" : nullptr;
    case ptx_compile_opts::_OptType::__line_info:
      return __opt.__value_.__boolean_ ? "-line-info" : nullptr;
    case ptx_compile_opts::_OptType::__fmad:
      return __opt.__value_.__boolean_ ? "--fmad=true" : "--fmad=false";
    case ptx_compile_opts::_OptType::__max_reg_count: {
      auto& __str = __opt_strs.emplace_back("--maxrregcount=");
      __str.append(::std::to_string(__opt.__value_.__int_));
      return __str.c_str();
    }
    case ptx_compile_opts::_OptType::__optimization_level:
      switch (__opt.__value_.__int_)
      {
        case 0:
          return "-Xptxas -O0";
        case 1:
          return "-Xptxas -O1";
        case 2:
          return "-Xptxas -O2";
        case 3:
          return "-Xptxas -O3";
        default:
          _CCCL_UNREACHABLE();
      }
    case ptx_compile_opts::_OptType::__pic:
      return __opt.__value_.__boolean_
             ? "-Xptxas --position-independent-code=true"
             : "-Xptxas --position-independent-code=false";
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] __compile_options
__make_nvrtc_options(const cuda_compile_opts& __cuda_opts, const ptx_compile_opts& __ptx_opts = {})
{
  __compile_options __ret{};
  __ret.__opt_ptrs.push_back("-no-source-include");

  for (const auto& __opt : __cuda_opts.__opts_)
  {
    const char* __opt_ptr = __make_nvrtc_option(__opt, __ret.__opt_strs);
    if (__opt_ptr != nullptr)
    {
      __ret.__opt_ptrs.push_back(__opt_ptr);
    }
  }
  for (const auto& __opt : __ptx_opts.__opts_)
  {
    const char* __opt_ptr = __make_nvrtc_option(__opt, __ret.__opt_strs);
    if (__opt_ptr != nullptr)
    {
      __ret.__opt_ptrs.push_back(__opt_ptr);
    }
  }
  return __ret;
}

[[nodiscard]] const char* __make_nvptx_option(ptx_compile_opts::_Opt __opt, ::std::vector<::std::string>& __opt_strs)
{
  switch (__opt.__type_)
  {
    case ptx_compile_opts::_OptType::__device_debug:
      return __opt.__value_.__boolean_ ? "--device-debug" : nullptr;
    case ptx_compile_opts::_OptType::__line_info:
      return __opt.__value_.__boolean_ ? "-line-info" : nullptr;
    case ptx_compile_opts::_OptType::__fmad:
      return __opt.__value_.__boolean_ ? "--fmad=true" : "--fmad=false";
    case ptx_compile_opts::_OptType::__max_reg_count: {
      auto& __str = __opt_strs.emplace_back("--maxrregcount=");
      __str.append(::std::to_string(__opt.__value_.__int_));
      return __str.c_str();
    }
    case ptx_compile_opts::_OptType::__optimization_level:
      switch (__opt.__value_.__int_)
      {
        case 0:
          return "-O0";
        case 1:
          return "-O1";
        case 2:
          return "-O2";
        case 3:
          return "-O3";
        default:
          _CCCL_UNREACHABLE();
      }
    case ptx_compile_opts::_OptType::__pic:
      return __opt.__value_.__boolean_ ? "--position-independent-code=true" : "--position-independent-code=false";
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] __compile_options __make_nvptx_options(const ptx_compile_opts& __ptx_opts)
{
  __compile_options __ret{};

  for (const auto& __opt : __ptx_opts.__opts_)
  {
    const char* __opt_ptr = __make_nvptx_option(__opt, __ret.__opt_strs);
    if (__opt_ptr != nullptr)
    {
      __ret.__opt_ptrs.push_back(__opt_ptr);
    }
  }
  return __ret;
}

//! @brief A lightweight wrapper for CUDA source code.
struct cuda_source_code
{
  _CUDA_VSTD::string_view name; //!< The name of the CUDA source code.
  _CUDA_VSTD::string_view code; //!< The CUDA source code to compile.
};

//! @brief Compile CUDA source code to PTX.
//!
//! @param __cuda_src_code The CUDA source code to compile.
//! @param __options The compilation options to use.
//! @param __name_exprs Optional name expressions to lower.
//!
//! @return A compile_cuda_to_ptx_result object.
[[nodiscard]] compile_cuda_to_ptx_result compile_to_ptx(
  cuda_source_code __cuda_src_code,
  const cuda_compile_opts& __cuda_opts,
  _CUDA_VSTD::span<const _CUDA_VSTD::string_view> __name_exprs = {})
{
  ::nvrtcProgram __program{};
  ::std::string __tmp{};

  __tmp.assign(__cuda_src_code.name.begin(), __cuda_src_code.name.end());
  if (::nvrtcCreateProgram(&__program, __cuda_src_code.code.data(), __tmp.c_str(), 0, nullptr, nullptr)
      != ::NVRTC_SUCCESS)
  {
    // todo: throw an exception if the program creation failed
  }

  for (auto __name_expr : __name_exprs)
  {
    __tmp.assign(__name_expr.begin(), __name_expr.end());
    if (::nvrtcAddNameExpression(__program, __tmp.c_str()) != ::NVRTC_SUCCESS)
    {
      // todo: throw an exception if the name expression could not be added
    }
  }

  [[maybe_unused]] auto [__opt_ptrs, __opt_strs] = __make_nvrtc_options(__cuda_opts);

  const auto __result = ::nvrtcCompileProgram(__program, static_cast<int>(__opt_ptrs.size()), __opt_ptrs.data());
  if (__result != ::NVRTC_SUCCESS && __result != ::NVRTC_ERROR_COMPILATION)
  {
    // todo: throw an exception if the compilation failed due to other reasons than compilation errors
  }
  return compile_cuda_to_ptx_result{__program, __result == ::NVRTC_SUCCESS};
}

//! @brief Compile CUDA source code to CUBIN.
//!
//! @param __cuda_src_code The CUDA source code to compile.
//! @param __cuda_opts The CUDA compilation options to use.
//! @param __ptx_opts The PTX compilation options to use.
//! @param __name_exprs Optional name expressions to lower.
//!
//! @return A compile_cuda_to_cubin_result object.
[[nodiscard]] compile_cuda_to_cubin_result compile_to_cubin(
  cuda_source_code __cuda_src_code,
  const cuda_compile_opts& __cuda_opts,
  const ptx_compile_opts& __ptx_opts,
  _CUDA_VSTD::span<const _CUDA_VSTD::string_view> __name_exprs = {})
{
  ::nvrtcProgram __program{};
  if (::nvrtcCreateProgram(&__program, __cuda_src_code.code.data(), "cuda_source.cu", 0, nullptr, nullptr)
      != ::NVRTC_SUCCESS)
  {
    // todo: throw an exception if the program creation failed
  }

  ::std::string __tmp{};
  for (auto __name_expr : __name_exprs)
  {
    __tmp.assign(__name_expr.begin(), __name_expr.end());
    if (::nvrtcAddNameExpression(__program, __tmp.c_str()) != ::NVRTC_SUCCESS)
    {
      // todo: throw an exception if the name expression could not be added
    }
  }

  [[maybe_unused]] auto [__opt_ptrs, __opt_strs] = __make_nvrtc_options(__cuda_opts, __ptx_opts);

  const auto __result = ::nvrtcCompileProgram(__program, static_cast<int>(__opt_ptrs.size()), __opt_ptrs.data());
  if (__result != ::NVRTC_SUCCESS && __result != ::NVRTC_ERROR_COMPILATION)
  {
    // todo: throw an exception if the compilation failed due to other reasons than compilation errors
  }
  return compile_cuda_to_cubin_result{__program, __result == ::NVRTC_SUCCESS};
}

//! @brief Compile CUDA source code to LTOIR
//!
//! @param __cuda_src_code The CUDA source code to compile.
//! @param __options The compilation options to use.
//! @param __name_exprs Optional name expressions to lower.
//!
//! @return A compile_cuda_to_ltoir_result object.
[[nodiscard]] compile_cuda_to_ltoir_result compile_to_ltoir(
  cuda_source_code __cuda_src_code,
  const cuda_compile_opts& __cuda_opts,
  _CUDA_VSTD::span<const _CUDA_VSTD::string_view> __name_exprs = {})
{
  ::nvrtcProgram __program{};
  ::std::string __tmp{};

  __tmp.assign(__cuda_src_code.name.begin(), __cuda_src_code.name.end());
  if (::nvrtcCreateProgram(&__program, __cuda_src_code.code.data(), __tmp.c_str(), 0, nullptr, nullptr)
      != ::NVRTC_SUCCESS)
  {
    // todo: throw an exception if the program creation failed
  }

  for (auto __name_expr : __name_exprs)
  {
    __tmp.assign(__name_expr.begin(), __name_expr.end());
    if (::nvrtcAddNameExpression(__program, __tmp.c_str()) != ::NVRTC_SUCCESS)
    {
      // todo: throw an exception if the name expression could not be added
    }
  }

  [[maybe_unused]] auto [__opt_ptrs, __opt_strs] = __make_nvrtc_options(__cuda_opts);
  __opt_ptrs.push_back("-dlto");

  const auto __result = ::nvrtcCompileProgram(__program, static_cast<int>(__opt_ptrs.size()), __opt_ptrs.data());
  if (__result != ::NVRTC_SUCCESS && __result != ::NVRTC_ERROR_COMPILATION)
  {
    // todo: throw an exception if the compilation failed due to other reasons than compilation errors
  }
  return compile_cuda_to_ltoir_result{__program, __result == ::NVRTC_SUCCESS};
}

//! @brief A lightweight wrapper for PTX source code.
struct ptx_source_code
{
  _CUDA_VSTD::string_view code; //!< The PTX source code to compile.

  //! @brief Constructor to create a ptx_source_code object.
  //!
  //! @param __code The PTX source code to compile.
  constexpr explicit ptx_source_code(_CUDA_VSTD::string_view __code) noexcept
      : code{__code}
  {}
};

//! @brief Compile PTX source code to CUBIN.
//!
//! @param __ptx_src_code The PTX source code to compile.
//! @param __ptx_opts The PTX compilation options to use.
//! @param __lowered_names Optional names to lower.
//!
//! @return A compile_ptx_to_cubin_result object.
[[nodiscard]] compile_ptx_to_cubin_result compile_to_cubin(
  ptx_source_code __ptx_src_code,
  const ptx_compile_opts& __ptx_opts,
  _CUDA_VSTD::span<const _CUDA_VSTD::string_view> __lowered_names = {})
{
  ::nvPTXCompilerHandle __handle{};
  if (::nvPTXCompilerCreate(&__handle, __ptx_src_code.code.size(), __ptx_src_code.code.data())
      != ::NVPTXCOMPILE_SUCCESS)
  {
    // todo: throw an exception if the program creation failed
  }

  auto [__opt_ptrs, __opt_strings] = __make_nvptx_options(__ptx_opts);

  ::std::string __tmp{"--entry=\""};
  for (_CUDA_VSTD::size_t __i = 0; __i < __lowered_names.size(); ++__i)
  {
    if (__i > 0)
    {
      __tmp.append(",");
    }
    __tmp.append(__lowered_names[__i].begin(), __lowered_names[__i].end());
  }
  __tmp.append("\"");
  __opt_strings.emplace_back(_CUDA_VSTD::move(__tmp));
  __opt_ptrs.push_back(__opt_strings.back().c_str());

  const auto __result = ::nvPTXCompilerCompile(__handle, static_cast<int>(__opt_ptrs.size()), __opt_ptrs.data());
  if (__result != ::NVPTXCOMPILE_SUCCESS && __result != ::NVPTXCOMPILE_ERROR_COMPILATION_FAILURE)
  {
    // todo: throw an exception if the compilation failed due to other reasons than compilation errors
  }
  return compile_ptx_to_cubin_result{__handle, __result == ::NVPTXCOMPILE_SUCCESS};
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILATION_COMPILE_CUH
