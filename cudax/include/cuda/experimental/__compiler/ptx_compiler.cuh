//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_PTX_COMPILER_CUH
#define _CUDAX___COMPILER_PTX_COMPILER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/move.h>
#include <cuda/std/span>
#include <cuda/std/string_view>

#include <cuda/experimental/__compiler/ptx_compile_options.cuh>
#include <cuda/experimental/__compiler/ptx_compile_result.cuh>
#include <cuda/experimental/__compiler/ptx_compile_source.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <nvPTXCompiler.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief A class representing the options for PTX compilation.
struct __nvptx_compile_options
{
  ::std::vector<const char*> __opt_ptrs; //!< The option pointers
  ::std::vector<::std::string> __opt_strs; //!< The option strings
};

//! @brief A class representing a PTX compiler.
class ptx_compiler
{
  unsigned __thread_limit_{1}; //!< The thread limit (0 means no limit)

  //! @brief Creates a nvPTX compiler handle from the PTX source.
  //!
  //! @param __ptx_src The PTX source to compile.
  //!
  //! @return The created nvPTX compiler handle.
  [[nodiscard]] static ::nvPTXCompilerHandle __make_handle(const ptx_compile_source& __ptx_src)
  {
    ::nvPTXCompilerHandle __handle{};
    if (::nvPTXCompilerCreate(&__handle, __ptx_src.__code_.size(), __ptx_src.__code_.data()) != ::NVPTXCOMPILE_SUCCESS)
    {
      // todo: throw an exception if the program creation failed
    }
    return __handle;
  }

  //! @brief Creates the compilation options for the nvPTX compiler for the current compiler state.
  //!
  //! @return The created nvPTX compile options.
  [[nodiscard]] __nvptx_compile_options __make_options() const
  {
    __nvptx_compile_options __ret{};

    // set thread limit
    if (__thread_limit_ != 1)
    {
      __ret.__opt_ptrs.push_back("-split-compile");
      __ret.__opt_ptrs.push_back(__ret.__opt_strs.emplace_back(::std::to_string(__thread_limit_)).c_str());
    }
    return __ret;
  }

  //! @brief Creates the compilation options for the nvPTX compiler for the given PTX options and the current compiler
  //!        state.
  //!
  //! @param __ptx_opts The PTX compilation options to use.
  //!
  //! @return The created nvPTX compile options.
  [[nodiscard]] __nvptx_compile_options __make_options(const ptx_compile_options& __ptx_opts) const
  {
    __nvptx_compile_options __ret = __make_options();

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
    __ret.__opt_ptrs.push_back(__ptx_opts.__fmad_ ? "-fmad=true" : "-fmad=false");

    // max register count
    if (__ptx_opts.__max_reg_count_ >= 0)
    {
      __ret.__opt_ptrs.push_back(__ret.__opt_strs.emplace_back("--maxrregcount").c_str());
      __ret.__opt_strs.back().append(::std::to_string(__ptx_opts.__max_reg_count_));
    }

    // optimization level
    switch (__ptx_opts.__optimization_level_)
    {
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O0):
        __ret.__opt_ptrs.push_back("-O0");
        break;
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O1):
        __ret.__opt_ptrs.push_back("-O1");
        break;
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O2):
        __ret.__opt_ptrs.push_back("-O2");
        break;
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O3):
        __ret.__opt_ptrs.push_back("-O3");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // position independent code flag
    __ret.__opt_ptrs.push_back(__ptx_opts.__pic_ ? "-pic=true" : "-pic=false");

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

  //! @brief Compiles the PTX source code using the nvPTX compiler.
  //!
  //! @param __handle The nvPTX compiler handle.
  //! @param __opt_ptrs The compilation options to use.
  //!
  //! @return `true` if the compilation was successful, `false` otherwise.
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

  //! @brief Set the thread limit for compilation.
  //!
  //! @param __limit The maximum number of threads to use for compilation. 0 means no limit.
  void set_thread_limit(unsigned __limit) noexcept
  {
    __thread_limit_ = __limit;
  }

  //! @brief Compile PTX source code to CUBIN.
  //!
  //! @param __ptx_src The PTX source code to compile.
  //! @param __ptx_opts The PTX compilation options to use.
  //! @param __lowered_names Optional names to lower.
  //!
  //! @return A compile_ptx_to_cubin_result object.
  [[nodiscard]] compile_ptx_to_cubin_result
  compile_to_cubin(const ptx_compile_source& __ptx_src, const ptx_compile_options& __ptx_opts)
  {
    auto __handle = __make_handle(__ptx_src);

    auto [__opt_ptrs, __opt_strings] = __make_options(__ptx_opts);

    __opt_ptrs.push_back("-e");
    ::std::string __tmp{"\""};
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

#endif // _CUDAX___COMPILER_PTX_COMPILER_CUH
