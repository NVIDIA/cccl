//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILER_CUDA_COMPILER_CUH
#define _CUDAX___COMPILER_CUDA_COMPILER_CUH

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
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/span>
#include <cuda/std/string_view>

#include <cuda/experimental/__compiler/cuda_compile_options.cuh>
#include <cuda/experimental/__compiler/cuda_compile_result.cuh>
#include <cuda/experimental/__compiler/cuda_compile_source.cuh>
#include <cuda/experimental/__compiler/ptx_compile_options.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <string>
#include <vector>

#include <nvrtc.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Helper structure for nvrtc options
struct __nvrtc_compile_options
{
  ::std::vector<const char*> __ptrs; //!< The option pointers
  ::std::vector<::std::string> __strs; //!< The option strings
};

//! @brief A class representing a CUDA compiler.
class cuda_compiler
{
  bool __enable_internal_cache_{true}; //!< Enable internal cache
  unsigned __thread_limit_{1}; //!< The thread limit (0 means no limit)
  bool __pch_auto_{false}; //!< Enable automatic precompiled headers
  ::std::string __pch_dir_{}; //!< The directory for precompiled headers

  //! @brief Create a NVRTC program from the CUDA source code.
  //!
  //! @param __src The CUDA source code to compile.
  //!
  //! @return The created NVRTC program.
  [[nodiscard]] static ::nvrtcProgram __make_program(const cuda_compile_source& __src)
  {
    ::nvrtcProgram __program{};
    if (::nvrtcCreateProgram(&__program, __src.__code_.c_str(), __src.__name_.c_str(), 0, nullptr, nullptr)
        != ::NVRTC_SUCCESS)
    {
      // todo: throw an exception if the program creation failed
    }
    return __program;
  }

  //! @brief Create NVRTC compile options from the compiler settings.
  //!
  //! @return The created NVRTC compile options.
  [[nodiscard]] __nvrtc_compile_options __make_options() const
  {
    __nvrtc_compile_options __ret{};

    // enable internal cache
    if (!__enable_internal_cache_)
    {
      __ret.__ptrs.push_back("-no-cache");
    }

    // set thread limit
    if (__thread_limit_ != 1)
    {
      __ret.__ptrs.push_back("-split-compile");
      __ret.__ptrs.push_back(nullptr); // placeholder for string pointer
      __ret.__strs.emplace_back(::std::to_string(__thread_limit_));
    }

    // enable auto PCH
    if (__pch_auto_)
    {
      __ret.__ptrs.push_back("-pch");
    }

    // PCH directory
    if (!__pch_dir_.empty())
    {
      __ret.__ptrs.push_back("-pch-dir");
      __ret.__ptrs.push_back(__pch_dir_.c_str());
    }

    return __ret;
  }

  //! @brief Create NVRTC compile options from the compiler settings and the cuda compile options.
  //!
  //! @param __cuda_opts The CUDA compile options.
  //!
  //! @return The created NVRTC compile options.
  [[nodiscard]] __nvrtc_compile_options __make_options(const cuda_compile_options& __cuda_opts) const
  {
    __nvrtc_compile_options __ret = __make_options();

    // disable automatic addition of source's directory to the include path
    __ret.__ptrs.push_back("-no-source-include");

    // C++ standard version
    switch (__cuda_opts.__std_version_)
    {
      case _CUDA_VSTD::to_underlying(cuda_std_version::cxx03):
        __ret.__ptrs.push_back("-std=c++03");
        break;
      case _CUDA_VSTD::to_underlying(cuda_std_version::cxx11):
        __ret.__ptrs.push_back("-std=c++11");
        break;
      case _CUDA_VSTD::to_underlying(cuda_std_version::cxx14):
        __ret.__ptrs.push_back("-std=c++14");
        break;
      case _CUDA_VSTD::to_underlying(cuda_std_version::cxx17):
        __ret.__ptrs.push_back("-std=c++17");
        break;
      case _CUDA_VSTD::to_underlying(cuda_std_version::cxx20):
        __ret.__ptrs.push_back("-std=c++20");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // virtual architecture
    switch (__cuda_opts.__virtual_arch_)
    {
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_75):
        __ret.__ptrs.push_back("-arch=compute_75");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_80):
        __ret.__ptrs.push_back("-arch=compute_80");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_86):
        __ret.__ptrs.push_back("-arch=compute_86");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_89):
        __ret.__ptrs.push_back("-arch=compute_89");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90):
        __ret.__ptrs.push_back("-arch=compute_90");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100):
        __ret.__ptrs.push_back("-arch=compute_100");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103):
        __ret.__ptrs.push_back("-arch=compute_103");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120):
        __ret.__ptrs.push_back("-arch=compute_120");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90a):
        __ret.__ptrs.push_back("-arch=compute_90a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100a):
        __ret.__ptrs.push_back("-arch=compute_100a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103a):
        __ret.__ptrs.push_back("-arch=compute_103a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120a):
        __ret.__ptrs.push_back("-arch=compute_120a");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // process dynamic options
    for (const auto& __dopt : __cuda_opts.__dyn_opts_)
    {
      switch (__dopt.__kind_)
      {
        case cuda_compile_options::_DynOptKind::__macro_def:
          __ret.__ptrs.push_back("-D");
          break;
        case cuda_compile_options::_DynOptKind::__macro_undef:
          __ret.__ptrs.push_back("-U");
          break;
        case cuda_compile_options::_DynOptKind::__include_path:
          __ret.__ptrs.push_back("-I");
          break;
        case cuda_compile_options::_DynOptKind::__force_include:
          __ret.__ptrs.push_back("-include");
          break;
      }
      __ret.__ptrs.push_back(__dopt.__value_.c_str());
    }

    return __ret;
  }

  //! @brief Create NVRTC compile options from the compiler settings, the cuda compile options, and the PTX compile
  //!        options.
  //!
  //! @param __cuda_opts The CUDA compile options.
  //! @param __ptx_opts The PTX compile options.
  //!
  //! @return The created NVRTC compile options.
  [[nodiscard]] __nvrtc_compile_options
  __make_options(const cuda_compile_options& __cuda_opts, const ptx_compile_options& __ptx_opts) const
  {
    auto __ret = __make_options(__cuda_opts);

    // device debug flag
    if (__ptx_opts.__device_debug_)
    {
      __ret.__ptrs.push_back("--device-debug");
    }

    // line info flag
    if (__ptx_opts.__line_info_)
    {
      __ret.__ptrs.push_back("-line-info");
    }

    // fmad flag
    __ret.__ptrs.push_back((__ptx_opts.__fmad_) ? "--fmad=true" : "--fmad=false");

    // max register count
    if (__ptx_opts.__max_reg_count_ >= 0)
    {
      __ret.__ptrs.push_back("--maxrregcount");
      __ret.__ptrs.push_back(nullptr); // placeholder for string pointer
      __ret.__strs.push_back(::std::to_string(__ptx_opts.__max_reg_count_));
    }

    // optimization level
    switch (__ptx_opts.__optimization_level_)
    {
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O0):
        __ret.__ptrs.push_back("-Xptxas");
        __ret.__ptrs.push_back("-O0");
        break;
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O1):
        __ret.__ptrs.push_back("-Xptxas");
        __ret.__ptrs.push_back("-O1");
        break;
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O2):
        __ret.__ptrs.push_back("-Xptxas");
        __ret.__ptrs.push_back("-O2");
        break;
      case _CUDA_VSTD::to_underlying(ptx_optimization_level::O3):
        __ret.__ptrs.push_back("-Xptxas");
        __ret.__ptrs.push_back("-O3");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    // position independent code flag
    __ret.__ptrs.push_back("-Xptxas");
    __ret.__ptrs.push_back(__ptx_opts.__pic_ ? "-pic=true" : "-pic=false");

    // binary architecture
    switch (__ptx_opts.__binary_arch_)
    {
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_75):
        __ret.__ptrs.push_back("-arch=sm_75");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_80):
        __ret.__ptrs.push_back("-arch=sm_80");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_86):
        __ret.__ptrs.push_back("-arch=sm_86");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_89):
        __ret.__ptrs.push_back("-arch=sm_89");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90):
        __ret.__ptrs.push_back("-arch=sm_90");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100):
        __ret.__ptrs.push_back("-arch=sm_100");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103):
        __ret.__ptrs.push_back("-arch=sm_103");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120):
        __ret.__ptrs.push_back("-arch=sm_120");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_90a):
        __ret.__ptrs.push_back("-arch=sm_90a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_100a):
        __ret.__ptrs.push_back("-arch=sm_100a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_103a):
        __ret.__ptrs.push_back("-arch=sm_103a");
        break;
      case _CUDA_VSTD::to_underlying(cuda::arch::id::sm_120a):
        __ret.__ptrs.push_back("-arch=sm_120a");
        break;
      default:
        _CCCL_UNREACHABLE();
    }

    return __ret;
  }

  //! @brief Add name expressions to the NVRTC program.
  //!
  //! @param __program The NVRTC program.
  //! @param __name_exprs The name expressions to add.
  //!
  //! @note This function must be called before __compile() is called.
  static void __add_name_expressions(::nvrtcProgram __program, _CUDA_VSTD::span<const ::std::string> __name_exprs)
  {
    for (const auto& __name_expr : __name_exprs)
    {
      if (::nvrtcAddNameExpression(__program, __name_expr.c_str()) != ::NVRTC_SUCCESS)
      {
        // todo: throw an exception if the name expression could not be added
      }
    }
  }

  //! @brief Compile the NVRTC program with the given options.
  //!
  //! @param __program The NVRTC program.
  //! @param __ptrs The compilation options.
  //!
  //! @return `true` if the compilation was successful; otherwise, `false`.
  [[nodiscard]] static bool __compile(::nvrtcProgram __program, __nvrtc_compile_options&& __opts)
  {
    // replace nullptrs in __opts.__ptrs with the corresponding strings
    auto __strs_it = __opts.__strs.begin();
    for (auto& __ptr : __opts.__ptrs)
    {
      if (__ptr == nullptr)
      {
        __ptr = __strs_it->c_str();
        ++__strs_it;
      }
    }

    const auto __result =
      ::nvrtcCompileProgram(__program, static_cast<int>(__opts.__ptrs.size()), __opts.__ptrs.data());
    return __result == ::NVRTC_SUCCESS;
  }

  //! @brief Gets the vector of lowered names for the given name expressions.
  //!
  //! @param __program The NVRTC program.
  //! @param __name_exprs The name expressions to lower.
  //!
  //! @return The vector of lowered names.
  //!
  //! @note This function must be called after __compile() has been called.
  [[nodiscard]] static ::std::vector<_CUDA_VSTD::string_view>
  __get_lowered_names(::nvrtcProgram __program, _CUDA_VSTD::span<const ::std::string> __name_exprs)
  {
    ::std::vector<_CUDA_VSTD::string_view> __lowered_names;
    __lowered_names.reserve(__name_exprs.size());

    for (const auto& __name_expr : __name_exprs)
    {
      const char* __lowered_name{};
      if (::nvrtcGetLoweredName(__program, __name_expr.c_str(), &__lowered_name) != ::NVRTC_SUCCESS)
      {
        // todo: throw an exception if the lowered name could not be retrieved
      }
      __lowered_names.push_back(_CUDA_VSTD::string_view{__lowered_name});
    }
    return __lowered_names;
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

  //! @brief Enable or disable automatic precompiled headers.
  //!
  //! @param __enable If `true`, automatic precompiled headers are enabled; otherwise, they are disabled.
  void enable_auto_precompiled_headers(bool __enable = true) noexcept
  {
    __pch_auto_ = __enable;
  }

  //! @brief Set precompiled headers directory.
  //!
  //! @param __dir_name The directory name for the precompiled headers.
  void set_precompiled_headers_dir(::std::string __dir_name) noexcept
  {
    __pch_dir_ = ::std::move(__dir_name);
  }

  //! @brief Compile CUDA source code to PTX.
  //!
  //! @param __cuda_src The CUDA source code to compile.
  //! @param __cuda_opts The CUDA compilation options to use.
  //!
  //! @return A compile_cuda_to_ptx_result object.
  [[nodiscard]] compile_cuda_to_ptx_result
  compile_to_ptx(const cuda_compile_source& __cuda_src, const cuda_compile_options& __cuda_opts)
  {
    auto __program = __make_program(__cuda_src);
    __add_name_expressions(__program, __cuda_src.__name_exprs_);

    auto __opts = __make_options(__cuda_opts);

    for (const auto& __pch_header : __cuda_src.__pch_headers_)
    {
      __opts.__ptrs.push_back("-use-pch");
      __opts.__ptrs.push_back(nullptr); // placeholder for string pointer
      __opts.__strs.push_back({__pch_header.begin(), __pch_header.end()});
    }

    const bool __success = __compile(__program, _CUDA_VSTD::move(__opts));

    ::std::vector<_CUDA_VSTD::string_view> __lowered_names;
    if (__success)
    {
      __lowered_names = __get_lowered_names(__program, __cuda_src.__name_exprs_);
    }

    return compile_cuda_to_ptx_result{__cuda_src.__id_, __program, __success, _CUDA_VSTD::move(__lowered_names)};
  }

  //! @brief Compile CUDA source code to CUBIN.
  //!
  //! @param __cuda_src The CUDA source code to compile.
  //! @param __cuda_opts The CUDA compilation options to use.
  //! @param __ptx_opts The PTX compilation options to use.
  //!
  //! @return A compile_cuda_to_cubin_result object.
  [[nodiscard]] compile_cuda_to_cubin_result compile_to_cubin(
    const cuda_compile_source& __cuda_src,
    const cuda_compile_options& __cuda_opts,
    const ptx_compile_options& __ptx_opts)
  {
    auto __program = __make_program(__cuda_src);
    __add_name_expressions(__program, __cuda_src.__name_exprs_);

    const bool __success = __compile(__program, __make_options(__cuda_opts, __ptx_opts));

    ::std::vector<_CUDA_VSTD::string_view> __lowered_names;
    if (__success)
    {
      __lowered_names = __get_lowered_names(__program, __cuda_src.__name_exprs_);
    }

    return compile_cuda_to_cubin_result{__cuda_src.__id_, __program, __success, _CUDA_VSTD::move(__lowered_names)};
  }

  //! @brief Compile CUDA source code to LTOIR.
  //!
  //! @param __cuda_src The CUDA source code to compile.
  //! @param __cuda_opts The CUDA compilation options to use.
  //!
  //! @return A compile_cuda_to_ltoir_result object.
  [[nodiscard]] compile_cuda_to_ltoir_result
  compile_to_ltoir(const cuda_compile_source& __cuda_src, const cuda_compile_options& __cuda_opts)
  {
    auto __program = __make_program(__cuda_src);
    __add_name_expressions(__program, __cuda_src.__name_exprs_);

    auto __opts = __make_options(__cuda_opts);
    __opts.__ptrs.push_back("-dlto");

    return compile_cuda_to_ltoir_result{__cuda_src.__id_, __program, __compile(__program, _CUDA_VSTD::move(__opts))};
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_CUDA_COMPILER_CUH
