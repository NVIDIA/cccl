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

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/string_view>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace cuda_compile_options
{

//! @brief Option to define a macro for the CUDA compilation.
struct define_macro_opt
{
  _CUDA_VSTD::string_view name; //!< The name of the macro to define.
  _CUDA_VSTD::string_view value = {}; //!< The value of the macro to define. If empty, the macro is defined without a
                                      //!< value.
};

//! @brief Option to undefine a macro for the CUDA compilation.
struct undefine_macro_opt
{
  _CUDA_VSTD::string_view name; //!< The name of the macro to undefine.
};

//! @brief Option to specify an include path for the CUDA compilation.
struct include_path_opt
{
  _CUDA_VSTD::string_view path; //!< The include path to add.
};

//! @brief Option to force include a file in the CUDA compilation.
struct force_include_opt
{
  _CUDA_VSTD::string_view file_name; //!< The file to force include.
};

//! @brief Option to specify the C++ standard version for the CUDA compilation.
enum class std_version_opt
{
  cxx03 = 03, //!< C++03 standard version.
  cxx11 = 11, //!< C++11 standard version.
  cxx14 = 14, //!< C++14 standard version.
  cxx17 = 17, //!< C++17 standard version.
  cxx20 = 20, //!< C++20 standard version.
};

// todo: add other options

} // namespace cuda_compile_options

//! @brief Class to hold CUDA compilation options.
class cuda_compile_opts
{
  friend class cuda_compiler;

  enum class _DynOptType
  {
    __define_macro,
    __undefine_macro,
    __include_path,
    __force_include,
  };
  struct _StringView2
  {
    _CUDA_VSTD::string_view __first_;
    _CUDA_VSTD::string_view __second_;
  };
  union _DynOptValue
  {
    _CUDA_VSTD::string_view __string_view_;
    _StringView2 __string_view2_;
  };
  struct _DynOpt
  {
    _DynOptType __type_;
    _DynOptValue __value_;
  };

  ::std::vector<_DynOpt> __dyn_opts_;
  unsigned __std_version_ : 8;

public:
  //! @brief Default constructor for CUDA compilation options.
  cuda_compile_opts() noexcept
      : __dyn_opts_{}
      , __std_version_{_CUDA_VSTD::to_underlying(cuda_compile_options::std_version_opt::cxx17)}
  {}

  //! @brief Adds a compilation option to the list of options.
  //!
  //! @tparam _Tp The type of the option to add.
  //!
  //! @param __opt The option to add.
  template <class _Tp>
  void add_option(const _Tp& __opt)
  {
    using namespace cuda_compile_options;

    [[maybe_unused]] _DynOpt __dyn_opt{};

    // todo: add value checking
    if constexpr (_CUDA_VSTD::is_same_v<_Tp, define_macro_opt>)
    {
      __dyn_opt.__type_                  = _DynOptType::__define_macro;
      __dyn_opt.__value_.__string_view2_ = {__opt.name, __opt.value};
      __dyn_opts_.push_back(__dyn_opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, undefine_macro_opt>)
    {
      __dyn_opt.__type_                 = _DynOptType::__undefine_macro;
      __dyn_opt.__value_.__string_view_ = __opt.name;
      __dyn_opts_.push_back(__dyn_opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, include_path_opt>)
    {
      __dyn_opt.__type_                 = _DynOptType::__include_path;
      __dyn_opt.__value_.__string_view_ = __opt.path;
      __dyn_opts_.push_back(__dyn_opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, force_include_opt>)
    {
      __dyn_opt.__type_                 = _DynOptType::__force_include;
      __dyn_opt.__value_.__string_view_ = __opt.file_name;
      __dyn_opts_.push_back(__dyn_opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, std_version_opt>)
    {
      __std_version_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<_Tp>, "Unsupported option type by cuda_compile_opts");
    }
  }

  //! @brief Adds multiple compilation options to the list of options.
  //!
  //! @tparam _Tps The types of the options to add.
  //!
  //! @param __opts The options to add.
  template <class... _Tps>
  void add_options(const _Tps&... __opts)
  {
    (add_option(__opts), ...);
  }
};

namespace ptx_compile_options
{

//! @brief Option to enable or disable device debugging information in PTX compilation.
enum class device_debug_opt : bool
{
};

//! @brief Option to enable or disable line information in PTX compilation.
enum class line_info_opt : bool
{
};

//! @brief Option to enable or disable fused multiply-add (FMA) operations in PTX compilation.
enum class fmad_opt : bool
{
};

//! @brief Option to specify the maximum number of registers per thread in PTX compilation.
enum class max_reg_count_opt : int
{
  __unspecified = -3,
  arch_min      = -2, //!< Use the minimum number of registers for the architecture.
  arch_max      = -1, //!< Use the maximum number of registers for the architecture.
};

//! @brief Option to specify the optimization level for PTX compilation.
enum class optimization_level_opt
{
  O0,
  O1,
  O2,
  O3,
};

//! @brief Option to enable or disable position-independent code (PIC) in PTX compilation.
enum class pic_opt : bool
{
};

// todo: add other options

} // namespace ptx_compile_options

//! @brief Class to hold PTX compilation options.
class ptx_compile_opts
{
  friend class cuda_compiler;
  friend class ptx_compiler;

  int __max_reg_count_;
  unsigned __optimization_level_ : 2;
  unsigned __device_debug_       : 1;
  unsigned __line_info_          : 1;
  unsigned __fmad_               : 1;
  unsigned __pic_                : 1;

public:
  //! @brief Default constructor for PTX compilation options.
  ptx_compile_opts() noexcept
      : __max_reg_count_{_CUDA_VSTD::to_underlying(ptx_compile_options::max_reg_count_opt::__unspecified)}
      , __optimization_level_{_CUDA_VSTD::to_underlying(ptx_compile_options::optimization_level_opt::O3)}
      , __device_debug_{false}
      , __line_info_{false}
      , __fmad_{false}
      , __pic_{false}
  {}

  //! @brief Adds a PTX compilation option to the list of options.
  //!
  //! @tparam _Tp The type of the option to add.
  //!
  //! @param __opt The option to add.
  template <class _Tp>
  void add_option(const _Tp& __opt)
  {
    using namespace ptx_compile_options;

    // todo: add value checking
    if constexpr (_CUDA_VSTD::is_same_v<_Tp, device_debug_opt>)
    {
      __device_debug_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, line_info_opt>)
    {
      __line_info_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, fmad_opt>)
    {
      __fmad_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, max_reg_count_opt>)
    {
      __max_reg_count_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, optimization_level_opt>)
    {
      __optimization_level_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, pic_opt>)
    {
      __pic_ = _CUDA_VSTD::to_underlying(__opt);
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<_Tp>, "Unsupported option type");
    }
  }

  //! @brief Adds multiple PTX compilation options to the list of options.
  //!
  //! @tparam _Tps The types of the options to add.
  //!
  //! @param __opts The options to add.
  template <class... _Tps>
  void add_options(const _Tps&... __opts)
  {
    (add_option(__opts), ...);
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___COMPILER_COMPILE_OPTIONS_CUH
