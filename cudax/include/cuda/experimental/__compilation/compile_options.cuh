//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___COMPILATION_COMPILE_OPTIONS_CUH
#define _CUDAX___COMPILATION_COMPILE_OPTIONS_CUH

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
  cxx17 = 17, //!< C++17 standard version.
  cxx20 = 20, //!< C++20 standard version.
};

} // namespace cuda_compile_options

//! @brief Class to hold CUDA compilation options.
class cuda_compile_opts
{
public: // todo: make this private
  enum class _OptType
  {
    __define_macro,
    __undefine_macro,
    __include_path,
    __force_include,
    __std_ver,
  };
  struct _StringView2
  {
    _CUDA_VSTD::string_view __first_;
    _CUDA_VSTD::string_view __second_;
  };
  union _OptValue
  {
    _CUDA_VSTD::string_view __string_view_;
    _StringView2 __string_view2_;
    int __int_;
  };
  struct _Opt
  {
    _OptType __type_;
    _OptValue __value_;
  };

  ::std::vector<_Opt> __opts_;

public:
  //! @brief Adds a compilation option to the list of options.
  //!
  //! @tparam _Tp The type of the option to add.
  //!
  //! @param __opt The option to add.
  template <class _Tp>
  void add_option(const _Tp& __opt)
  {
    using namespace cuda_compile_options;

    _Opt __new_opt{};

    // todo: add value checking
    if constexpr (_CUDA_VSTD::is_same_v<_Tp, define_macro_opt>)
    {
      __new_opt.__type_                  = _OptType::__define_macro;
      __new_opt.__value_.__string_view2_ = {__opt.name, __opt.value};
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, undefine_macro_opt>)
    {
      __new_opt.__type_                 = _OptType::__undefine_macro;
      __new_opt.__value_.__string_view_ = __opt.name;
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, include_path_opt>)
    {
      __new_opt.__type_                 = _OptType::__include_path;
      __new_opt.__value_.__string_view_ = __opt.path;
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, force_include_opt>)
    {
      __new_opt.__type_                 = _OptType::__force_include;
      __new_opt.__value_.__string_view_ = __opt.file_name;
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, std_version_opt>)
    {
      __new_opt.__type_         = _OptType::__std_ver;
      __new_opt.__value_.__int_ = static_cast<int>(__opt);
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<_Tp>, "Unsupported option type by cuda_compile_opts");
    }
    __opts_.push_back(__new_opt);
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
enum class max_reg_count_opt : unsigned
{
};

//! @brief Option to specify the optimization level for PTX compilation.
enum class optimization_level_opt : int
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

} // namespace ptx_compile_options

//! @brief Class to hold PTX compilation options.
class ptx_compile_opts
{
public: // todo: make this private
  enum class _OptType
  {
    __device_debug,
    __line_info,
    __fmad,
    __max_reg_count,
    __optimization_level,
    __pic,
  };
  union _OptValue
  {
    _CUDA_VSTD::string_view __string_view_;
    int __int_;
    unsigned __unsigned_;
    bool __boolean_;
  };
  struct _Opt
  {
    _OptType __type_;
    _OptValue __value_;
  };

  ::std::vector<_Opt> __opts_;

public:
  //! @brief Adds a PTX compilation option to the list of options.
  //!
  //! @tparam _Tp The type of the option to add.
  //!
  //! @param __opt The option to add.
  template <class _Tp>
  void add_option(const _Tp& __opt)
  {
    using namespace ptx_compile_options;

    _Opt __new_opt{};

    // todo: add value checking
    if constexpr (_CUDA_VSTD::is_same_v<_Tp, device_debug_opt>)
    {
      __new_opt.__type_             = _OptType::__device_debug;
      __new_opt.__value_.__boolean_ = static_cast<bool>(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, line_info_opt>)
    {
      __new_opt.__type_             = _OptType::__line_info;
      __new_opt.__value_.__boolean_ = static_cast<bool>(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, fmad_opt>)
    {
      __new_opt.__type_             = _OptType::__fmad;
      __new_opt.__value_.__boolean_ = static_cast<bool>(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, max_reg_count_opt>)
    {
      __new_opt.__type_              = _OptType::__max_reg_count;
      __new_opt.__value_.__unsigned_ = static_cast<unsigned>(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, optimization_level_opt>)
    {
      __new_opt.__type_         = _OptType::__optimization_level;
      __new_opt.__value_.__int_ = static_cast<int>(__opt);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, pic_opt>)
    {
      __new_opt.__type_             = _OptType::__pic;
      __new_opt.__value_.__boolean_ = static_cast<bool>(__opt);
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<_Tp>, "Unsupported option type");
    }
    __opts_.push_back(__new_opt);
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

#endif // _CUDAX___COMPILATION_COMPILE_OPTIONS_CUH
