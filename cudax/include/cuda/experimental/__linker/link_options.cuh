//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LINKER_LINK_OPTIONS_CUH
#define _CUDAX___LINKER_LINK_OPTIONS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

// Currently ptx and cubin link options are the same, but this may change in the future.
namespace __link_options
{

} // namespace __link_options

struct __link_opts
{
  //! @brief Adds a link option.
  //!
  //! @tparam _Tp The type of the option to add.
  //!
  //! @param option The option to add.
  template <class _Tp>
  void add_option(const _Tp& option)
  {}

  //! @brief Adds multiple link options.
  //!
  //! @tparam _Tps The types of the options to add.
  //!
  //! @param options The options to add.
  template <class... _Tps>
  void add_options(const _Tps&... options)
  {
    (add_option(options), ...);
  }
};

namespace ptx_link_options
{

using namespace cuda::experimental::__link_options;

} // namespace ptx_link_options

class ptx_link_opts : public __link_opts
{};

namespace cubin_link_options
{

using namespace cuda::experimental::__link_options;

} // namespace cubin_link_options

class cubin_link_opts : public __link_opts
{};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LINKER_LINK_OPTIONS_CUH
