//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_PARAM_KIND
#define _CUDAX__LAUNCH_PARAM_KIND

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/maybe_const.h>

#include <cuda/experimental/__detail/utility.cuh>

namespace cuda::experimental
{
namespace detail
{
enum class __param_kind : unsigned
{
  _in    = 1,
  _out   = 2,
  _inout = 3
};

_CCCL_NODISCARD _CCCL_HOST_DEVICE inline constexpr __param_kind operator&(__param_kind __a, __param_kind __b) noexcept
{
  return __param_kind(unsigned(__a) & unsigned(__b));
}

template <typename _Ty, __param_kind _Kind>
struct _CCCL_NODISCARD __box
{
  ::cuda::std::__maybe_const<_Kind == __param_kind::_in, _Ty>& __val;
};

struct __in_t
{
  template <class _Ty>
  __box<_Ty, __param_kind::_in> operator()(const _Ty& __v) const noexcept
  {
    return {__v};
  }
};

struct __out_t
{
  template <class _Ty>
  __box<_Ty, __param_kind::_out> operator()(_Ty& __v) const noexcept
  {
    return {__v};
  }
};

struct __inout_t
{
  template <class _Ty>
  __box<_Ty, __param_kind::_inout> operator()(_Ty& __v) const noexcept
  {
    return {__v};
  }
};

} // namespace detail

_CCCL_GLOBAL_CONSTANT detail::__in_t in{};
_CCCL_GLOBAL_CONSTANT detail::__out_t out{};
_CCCL_GLOBAL_CONSTANT detail::__inout_t inout{};

} // namespace cuda::experimental

#endif // _CUDAX__LAUNCH_PARAM_KIND
