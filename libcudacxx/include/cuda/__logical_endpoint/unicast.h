//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LOGICAL_ENDPOINT_UNICAST_H
#define _CUDA___LOGICAL_ENDPOINT_UNICAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__logical_endpoint/common.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Non-owning reference to a unicast CUDA logical endpoint.
//!
//! This type is trivially copyable and can be passed to device code directly, including raw `<<<>>>` kernel launches.
class unicast_logical_endpoint_ref
    : public ::cuda::__detail::__logical_endpoint_ref_base<::cuda::__detail::__logical_endpoint_type::__unicast>
{
  using __base = ::cuda::__detail::__logical_endpoint_ref_base<::cuda::__detail::__logical_endpoint_type::__unicast>;

public:
  //! @brief Creates a unicast endpoint reference from a logical endpoint ID.
  //!
  //! @param[in] __id The logical endpoint ID.
  _CCCL_HOST_DEVICE_API explicit constexpr unicast_logical_endpoint_ref(logical_endpoint_id __id) noexcept
      : __base{__id}
  {}
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LOGICAL_ENDPOINT_UNICAST_H
