//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EXECUTION_POLICY_H
#define _CUDA___EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/__fwd/execution_policy.h>
#  include <cuda/__memory_resource/device_memory_pool.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/forward.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

template <bool _HasStream>
struct __policy_stream_holder
{
  ::cuda::stream_ref __stream_;

  _CCCL_HOST_API constexpr __policy_stream_holder(::cuda::stream_ref __stream) noexcept
      : __stream_(__stream)
  {}

  //! @brief Return the stream stored in the holder or a default stream
  [[nodiscard]] _CCCL_HOST_API ::cuda::stream_ref query(const ::cuda::get_stream_t&) const noexcept
  {
    return __stream_;
  }
};

template <>
struct __policy_stream_holder<false>
{
  //! @brief Return the stream stored in the holder or a default stream
  [[nodiscard]] _CCCL_HOST_API ::cuda::stream_ref query(const ::cuda::get_stream_t&) const noexcept
  {
    return ::cuda::stream_ref{cudaStreamPerThread};
  }
};

template <bool _HasResource>
struct __policy_memory_resource_holder
{
  using __resource_t = ::cuda::mr::any_resource<::cuda::mr::device_accessible>;

  __resource_t __resource_;

  _CCCL_TEMPLATE(class _Resource)
  _CCCL_REQUIRES(::cuda::mr::resource_with<_Resource, ::cuda::mr::device_accessible>)
  _CCCL_HOST_API constexpr __policy_memory_resource_holder(_Resource&& __resource) noexcept
      : __resource_(::cuda::std::forward<_Resource>(__resource))
  {}
};

template <>
struct __policy_memory_resource_holder<false>
{};

template <uint32_t _Policy>
struct _CCCL_DECLSPEC_EMPTY_BASES __execution_policy_base<_Policy, __execution_backend::__cuda>
    : __execution_policy_base<_Policy, __execution_backend::__none>
    , protected __policy_stream_holder<__cuda_policy_with_stream<_Policy>>
    , protected __policy_memory_resource_holder<__cuda_policy_with_memory_resource<_Policy>>
{
  _CCCL_HIDE_FROM_ABI constexpr __execution_policy_base() noexcept = default;

  //! Forward the queries
  using __policy_stream_holder<__cuda_policy_with_stream<_Policy>>::query;

  //! @brief Set the current stream
  _CCCL_TEMPLATE(bool _WithStream = __cuda_policy_with_stream<_Policy>)
  _CCCL_REQUIRES(_WithStream)
  [[nodiscard]] _CCCL_HOST_API __execution_policy_base& set_stream(::cuda::stream_ref __stream) noexcept
  {
    this->__stream_ = __stream;
    return *this;
  }

  //! @brief Convert to a policy that holds a stream
  _CCCL_TEMPLATE(bool _WithStream = __cuda_policy_with_stream<_Policy>)
  _CCCL_REQUIRES((!_WithStream))
  [[nodiscard]] _CCCL_HOST_API auto set_stream(::cuda::stream_ref __stream) const noexcept
  {
    constexpr uint32_t __new_policy = _Policy | (static_cast<uint32_t>(__cuda_backend_options::__with_stream) << 16);
    return __execution_policy_base<__new_policy>{*this, __stream};
  }

  //! @brief Set the current memory resource
  _CCCL_TEMPLATE(class _Resource, bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(::cuda::mr::resource_with<_Resource, ::cuda::mr::device_accessible> _CCCL_AND _WithResource)
  [[nodiscard]] _CCCL_HOST_API __execution_policy_base& set_memory_resource(_Resource&& __resource) noexcept
  {
    this->__resource_ = __resource;
    return *this;
  }

  //! @brief Convert to a policy that holds a memory resource
  _CCCL_TEMPLATE(class _Resource, bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(::cuda::mr::resource_with<_Resource, ::cuda::mr::device_accessible> _CCCL_AND(!_WithResource))
  [[nodiscard]] _CCCL_HOST_API auto set_memory_resource(_Resource&& __resource) const noexcept
  {
    constexpr uint32_t __new_policy =
      _Policy | (static_cast<uint32_t>(__cuda_backend_options::__with_memory_resource) << 16);
    return __execution_policy_base<__new_policy>{*this, __resource};
  }

  //! @brief Return either a stored or a default memory resource
  //! @note We cannot put that into the __policy_memory_resource_holder because we need a stream for the device
  [[nodiscard]] _CCCL_HOST_API auto query(const ::cuda::mr::get_memory_resource_t&) const noexcept
  {
    if constexpr (__cuda_policy_with_memory_resource<_Policy>)
    {
      return this->__resource_;
    }
    else
    {
      ::cuda::stream_ref __stream = this->query(::cuda::get_stream);
      return ::cuda::device_default_memory_pool(__stream.device());
    }
  }

private:
  template <uint32_t, __execution_backend>
  friend struct __execution_policy_base;

  _CCCL_TEMPLATE(uint32_t _OtherPolicy,
                 bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithStream _CCCL_AND _WithResource)
  _CCCL_HOST_API constexpr __execution_policy_base(
    const __execution_policy_base<_OtherPolicy, __execution_backend::__cuda>& __policy,
    ::cuda::stream_ref __stream) noexcept
      : __policy_stream_holder<_WithStream>(__stream)
      , __policy_memory_resource_holder<_WithResource>(__policy.query(::cuda::mr::get_memory_resource))
  {}

  _CCCL_TEMPLATE(uint32_t _OtherPolicy,
                 bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithStream _CCCL_AND(!_WithResource))
  _CCCL_HOST_API constexpr __execution_policy_base(
    const __execution_policy_base<_OtherPolicy, __execution_backend::__cuda>& __policy,
    ::cuda::stream_ref __stream) noexcept
      : __policy_stream_holder<_WithStream>(__stream)
      , __policy_memory_resource_holder<_WithResource>()
  {}

  _CCCL_TEMPLATE(uint32_t _OtherPolicy,
                 class _Resource,
                 bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithStream _CCCL_AND _WithResource)
  _CCCL_HOST_API constexpr __execution_policy_base(
    const __execution_policy_base<_OtherPolicy, __execution_backend::__cuda>& __policy, _Resource&& __resource) noexcept
      : __policy_stream_holder<_WithStream>(__policy.query(::cuda::get_stream))
      , __policy_memory_resource_holder<_WithResource>(::cuda::std::forward<_Resource>(__resource))
  {}

  _CCCL_TEMPLATE(uint32_t _OtherPolicy,
                 class _Resource,
                 bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES((!_WithStream) _CCCL_AND _WithResource)
  _CCCL_HOST_API constexpr __execution_policy_base(
    const __execution_policy_base<_OtherPolicy, __execution_backend::__cuda>& __policy, _Resource&& __resource) noexcept
      : __policy_stream_holder<_WithStream>()
      , __policy_memory_resource_holder<_WithResource>(::cuda::std::forward<_Resource>(__resource))
  {}
};

_CCCL_END_NAMESPACE_EXECUTION

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

using __cub_parallel_unsequenced_policy =
  ::cuda::std::execution::__execution_policy_base<::cuda::std::execution::__with_cuda_backend<static_cast<uint32_t>(
    ::cuda::std::execution::__execution_policy::__parallel_unsequenced)>()>;
_CCCL_GLOBAL_CONSTANT __cub_parallel_unsequenced_policy __cub_par_unseq{};

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA___EXECUTION_POLICY_H
