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

#  include <cuda/__functional/call_or.h>
#  include <cuda/__fwd/execution_policy.h>
#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

template <bool _HasStream>
struct __policy_stream_holder
{
  ::cuda::stream_ref __stream_;

  _CCCL_HOST_API constexpr __policy_stream_holder(::cuda::stream_ref __stream) noexcept
      : __stream_(__stream)
  {}
};

template <>
struct __policy_stream_holder<false>
{
  _CCCL_HIDE_FROM_ABI __policy_stream_holder() = default;

  //! @brief Dummy constructor to simplify implementation of the cuda policy
  _CCCL_HOST_API constexpr __policy_stream_holder(::cuda::stream_ref) noexcept {}
};

template <bool _HasResource>
struct __policy_memory_resource_holder
{
  ::cuda::mr::resource_ref<::cuda::mr::device_accessible> __resource_;

  _CCCL_HIDE_FROM_ABI __policy_memory_resource_holder() = default;

  _CCCL_HOST_API constexpr __policy_memory_resource_holder(
    ::cuda::mr::resource_ref<::cuda::mr::device_accessible> __resource) noexcept
      : __resource_(__resource)
  {}
};

template <>
struct __policy_memory_resource_holder<false>
{
  _CCCL_HIDE_FROM_ABI __policy_memory_resource_holder() = default;

  //! @brief Dummy constructor to simplify implementation of the cuda policy
  _CCCL_HOST_API constexpr __policy_memory_resource_holder(
    ::cuda::mr::resource_ref<::cuda::mr::device_accessible>) noexcept
  {}
};

template <uint32_t _Policy>
struct _CCCL_DECLSPEC_EMPTY_BASES __execution_policy_base<_Policy, __execution_backend::__cuda>
    : __execution_policy_base<_Policy, __execution_backend::__none>
    , protected __policy_stream_holder<__cuda_policy_with_stream<_Policy>>
    , protected __policy_memory_resource_holder<__cuda_policy_with_memory_resource<_Policy>>
{
private:
  template <uint32_t, __execution_backend>
  friend struct __execution_policy_base;

  using __stream_holder   = __policy_stream_holder<__cuda_policy_with_stream<_Policy>>;
  using __resource_holder = __policy_memory_resource_holder<__cuda_policy_with_memory_resource<_Policy>>;

  _CCCL_TEMPLATE(bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES((_WithStream) _CCCL_AND(!_WithResource))
  _CCCL_HOST_API constexpr __execution_policy_base(::cuda::stream_ref __stream) noexcept
      : __stream_holder(__stream)
      , __resource_holder()
  {}

  _CCCL_TEMPLATE(bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES((!_WithStream) _CCCL_AND _WithResource)
  _CCCL_HOST_API constexpr __execution_policy_base(
    ::cuda::mr::resource_ref<::cuda::mr::device_accessible> __resource) noexcept
      : __stream_holder()
      , __resource_holder(__resource)
  {}

  _CCCL_TEMPLATE(bool _WithStream   = __cuda_policy_with_stream<_Policy>,
                 bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithStream _CCCL_AND _WithResource)
  _CCCL_HOST_API constexpr __execution_policy_base(
    ::cuda::stream_ref __stream, ::cuda::mr::resource_ref<::cuda::mr::device_accessible> __resource) noexcept
      : __stream_holder(__stream)
      , __resource_holder(__resource)
  {}

public:
  _CCCL_HIDE_FROM_ABI constexpr __execution_policy_base() noexcept = default;

  //! @brief Convert to a policy that holds a stream
  //! @note This cannot be merged with the other case where we already have a stream as this needs to be const qualified
  //!       This is because we start with a constexpr global and modify that through set_stream
  _CCCL_TEMPLATE(bool _WithStream = __cuda_policy_with_stream<_Policy>)
  _CCCL_REQUIRES((!_WithStream))
  [[nodiscard]] _CCCL_HOST_API auto set_stream(::cuda::stream_ref __stream) const noexcept
  {
    constexpr uint32_t __new_policy = __set_cuda_backend_option<_Policy, __cuda_backend_options::__with_stream>;
    if constexpr (__cuda_policy_with_memory_resource<_Policy>)
    {
      return __execution_policy_base<__new_policy>{__stream, this->__resource_};
    }
    else
    {
      return __execution_policy_base<__new_policy>{__stream};
    }
  }

  //! @brief Set the current stream
  _CCCL_TEMPLATE(bool _WithStream = __cuda_policy_with_stream<_Policy>)
  _CCCL_REQUIRES(_WithStream)
  [[nodiscard]] _CCCL_HOST_API __execution_policy_base& set_stream(::cuda::stream_ref __stream) noexcept
  {
    this->__stream_ = __stream;
    return *this;
  }

  //! @brief Return the stream stored in the holder or a default stream
  _CCCL_TEMPLATE(bool _WithStream = __cuda_policy_with_stream<_Policy>)
  _CCCL_REQUIRES(_WithStream)
  [[nodiscard]] _CCCL_HOST_API ::cuda::stream_ref query(const ::cuda::get_stream_t&) const noexcept
  {
    return this->__stream_;
  }

  //! @brief Convert to a policy that holds a memory resource
  //! @warning We hold the memory resource by reference, so passing rvalue is a bug
  _CCCL_TEMPLATE(class _Resource, bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES((!_WithResource))
  [[nodiscard]] _CCCL_HOST_API auto set_memory_resource(_Resource&&) const = delete;

  //! @brief Convert to a policy that holds a memory resource
  //! @note This cannot be merged with the other case as this needs to be const qualified
  //!       This is because we start with a constexpr global and modify that through set_stream
  _CCCL_TEMPLATE(class _Resource, bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES((!_WithResource))
  [[nodiscard]] _CCCL_HOST_API auto set_memory_resource(_Resource& __resource) const noexcept
  {
    constexpr uint32_t __new_policy =
      __set_cuda_backend_option<_Policy, __cuda_backend_options::__with_memory_resource>;
    if constexpr (__cuda_policy_with_stream<_Policy>)
    {
      return __execution_policy_base<__new_policy>{this->__stream_, __resource};
    }
    else
    {
      return __execution_policy_base<__new_policy>{__resource};
    }
  }

  //! @brief Set the current memory resource
  //! @warning We hold the memory resource by reference, so passing rvalue is a bug
  _CCCL_TEMPLATE(class _Resource, bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithResource)
  [[nodiscard]] _CCCL_HOST_API __execution_policy_base& set_memory_resource(_Resource&&) = delete;

  //! @brief Set the current memory resource
  _CCCL_TEMPLATE(class _Resource, bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithResource)
  [[nodiscard]] _CCCL_HOST_API __execution_policy_base& set_memory_resource(_Resource& __resource) noexcept
  {
    this->__resource_ = __resource;
    return *this;
  }

  //! @brief Return either a stored or a default memory resource
  _CCCL_TEMPLATE(bool _WithResource = __cuda_policy_with_memory_resource<_Policy>)
  _CCCL_REQUIRES(_WithResource)
  [[nodiscard]] _CCCL_HOST_API auto query(const ::cuda::mr::get_memory_resource_t&) const noexcept
  {
    return this->__resource_;
  }

  template <uint32_t _OtherPolicy, __execution_backend _OtherBackend>
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(
    const __execution_policy_base& __lhs, const __execution_policy_base<_OtherPolicy, _OtherBackend>& __rhs) noexcept
  {
    if constexpr (_Policy != _OtherPolicy)
    {
      return false;
    }

    if constexpr (__cuda_policy_with_stream<_Policy>)
    {
      if (__lhs.query(::cuda::get_stream) != __rhs.query(::cuda::get_stream))
      {
        return false;
      }
    }

    if constexpr (__cuda_policy_with_memory_resource<_Policy>)
    {
      if (__lhs.query(::cuda::mr::get_memory_resource) != __rhs.query(::cuda::mr::get_memory_resource))
      {
        return false;
      }
    }

    return true;
  }

#  if _CCCL_STD_VER <= 2017
  template <uint32_t _OtherPolicy, __execution_backend _OtherBackend>
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(
    const __execution_policy_base& __lhs, const __execution_policy_base<_OtherPolicy, _OtherBackend>& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#  endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

using __cub_parallel_unsequenced_policy =
  ::cuda::std::execution::__execution_policy_base<::cuda::std::execution::__with_cuda_backend<static_cast<uint32_t>(
    ::cuda::std::execution::__execution_policy::__parallel_unsequenced)>()>;
_CCCL_GLOBAL_CONSTANT __cub_parallel_unsequenced_policy __cub_par_unseq{};

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA___EXECUTION_POLICY_H
