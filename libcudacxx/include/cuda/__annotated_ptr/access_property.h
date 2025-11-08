//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_H
#define _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__annotated_ptr/access_property_encoding.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename>
class __annotated_ptr_base; // forward declaration

class access_property
{
private:
  uint64_t __descriptor = __l2_interleave_normal;

  friend class __annotated_ptr_base<access_property>;

  // needed by __annotated_ptr_base
  _CCCL_API constexpr access_property(uint64_t __descriptor1) noexcept
      : __descriptor{__descriptor1}
  {}

public:
  struct shared
  {};
  struct global
  {};
  struct persisting
  {
#if _CCCL_HAS_CTK()
    [[nodiscard]] _CCCL_API constexpr operator ::cudaAccessProperty() const noexcept
    {
      return ::cudaAccessProperty::cudaAccessPropertyPersisting;
    }
#endif // _CCCL_HAS_CTK()
  };
  struct streaming
  {
#if _CCCL_HAS_CTK()
    [[nodiscard]] _CCCL_API constexpr operator ::cudaAccessProperty() const noexcept
    {
      return ::cudaAccessProperty::cudaAccessPropertyStreaming;
    }
#endif // _CCCL_HAS_CTK()
  };
  struct normal
  {
#if _CCCL_HAS_CTK()
    [[nodiscard]] _CCCL_API constexpr operator ::cudaAccessProperty() const noexcept
    {
      return ::cudaAccessProperty::cudaAccessPropertyNormal;
    }
#endif // _CCCL_HAS_CTK()
  };

  _CCCL_HIDE_FROM_ABI access_property() noexcept = default;

  _CCCL_API constexpr access_property(normal, float __fraction) noexcept
      : __descriptor{
          ::cuda::__l2_interleave(__l2_evict_t::_L2_Evict_Normal_Demote, __l2_evict_t::_L2_Evict_Unchanged, __fraction)}
  {}
  _CCCL_API constexpr access_property(streaming, float __fraction) noexcept
      : __descriptor{
          ::cuda::__l2_interleave(__l2_evict_t::_L2_Evict_First, __l2_evict_t::_L2_Evict_Unchanged, __fraction)}
  {}
  _CCCL_API constexpr access_property(persisting, float __fraction) noexcept
      : __descriptor{
          ::cuda::__l2_interleave(__l2_evict_t::_L2_Evict_Last, __l2_evict_t::_L2_Evict_Unchanged, __fraction)}
  {}
  _CCCL_API constexpr access_property(normal, float __fraction, streaming) noexcept
      : __descriptor{
          ::cuda::__l2_interleave(__l2_evict_t::_L2_Evict_Normal_Demote, __l2_evict_t::_L2_Evict_First, __fraction)}
  {}
  _CCCL_API constexpr access_property(persisting, float __fraction, streaming) noexcept
      : __descriptor{::cuda::__l2_interleave(__l2_evict_t::_L2_Evict_Last, __l2_evict_t::_L2_Evict_First, __fraction)}
  {}

  _CCCL_API constexpr access_property(global) noexcept {}

  _CCCL_API constexpr access_property(normal) noexcept
      : access_property{normal{}, 1.0f}
  {}
  _CCCL_API constexpr access_property(streaming) noexcept
      : access_property{streaming{}, 1.0f}
  {}
  _CCCL_API constexpr access_property(persisting) noexcept
      : access_property{persisting{}, 1.0f}
  {}

  _CCCL_API inline access_property(void* __ptr, size_t __primary_bytes, size_t __total_bytes, normal) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_Normal_Demote,
          __l2_evict_t::_L2_Evict_Unchanged,
          __ptr,
          __primary_bytes,
          __total_bytes)}
  {}

  _CCCL_API inline access_property(void* __ptr, size_t __primary_bytes, size_t __total_bytes, streaming) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_First, __l2_evict_t::_L2_Evict_Unchanged, __ptr, __primary_bytes, __total_bytes)}
  {}

  _CCCL_API inline access_property(void* __ptr, size_t __primary_bytes, size_t __total_bytes, persisting) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_Last, __l2_evict_t::_L2_Evict_Unchanged, __ptr, __primary_bytes, __total_bytes)}
  {}

  _CCCL_API inline access_property(void* __ptr, size_t __primary_bytes, size_t __total_bytes, global, streaming) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_Unchanged, __l2_evict_t::_L2_Evict_First, __ptr, __primary_bytes, __total_bytes)}
  {}

  _CCCL_API inline access_property(void* __ptr, size_t __primary_bytes, size_t __total_bytes, normal, streaming) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_Normal_Demote, __l2_evict_t::_L2_Evict_First, __ptr, __primary_bytes, __total_bytes)}
  {}

  _CCCL_API inline access_property(
    void* __ptr, size_t __primary_bytes, size_t __total_bytes, streaming, streaming) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_First, __l2_evict_t::_L2_Evict_First, __ptr, __primary_bytes, __total_bytes)}
  {}

  _CCCL_API inline access_property(
    void* __ptr, size_t __primary_bytes, size_t __total_bytes, persisting, streaming) noexcept
      : __descriptor{::cuda::__block_encoding(
          __l2_evict_t::_L2_Evict_Last, __l2_evict_t::_L2_Evict_First, __ptr, __primary_bytes, __total_bytes)}
  {}

  [[nodiscard]] _CCCL_API constexpr explicit operator uint64_t() const noexcept
  {
    return __descriptor;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_H
