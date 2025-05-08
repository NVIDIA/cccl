//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ANNOTATED_PTR_ANNOTATED_PTR_BASE
#define _CUDA___ANNOTATED_PTR_ANNOTATED_PTR_BASE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__annotated_ptr/access_property.h>
#include <cuda/__annotated_ptr/associate_access_property.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _AccessProperty>
class __annotated_ptr_base
{
protected:
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint64_t __default_property() noexcept
  {
    return _CUDA_VSTD::is_same_v<_AccessProperty, access_property::global>     ? __l2_interleave_normal
         : _CUDA_VSTD::is_same_v<_AccessProperty, access_property::normal>     ? __l2_interleave_normal_demote
         : _CUDA_VSTD::is_same_v<_AccessProperty, access_property::persisting> ? __l2_interleave_persisting
         : _CUDA_VSTD::is_same_v<_AccessProperty, access_property::streaming>
           ? __l2_interleave_streaming
           : 0; // access_property::shared;
  }

  static constexpr uint64_t __prop = __default_property();

  _CCCL_HIDE_FROM_ABI __annotated_ptr_base() noexcept = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __annotated_ptr_base(_AccessProperty) noexcept {}

#if _CCCL_HAS_CUDA_COMPILER()

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __apply_prop(void* __p) const
  {
    return ::cuda::__associate(__p, _AccessProperty{});
  }

#endif // _CCCL_HAS_CUDA_COMPILER()

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _AccessProperty __get_property() const noexcept
  {
    return _AccessProperty{};
  }
};

//----------------------------------------------------------------------------------------------------------------------
// Specialization for dynamic access property

template <>
class __annotated_ptr_base<access_property>
{
protected:
  uint64_t __prop = static_cast<uint64_t>(access_property{});

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __annotated_ptr_base(access_property __property) noexcept
      : __prop{static_cast<uint64_t>(__property)}
  {}

  _CCCL_HIDE_FROM_ABI __annotated_ptr_base() noexcept = default;

#if _CCCL_HAS_CUDA_COMPILER()
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __apply_prop(void* __p) const
  {
    return ::cuda::__associate_raw_descriptor(__p, __prop);
  }
#endif // _CCCL_HAS_CUDA_COMPILER()

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr access_property __get_property() const noexcept
  {
    return access_property{__prop};
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___ANNOTATED_PTR_ANNOTATED_PTR_BASE
