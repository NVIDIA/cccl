//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_SHARED_MEMORY_ACCESSOR_H
#define _CUDA___MDSPAN_SHARED_MEMORY_ACCESSOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/address_space.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Accessor>
class __shared_memory_accessor;

template <typename _Accessor>
using shared_memory_accessor = __shared_memory_accessor<_Accessor>;

/***********************************************************************************************************************
 * Accessor Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_shared_memory_accessor_v = false;

template <typename _Accessor>
inline constexpr bool is_shared_memory_accessor_v<__shared_memory_accessor<_Accessor>> = true;

#define _CCCL_VERIFY_DEVICE_ONLY_USAGE(_VALUE) \
  NV_IF_TARGET(                                \
    NV_IS_HOST,                                \
    (_CCCL_VERIFY(false, "the function cannot be used in HOST code"); _CCCL_UNREACHABLE(); return (_VALUE);))

#define _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR() \
  NV_IF_TARGET(NV_IS_HOST, (_CCCL_VERIFY(false, "the function cannot be used in HOST code");))

/***********************************************************************************************************************
 * Shared Memory Accessor
 **********************************************************************************************************************/

#if _CCCL_CUDA_COMPILATION()

// TODO: move to a more appropriate place
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __max_smem_allocation_bytes() noexcept
{
  const auto __total_smem_size   = ::cuda::ptx::get_sreg_total_smem_size();
  const auto __dynamic_smem_size = ::cuda::ptx::get_sreg_dynamic_smem_size();
  const auto __static_smem_size  = __total_smem_size - __dynamic_smem_size;
  const auto __max_smem_size     = ::max(__static_smem_size, __dynamic_smem_size);
  return __max_smem_size;
}
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Accessor>
class __shared_memory_accessor : public _Accessor
{
  static_assert(::cuda::std::is_pointer_v<typename _Accessor::data_handle_type>, "Accessor must be pointer based");

  using __data_handle_type = typename _Accessor::data_handle_type;
  using __element_type     = ::cuda::std::remove_pointer_t<__data_handle_type>;

  static constexpr bool __is_access_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().access(::cuda::std::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(::cuda::std::declval<_Accessor>().offset(::cuda::std::declval<__data_handle_type>(), 0));

public:
  using offset_policy    = __shared_memory_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __element_type*;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(class _Accessor2 = _Accessor)
  _CCCL_REQUIRES(::cuda::std::is_default_constructible_v<_Accessor2>)
  _CCCL_API constexpr __shared_memory_accessor() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Accessor2>)
      : _Accessor{}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_API constexpr __shared_memory_accessor(const _Accessor& __acc) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_API constexpr __shared_memory_accessor(_Accessor&& __acc) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Accessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    ::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr __shared_memory_accessor(const __shared_memory_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, const _OtherAccessor&> _CCCL_AND(
    !::cuda::std::is_convertible_v<const _OtherAccessor&, _Accessor>))
  _CCCL_API constexpr explicit __shared_memory_accessor(const __shared_memory_accessor<_OtherAccessor>& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, const _OtherAccessor&>)
      : _Accessor{__acc}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    ::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr __shared_memory_accessor(__shared_memory_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Accessor, _OtherAccessor> _CCCL_AND(
    !::cuda::std::is_convertible_v<_OtherAccessor, _Accessor>))
  _CCCL_API constexpr explicit __shared_memory_accessor(__shared_memory_accessor<_OtherAccessor>&& __acc) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Accessor, _OtherAccessor>)
      : _Accessor{::cuda::std::move(__acc)}
  {
    _CCCL_VERIFY_DEVICE_ONLY_USAGE_CTOR();
  }

  _CCCL_API reference access(__element_type* __p, ::cuda::std::size_t __i) const noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(
      NV_IS_DEVICE,
      (bool __is_shared_mem = ::__isShared(__p); //
       _CCCL_ASSERT(__is_shared_mem, "__p is not a shared memory pointer");
       _CCCL_ASSERT(__i <= ::cuda::__max_smem_allocation_bytes() / sizeof(__element_type),
                    "__i exceeds the maximum shared memory allocation size");
       _CCCL_ASSUME(__is_shared_mem);
       return _Accessor::access(__p, __i);))
    _CCCL_VERIFY_DEVICE_ONLY_USAGE(_Accessor::access(__p, __i));
  }

  _CCCL_API data_handle_type offset(__element_type* __p, ::cuda::std::size_t __i) const noexcept(__is_offset_noexcept)
  {
    NV_IF_TARGET(
      NV_IS_DEVICE,
      (bool __is_shared_mem = ::__isShared(__p); //
       _CCCL_ASSERT(__is_shared_mem, "__p is not a shared memory pointer");
       _CCCL_ASSERT(__i <= ::cuda::__max_smem_allocation_bytes() / sizeof(__element_type),
                    "__i exceeds the maximum shared memory allocation size");
       _CCCL_ASSUME(__is_shared_mem);
       return _Accessor::offset(__p, __i);))
    _CCCL_VERIFY_DEVICE_ONLY_USAGE(_Accessor::offset(__p, __i));
  }

  [[nodiscard]] _CCCL_API static bool __detectably_invalid(
    [[maybe_unused]] data_handle_type __p, [[maybe_unused]] ::cuda::std::size_t __size_bytes) noexcept
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (bool __is_shared_mem     = ::cuda::device::is_address_from(__p, device::address_space::shared);
                  bool __exceeds_smem_size = __size_bytes > ::cuda::__max_smem_allocation_bytes();
                  return __is_shared_mem || __exceeds_smem_size;))
    _CCCL_VERIFY_DEVICE_ONLY_USAGE(false);
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_SHARED_MEMORY_ACCESSOR_H
