//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_RTTI_H
#define __CUDAX_DETAIL_BASIC_ANY_RTTI_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_ptrs.cuh>

#include <typeinfo> // IWYU pragma: keep (for std::bad_cast)

#include <nv/target>

_CCCL_PUSH_MACROS
#undef interface

namespace cuda::experimental
{
//!
//! __iunknown: Logically, the root of all interfaces.
//!
struct iunknown : interface<_CUDA_VSTD::__type_always<iunknown>::__call>
{};

//!
//! bad_any_cast
//!
struct bad_any_cast : ::std::bad_cast
{
  bad_any_cast() noexcept                                       = default;
  bad_any_cast(bad_any_cast const&) noexcept                    = default;
  ~bad_any_cast() noexcept override                             = default;
  auto operator=(bad_any_cast const&) noexcept -> bad_any_cast& = default;

  auto what() const noexcept -> char const* override
  {
    return "cannot cast value to target type";
  }
};

[[noreturn]] _CUDAX_HOST_API inline void __throw_bad_any_cast()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw bad_any_cast();), (_CUDA_VSTD_NOVERSION::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

struct __rtti_base : detail::__immovable
{
  _CUDAX_HOST_API constexpr __rtti_base(
    __vtable_kind __kind, uint16_t __nbr_interfaces, _CUDA_VSTD::__type_info_ref __self) noexcept
      : __kind_(__kind)
      , __nbr_interfaces_(__nbr_interfaces)
      , __typeid_(&__self)
  {}

  uint8_t __version_                    = __basic_any_version;
  __vtable_kind __kind_                 = __vtable_kind::__normal;
  uint16_t __nbr_interfaces_            = 0;
  uint32_t const __cookie_              = 0xDEADBEEF;
  _CUDA_VSTD::__type_info_ptr __typeid_ = nullptr;
};

static_assert(sizeof(__rtti_base) == sizeof(uint64_t) + sizeof(void*));

// Used to map an interface typeid to a pointer to the vtable for that interface.
struct __base_info
{
  _CUDA_VSTD::__type_info_ptr __typeid_;
  __base_vptr __vptr_;
};

inline constexpr size_t __half_size_t_bits = sizeof(size_t) * CHAR_BIT / 2;

// The metadata for the type-erased object. All vtables have an rtti sub-object,
// which contains a sub-object of this type.
struct __object_metadata
{
  size_t __size_ : __half_size_t_bits;
  size_t __align_ : __half_size_t_bits;
  _CUDA_VSTD::__type_info_ptr __object_typeid_;
  _CUDA_VSTD::__type_info_ptr __pointer_typeid_;
  _CUDA_VSTD::__type_info_ptr __const_pointer_typeid_;
};

template <class _Tp>
_CCCL_GLOBAL_CONSTANT __object_metadata __object_metadata_v = {
  sizeof(_Tp), alignof(_Tp), &_CCCL_TYPEID(_Tp), &_CCCL_TYPEID(_Tp*), &_CCCL_TYPEID(_Tp const*)};

template <class _Tp>
_CUDAX_HOST_API void __dtor_fn(void* __pv, bool __small) noexcept
{
  __small ? static_cast<_Tp*>(__pv)->~_Tp() //
          : delete *static_cast<_Tp**>(__pv);
}

// All vtables have an rtti sub-object. This object has several responsibilities:
// * It contains the destructor for the type-erased object.
// * It contains the metadata for the type-erased object.
// * It contains a map from the base interfaces typeids to their vtables for use
//   in dynamic_cast-like functionality.
struct __rtti : __rtti_base
{
  template <class _Tp, class _Super, class... _Interfaces>
  _CUDAX_TRIVIAL_HOST_API constexpr __rtti(
    __tag<_Tp, _Super>, __tag<_Interfaces...>, __base_info const* __base_vptr_map) noexcept
      : __rtti_base{__vtable_kind::__rtti, sizeof...(_Interfaces), _CCCL_TYPEID(__rtti)}
      , __dtor_(&__dtor_fn<_Tp>)
      , __object_info_(&__object_metadata_v<_Tp>)
      , __interface_typeid_{&_CCCL_TYPEID(_Super)}
      , __base_vptr_map_{__base_vptr_map}
  {}

  template <class... _Interfaces>
  [[nodiscard]] _CUDAX_HOST_API auto __query_interface(__iset<_Interfaces...>) const noexcept
    -> __vptr_for<__iset<_Interfaces...>>
  {
    // TODO: find a way to check at runtime that the requested __iset is a subset
    // of the interfaces in the vtable.
    return static_cast<__vptr_for<__iset<_Interfaces...>>>(this);
  }

  // Sequentially search the base_vptr_map for the requested interface by
  // comparing typeids. If the requested interface is found, return a pointer to
  // its vtable; otherwise, return nullptr.
  template <class _Interface>
  [[nodiscard]] _CUDAX_HOST_API auto __query_interface(_Interface) const noexcept -> __vptr_for<_Interface>
  {
    // On sane implementations, comparing type_info objects first compares their
    // addresses and, if that fails, it does a string comparison. What we want is
    // to check _all_ the addresses first, and only if they all fail, resort to
    // string comparisons. So do two passes over the __base_vptr_map.
    constexpr _CUDA_VSTD::__type_info_ref __id = _CCCL_TYPEID(_Interface);

    for (size_t __i = 0; __i < __nbr_interfaces_; ++__i)
    {
      if (&__id == __base_vptr_map_[__i].__typeid_)
      {
        return static_cast<__vptr_for<_Interface>>(__base_vptr_map_[__i].__vptr_);
      }
    }

    for (size_t __i = 0; __i < __nbr_interfaces_; ++__i)
    {
      if (__id == *__base_vptr_map_[__i].__typeid_)
      {
        return static_cast<__vptr_for<_Interface>>(__base_vptr_map_[__i].__vptr_);
      }
    }

    return nullptr;
  }

  void (*__dtor_)(void*, bool) noexcept;
  __object_metadata const* __object_info_;
  _CUDA_VSTD::__type_info_ptr __interface_typeid_ = nullptr;
  __base_info const* __base_vptr_map_;
};

template <size_t _NbrInterfaces>
struct __rtti_ex : __rtti
{
  template <class _Tp, class _Super, class... _Interfaces, class _VPtr>
  _CUDAX_HOST_API constexpr __rtti_ex(__tag<_Tp, _Super> __type, __tag<_Interfaces...> __ibases, _VPtr __self) noexcept
      : __rtti{__type, __ibases, __base_vptr_array}
      , __base_vptr_array{{&_CCCL_TYPEID(_Interfaces), static_cast<__vptr_for<_Interfaces>>(__self)}...}
  {}

  __base_info __base_vptr_array[_NbrInterfaces];
};

//!
//! __try_vptr_cast
//!
//! This function ignores const qualification on the source and destination
//! interfaces.
//!
template <class _SrcInterface, class _DstInterface>
[[nodiscard]] _CUDAX_HOST_API auto __try_vptr_cast(__vptr_for<_SrcInterface> __src_vptr) noexcept
  -> __vptr_for<_DstInterface>
{
  static_assert(_CUDA_VSTD::is_class_v<_SrcInterface> && _CUDA_VSTD::is_class_v<_DstInterface>, "expected class types");
  if (__src_vptr == nullptr)
  {
    return nullptr;
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_SrcInterface const, _DstInterface const>)
  {
    return __src_vptr;
  }
  else if constexpr (extension_of<_SrcInterface, _DstInterface>)
  {
    //! Fast up-casts:
    return __src_vptr->__query_interface(_DstInterface());
  }
  else
  {
    //! Slow down-casts and cross-casts:
    __rtti const* rtti = __src_vptr->__query_interface(iunknown());
    return rtti->__query_interface(_DstInterface());
  }
}

template <class _SrcInterface, class _DstInterface>
[[nodiscard]] _CUDAX_HOST_API auto __vptr_cast(__vptr_for<_SrcInterface> __src_vptr) //
  noexcept(_CUDA_VSTD::is_same_v<_SrcInterface, _DstInterface>) //
  -> __vptr_for<_DstInterface>
{
  if constexpr (_CUDA_VSTD::is_same_v<_SrcInterface, _DstInterface>)
  {
    return __src_vptr;
  }
  else
  {
    auto __dst_vptr = __try_vptr_cast<_SrcInterface, _DstInterface>(__src_vptr);
    if (!__dst_vptr && __src_vptr)
    {
      __throw_bad_any_cast();
    }
    return __dst_vptr;
  }
  _CCCL_UNREACHABLE();
}

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_RTTI_H
