//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_RESOURCE_REF_H
#define _CUDA__MEMORY_RESOURCE_RESOURCE_REF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/std/__concepts/_One_of.h>
#  include <cuda/std/__concepts/all_of.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/is_base_of.h>
#  include <cuda/std/cstddef>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

enum class _AllocType
{
  _Default,
  _Async,
};

struct _Alloc_vtable
{
  using _AllocFn   = void* (*) (void*, size_t, size_t);
  using _DeallocFn = void (*)(void*, void*, size_t, size_t);
  using _EqualFn   = bool (*)(void*, void*);

  _AllocFn __alloc_fn;
  _DeallocFn __dealloc_fn;
  _EqualFn __equal_fn;

  constexpr _Alloc_vtable(_AllocFn __alloc_fn_, _DeallocFn __dealloc_fn_, _EqualFn __equal_fn_) noexcept
      : __alloc_fn(__alloc_fn_)
      , __dealloc_fn(__dealloc_fn_)
      , __equal_fn(__equal_fn_)
  {}
};

struct _Async_alloc_vtable : public _Alloc_vtable
{
  using _AsyncAllocFn   = void* (*) (void*, size_t, size_t, ::cuda::stream_ref);
  using _AsyncDeallocFn = void (*)(void*, void*, size_t, size_t, ::cuda::stream_ref);

  _AsyncAllocFn __async_alloc_fn;
  _AsyncDeallocFn __async_dealloc_fn;

  constexpr _Async_alloc_vtable(
    _Alloc_vtable::_AllocFn __alloc_fn_,
    _Alloc_vtable::_DeallocFn __dealloc_fn_,
    _Alloc_vtable::_EqualFn __equal_fn_,
    _AsyncAllocFn __async_alloc_fn_,
    _AsyncDeallocFn __async_dealloc_fn_) noexcept
      : _Alloc_vtable(__alloc_fn_, __dealloc_fn_, __equal_fn_)
      , __async_alloc_fn(__async_alloc_fn_)
      , __async_dealloc_fn(__async_dealloc_fn_)
  {}
};

// clang-format off
struct _Resource_vtable_builder
{
    template <class _Resource, class _Property>
    static __property_value_t<_Property> _Get_property(void* __res) noexcept {
        return get_property(*static_cast<const _Resource *>(__res), _Property{});
    }

    template <class _Resource>
    static void* _Alloc(void* __object, size_t __bytes, size_t __alignment) {
        return static_cast<_Resource *>(__object)->allocate(__bytes, __alignment);
    }

    template <class _Resource>
    static void _Dealloc(void* __object, void* __ptr, size_t __bytes, size_t __alignment) {
        return static_cast<_Resource *>(__object)->deallocate(__ptr, __bytes, __alignment);
    }

    template <class _Resource>
    static void* _Alloc_async(void* __object, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream) {
        return static_cast<_Resource *>(__object)->allocate_async(__bytes, __alignment, __stream);
    }

    template <class _Resource>
    static void _Dealloc_async(void* __object, void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream) {
        return static_cast<_Resource *>(__object)->deallocate_async(__ptr, __bytes, __alignment, __stream);
    }

    template <class _Resource>
    static bool _Equal(void* __left, void* __right) {
        return *static_cast<_Resource *>(__left) == *static_cast<_Resource *>(__right);
    }

    _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type)
      _LIBCUDACXX_REQUIRES((_Alloc_type == _AllocType::_Default))
     static constexpr _Alloc_vtable _Create() noexcept
    {
      return {&_Resource_vtable_builder::_Alloc<_Resource>,
              &_Resource_vtable_builder::_Dealloc<_Resource>,
              &_Resource_vtable_builder::_Equal<_Resource>};
    }

    _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type)
      _LIBCUDACXX_REQUIRES((_Alloc_type == _AllocType::_Async))
     static constexpr _Async_alloc_vtable _Create() noexcept
    {
      return {&_Resource_vtable_builder::_Alloc<_Resource>,
              &_Resource_vtable_builder::_Dealloc<_Resource>,
              &_Resource_vtable_builder::_Equal<_Resource>,
              &_Resource_vtable_builder::_Alloc_async<_Resource>,
              &_Resource_vtable_builder::_Dealloc_async<_Resource>};
    }
};
// clang-format on

template <class _Property>
struct _Property_vtable
{
  using _PropertyFn         = __property_value_t<_Property> (*)(void*);
  _PropertyFn __property_fn = nullptr;

  constexpr _Property_vtable(_PropertyFn __property_fn_) noexcept
      : __property_fn(__property_fn_)
  {}
};

template <_AllocType _Alloc_type, class... _Properties>
class basic_resource_ref;

template <class... _Properties>
struct _Resource_vtable : public _Property_vtable<_Properties>...
{
  template <class... _PropertyFns>
  constexpr _Resource_vtable(_PropertyFns... __property_fn_) noexcept
      : _Property_vtable<_Properties>(__property_fn_)...
  {}

  template <_AllocType _Alloc_type, class... _OtherProperties>
  constexpr _Resource_vtable(basic_resource_ref<_Alloc_type, _OtherProperties...>& __ref) noexcept
      : _Property_vtable<_Properties>(__ref._Property_vtable<_Properties>::__property_fn)...
  {}

  template <class _Resource>
  static constexpr _Resource_vtable _Create() noexcept
  {
    return {&_Resource_vtable_builder::_Get_property<_Resource, _Properties>...};
  }
};

template <class... _Properties>
struct _Filtered;

template <bool _IsUniqueProperty>
struct _Property_filter
{
  template <class _Property, class... _Properties>
  using _Filtered_properties =
    typename _Filtered<_Properties...>::_Filtered_vtable::template _Append_property<_Property>;
};
template <>
struct _Property_filter<false>
{
  template <class _Property, class... _Properties>
  using _Filtered_properties = typename _Filtered<_Properties...>::_Filtered_vtable;
};

template <class _Property, class... _Properties>
struct _Filtered<_Property, _Properties...>
{
  using _Filtered_vtable =
    typename _Property_filter<property_with_value<_Property> && !_CUDA_VSTD::_One_of<_Property, _Properties...>>::
      template _Filtered_properties<_Property, _Properties...>;

  template <class _OtherPropery>
  using _Append_property = _Filtered<_OtherPropery, _Property, _Properties...>;

  using _Vtable = _Resource_vtable<_Property, _Properties...>;
};

template <>
struct _Filtered<>
{
  using _Filtered_vtable = _Filtered<>;

  template <class _OtherPropery>
  using _Append_property = _Filtered<_OtherPropery>;

  using _Vtable = _Resource_vtable<>;
};

template <class... _Properties>
using _Filtered_vtable = typename _Filtered<_Properties...>::_Filtered_vtable::_Vtable;

template <class _Vtable>
struct _Alloc_base
{
  static_assert(_CUDA_VSTD::is_base_of_v<_Alloc_vtable, _Vtable>, "");

  _Alloc_base(void* __object_, const _Vtable* __static_vtabl_) noexcept
      : __object(__object_)
      , __static_vtable(__static_vtabl_)
  {}

  _CCCL_NODISCARD void* allocate(size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __static_vtable->__alloc_fn(__object, __bytes, __alignment);
  }

  void deallocate(void* _Ptr, size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    __static_vtable->__dealloc_fn(__object, _Ptr, __bytes, __alignment);
  }

protected:
  void* __object                 = nullptr;
  const _Vtable* __static_vtable = nullptr;
};

template <class _Vtable>
struct _Async_alloc_base : public _Alloc_base<_Vtable>
{
  static_assert(_CUDA_VSTD::is_base_of_v<_Async_alloc_vtable, _Vtable>, "");

  _Async_alloc_base(void* __object_, const _Vtable* __static_vtabl_) noexcept
      : _Alloc_base<_Vtable>(__object_, __static_vtabl_)
  {}

  _CCCL_NODISCARD void* allocate_async(size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return this->__static_vtable->__async_alloc_fn(this->__object, __bytes, __alignment, __stream);
  }

  _CCCL_NODISCARD void* allocate_async(size_t __bytes, ::cuda::stream_ref __stream)
  {
    return this->__static_vtable->__async_alloc_fn(this->__object, __bytes, alignof(max_align_t), __stream);
  }

  void deallocate_async(void* _Ptr, size_t __bytes, ::cuda::stream_ref __stream)
  {
    this->__static_vtable->__async_dealloc_fn(this->__object, _Ptr, __bytes, alignof(max_align_t), __stream);
  }

  void deallocate_async(void* _Ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    this->__static_vtable->__async_dealloc_fn(this->__object, _Ptr, __bytes, __alignment, __stream);
  }
};

template <_AllocType _Alloc_type>
using _Resource_ref_base = _CUDA_VSTD::
  _If<_Alloc_type == _AllocType::_Default, _Alloc_base<_Alloc_vtable>, _Async_alloc_base<_Async_alloc_vtable>>;

template <_AllocType _Alloc_type>
using _Vtable_store = _CUDA_VSTD::_If<_Alloc_type == _AllocType::_Default, _Alloc_vtable, _Async_alloc_vtable>;

template <_AllocType _Alloc_type, class _Resource>
_LIBCUDACXX_INLINE_VAR constexpr _Vtable_store<_Alloc_type> __alloc_vtable =
  _Resource_vtable_builder::template _Create<_Resource, _Alloc_type>();

template <class>
_LIBCUDACXX_INLINE_VAR constexpr bool _Is_basic_resource_ref = false;

template <_AllocType _Alloc_type, class... _Properties>
class basic_resource_ref
    : public _Resource_ref_base<_Alloc_type>
    , private _Filtered_vtable<_Properties...>
{
private:
  template <_AllocType, class...>
  friend class basic_resource_ref;

  template <class...>
  friend struct _Resource_vtable;

  using __vtable = _Filtered_vtable<_Properties...>;

  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    _CUDA_VSTD::__all_of<_CUDA_VSTD::_One_of<_Properties, _OtherProperties...>...>;

public:
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_basic_resource_ref<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Default)
                         _LIBCUDACXX_AND resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource& __res) noexcept
      : _Resource_ref_base<_Alloc_type>(_CUDA_VSTD::addressof(__res), &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_basic_resource_ref<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Async)
                         _LIBCUDACXX_AND async_resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource& __res) noexcept
      : _Resource_ref_base<_Alloc_type>(_CUDA_VSTD::addressof(__res), &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_basic_resource_ref<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Default)
                         _LIBCUDACXX_AND resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource* __res) noexcept
      : _Resource_ref_base<_Alloc_type>(__res, &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_basic_resource_ref<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Async)
                         _LIBCUDACXX_AND async_resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource* __res) noexcept
      : _Resource_ref_base<_Alloc_type>(__res, &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES(__properties_match<_OtherProperties...>)
  basic_resource_ref(basic_resource_ref<_Alloc_type, _OtherProperties...> __ref) noexcept
      : _Resource_ref_base<_Alloc_type>(__ref.__object, __ref.__static_vtable)
      , __vtable(__ref)
  {}

  _LIBCUDACXX_TEMPLATE(_AllocType _OtherAllocType, class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((_OtherAllocType == _AllocType::_Async) _LIBCUDACXX_AND(_OtherAllocType != _Alloc_type)
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  basic_resource_ref(basic_resource_ref<_OtherAllocType, _OtherProperties...> __ref) noexcept
      : _Resource_ref_base<_Alloc_type>(__ref.__object, __ref.__static_vtable)
      , __vtable(__ref)
  {}

  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator==(const basic_resource_ref<_Alloc_type, _OtherProperties...>& __right) const
  {
    return (this->__static_vtable->__equal_fn == __right.__static_vtable->__equal_fn) //
        && this->__static_vtable->__equal_fn(this->__object, __right.__object);
  }

  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator!=(const basic_resource_ref<_Alloc_type, _OtherProperties...>& __right) const
  {
    return !(*this == __right);
  }

  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(
    (!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>) //
  friend void get_property(const basic_resource_ref&, _Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(property_with_value<_Property> _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>) //
  _CCCL_NODISCARD_FRIEND __property_value_t<_Property> get_property(const basic_resource_ref& __res, _Property) noexcept
  {
    return __res._Property_vtable<_Property>::__property_fn(__res.__object);
  }
};

template <_AllocType _Alloc_type, class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr bool _Is_basic_resource_ref<basic_resource_ref<_Alloc_type, _Properties...>> = true;

template <class... _Properties> //
using resource_ref = basic_resource_ref<_AllocType::_Default, _Properties...>;

template <class... _Properties> //
using async_resource_ref = basic_resource_ref<_AllocType::_Async, _Properties...>;

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_RESOURCE_REF_H
