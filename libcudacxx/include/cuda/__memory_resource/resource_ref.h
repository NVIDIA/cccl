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
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/is_base_of.h>
#  include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#  include <cuda/std/__type_traits/type_set.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cstddef>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

union _AnyResourceStorage
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _AnyResourceStorage(void* __ptr = nullptr) noexcept
      : __ptr_(__ptr)
  {}

  void* __ptr_;
  char __buf_[3 * sizeof(void*)];
};

template <class _Resource>
constexpr bool _IsSmall() noexcept
{
  return (sizeof(_Resource) <= sizeof(_AnyResourceStorage)) //
      && (alignof(_AnyResourceStorage) % alignof(_Resource) == 0)
      && _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Resource);
}

template <class _Resource>
constexpr _Resource* _Any_resource_cast(_AnyResourceStorage* __object) noexcept
{
  return static_cast<_Resource*>(_IsSmall<_Resource>() ? __object->__buf_ : __object->__ptr_);
}

template <class _Resource>
constexpr const _Resource* _Any_resource_cast(const _AnyResourceStorage* __object) noexcept
{
  return static_cast<const _Resource*>(_IsSmall<_Resource>() ? __object->__buf_ : __object->__ptr_);
}

enum class _WrapperType
{
  _Reference,
  _Owning
};

enum class _AllocType
{
  _Default,
  _Async,
};

struct _Alloc_vtable
{
  using _AllocFn   = void* (*) (void*, size_t, size_t);
  using _DeallocFn = void (*)(void*, void*, size_t, size_t) _LIBCUDACXX_FUNCTION_TYPE_NOEXCEPT;
  using _EqualFn   = bool (*)(void*, void*);
  using _DestroyFn = void (*)(_AnyResourceStorage*) _LIBCUDACXX_FUNCTION_TYPE_NOEXCEPT;
  using _MoveFn    = void (*)(_AnyResourceStorage*, _AnyResourceStorage*) _LIBCUDACXX_FUNCTION_TYPE_NOEXCEPT;
  using _CopyFn    = void (*)(_AnyResourceStorage*, const _AnyResourceStorage*);

  bool __is_small;
  _AllocFn __alloc_fn;
  _DeallocFn __dealloc_fn;
  _EqualFn __equal_fn;
  _DestroyFn __destroy_fn;
  _MoveFn __move_fn;
  _CopyFn __copy_fn;

  constexpr _Alloc_vtable(
    bool __is_small_,
    _AllocFn __alloc_fn_,
    _DeallocFn __dealloc_fn_,
    _EqualFn __equal_fn_,
    _DestroyFn __destroy_fn_,
    _MoveFn __move_fn_,
    _CopyFn __copy_fn_) noexcept
      : __is_small(__is_small_)
      , __alloc_fn(__alloc_fn_)
      , __dealloc_fn(__dealloc_fn_)
      , __equal_fn(__equal_fn_)
      , __destroy_fn(__destroy_fn_)
      , __move_fn(__move_fn_)
      , __copy_fn(__copy_fn_)
  {}
};

struct _Async_alloc_vtable : public _Alloc_vtable
{
  using _AsyncAllocFn   = void* (*) (void*, size_t, size_t, ::cuda::stream_ref);
  using _AsyncDeallocFn = void (*)(void*, void*, size_t, size_t, ::cuda::stream_ref);

  _AsyncAllocFn __async_alloc_fn;
  _AsyncDeallocFn __async_dealloc_fn;

  constexpr _Async_alloc_vtable(
    bool __is_small_,
    _Alloc_vtable::_AllocFn __alloc_fn_,
    _Alloc_vtable::_DeallocFn __dealloc_fn_,
    _Alloc_vtable::_EqualFn __equal_fn_,
    _Alloc_vtable::_DestroyFn __destroy_fn_,
    _Alloc_vtable::_MoveFn __move_fn_,
    _Alloc_vtable::_CopyFn __copy_fn_,
    _AsyncAllocFn __async_alloc_fn_,
    _AsyncDeallocFn __async_dealloc_fn_) noexcept
      : _Alloc_vtable(__is_small_, __alloc_fn_, __dealloc_fn_, __equal_fn_, __destroy_fn_, __move_fn_, __copy_fn_)
      , __async_alloc_fn(__async_alloc_fn_)
      , __async_dealloc_fn(__async_dealloc_fn_)
  {}
};

struct _Resource_vtable_builder
{
  template <_WrapperType _Wrapper_type>
  using __wrapper_type = _CUDA_VSTD::integral_constant<_WrapperType, _Wrapper_type>;

  template <class _Resource, class _Property>
  static __property_value_t<_Property> _Get_property(void* __res) noexcept
  {
    return get_property(*static_cast<const _Resource*>(__res), _Property{});
  }

  template <class _Resource>
  static void* _Alloc(void* __object, size_t __bytes, size_t __alignment)
  {
    return static_cast<_Resource*>(__object)->allocate(__bytes, __alignment);
  }

  template <class _Resource>
  static void _Dealloc(void* __object, void* __ptr, size_t __bytes, size_t __alignment) noexcept
  {
    // TODO: this breaks RMM because their memory resources do not declare their
    // deallocate functions to be noexcept. Comment out the check for now until
    // we can fix RMM.
    // static_assert(noexcept(static_cast<_Resource*>(__object)->deallocate(__ptr, __bytes, __alignment)));
    return static_cast<_Resource*>(__object)->deallocate(__ptr, __bytes, __alignment);
  }

  template <class _Resource>
  static void* _Alloc_async(void* __object, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return static_cast<_Resource*>(__object)->allocate_async(__bytes, __alignment, __stream);
  }

  template <class _Resource>
  static void
  _Dealloc_async(void* __object, void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return static_cast<_Resource*>(__object)->deallocate_async(__ptr, __bytes, __alignment, __stream);
  }

  template <class _Resource>
  static bool _Equal(void* __left, void* __rhs)
  {
    return *static_cast<_Resource*>(__left) == *static_cast<_Resource*>(__rhs);
  }

  template <class _Resource>
  static void _Destroy_impl(_AnyResourceStorage* __object_, __wrapper_type<_WrapperType::_Owning>) noexcept
  {
    _Resource* __object = _Any_resource_cast<_Resource>(__object_);
    _CCCL_IF_CONSTEXPR (_IsSmall<_Resource>())
    {
      __object->~_Resource();
    }
    else
    {
      delete __object;
    }
  }

  template <class _Resource>
  static void _Destroy_impl(_AnyResourceStorage*, __wrapper_type<_WrapperType::_Reference>) noexcept
  {}

  template <class _Resource, _WrapperType _Wrapper_type>
  static void _Destroy(_AnyResourceStorage* __object) noexcept
  {
    _Destroy_impl<_Resource>(__object, __wrapper_type<_Wrapper_type>{});
  }

  template <class _Resource>
  static void _Move_impl(
    _AnyResourceStorage* __object, _AnyResourceStorage* __other_, __wrapper_type<_WrapperType::_Owning>) noexcept
  {
    _CCCL_IF_CONSTEXPR (_IsSmall<_Resource>())
    {
      _Resource* __other = _Any_resource_cast<_Resource>(__other_);
      ::new (static_cast<void*>(__object->__buf_)) _Resource(_CUDA_VSTD::move(*__other));
      __other->~_Resource();
    }
    else
    {
      __object->__ptr_ = _CUDA_VSTD::exchange(__other_->__ptr_, nullptr);
    }
  }

  template <class _Resource>
  static void _Move_impl(_AnyResourceStorage*, _AnyResourceStorage*, __wrapper_type<_WrapperType::_Reference>) noexcept
  {}

  template <class _Resource, _WrapperType _Wrapper_type>
  static void _Move(_AnyResourceStorage* __object, _AnyResourceStorage* __other) noexcept
  {
    _Move_impl<_Resource>(__object, __other, __wrapper_type<_Wrapper_type>{});
  }

  template <class _Resource>
  static void _Copy_impl(
    _AnyResourceStorage* __object, const _AnyResourceStorage* __other, __wrapper_type<_WrapperType::_Owning>) noexcept
  {
    _CCCL_IF_CONSTEXPR (_IsSmall<_Resource>())
    {
      ::new (static_cast<void*>(__object->__buf_)) _Resource(*_Any_resource_cast<_Resource>(__other));
    }
    else
    {
      __object->__ptr_ = new _Resource(*_Any_resource_cast<_Resource>(__other));
    }
  }

  template <class _Resource>
  static void
  _Copy_impl(_AnyResourceStorage*, const _AnyResourceStorage*, __wrapper_type<_WrapperType::_Reference>) noexcept
  {}

  template <class _Resource, _WrapperType _Wrapper_type>
  static void _Copy(_AnyResourceStorage* __object, const _AnyResourceStorage* __other)
  {
    _Copy_impl<_Resource>(__object, __other, __wrapper_type<_Wrapper_type>{});
  }

  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type, _WrapperType _Wrapper_type)
  _LIBCUDACXX_REQUIRES((_Alloc_type == _AllocType::_Default))
  static constexpr _Alloc_vtable _Create() noexcept
  {
    return {_IsSmall<_Resource>(),
            &_Resource_vtable_builder::_Alloc<_Resource>,
            &_Resource_vtable_builder::_Dealloc<_Resource>,
            &_Resource_vtable_builder::_Equal<_Resource>,
            &_Resource_vtable_builder::_Destroy<_Resource, _Wrapper_type>,
            &_Resource_vtable_builder::_Move<_Resource, _Wrapper_type>,
            &_Resource_vtable_builder::_Copy<_Resource, _Wrapper_type>};
  }

  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type, _WrapperType _Wrapper_type)
  _LIBCUDACXX_REQUIRES((_Alloc_type == _AllocType::_Async))
  static constexpr _Async_alloc_vtable _Create() noexcept
  {
    return {_IsSmall<_Resource>(),
            &_Resource_vtable_builder::_Alloc<_Resource>,
            &_Resource_vtable_builder::_Dealloc<_Resource>,
            &_Resource_vtable_builder::_Equal<_Resource>,
            &_Resource_vtable_builder::_Destroy<_Resource, _Wrapper_type>,
            &_Resource_vtable_builder::_Move<_Resource, _Wrapper_type>,
            &_Resource_vtable_builder::_Copy<_Resource, _Wrapper_type>,
            &_Resource_vtable_builder::_Alloc_async<_Resource>,
            &_Resource_vtable_builder::_Dealloc_async<_Resource>};
  }
};

template <class _Property>
using __property_fn_t = __property_value_t<_Property> (*)(void*);

template <class _Property>
struct _Property_vtable
{
  __property_fn_t<_Property> __property_fn = nullptr;

  constexpr _Property_vtable(__property_fn_t<_Property> __property_fn_) noexcept
      : __property_fn(__property_fn_)
  {}
};

template <_AllocType _Alloc_type, class... _Properties>
class basic_resource_ref;

template <class... _Properties>
struct _Resource_vtable : public _Property_vtable<_Properties>...
{
  constexpr _Resource_vtable(__property_fn_t<_Properties>... __property_fn_) noexcept
      : _Property_vtable<_Properties>(__property_fn_)...
  {}

  template <class... _OtherProperties>
  constexpr _Resource_vtable(const _Resource_vtable<_OtherProperties...>& __other) noexcept
      : _Property_vtable<_Properties>(__other._Property_vtable<_Properties>::__property_fn)...
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
  using _Filtered_vtable = typename _Property_filter<
    property_with_value<_Property> && !_CUDA_VSTD::__is_included_in<_Property, _Properties...>>::
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

template <_WrapperType _Wrapper_type>
using __alloc_object_storage_t = _CUDA_VSTD::_If<_Wrapper_type == _WrapperType::_Reference, void*, _AnyResourceStorage>;

template <class _Vtable, _WrapperType _Wrapper_type>
struct _Alloc_base
{
  static_assert(_CUDA_VSTD::is_base_of_v<_Alloc_vtable, _Vtable>, "");

  _Alloc_base(void* __object_, const _Vtable* __static_vtabl_) noexcept
      : __object(__object_)
      , __static_vtable(__static_vtabl_)
  {}

  _CCCL_NODISCARD void* allocate(size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __static_vtable->__alloc_fn(_Get_object(), __bytes, __alignment);
  }

  void deallocate(void* _Ptr, size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t)) noexcept
  {
    __static_vtable->__dealloc_fn(_Get_object(), _Ptr, __bytes, __alignment);
  }

protected:
  static _CCCL_FORCEINLINE void* _Get_object_(bool, void* __object) noexcept
  {
    return __object;
  }

  static _CCCL_FORCEINLINE void* _Get_object_(const bool __is_small, const _AnyResourceStorage& __object) noexcept
  {
    const void* __pv = __is_small ? __object.__buf_ : __object.__ptr_;
    return const_cast<void*>(__pv);
  }

  void* _Get_object() const noexcept
  {
    return _Get_object_(__static_vtable->__is_small, this->__object);
  }

  __alloc_object_storage_t<_Wrapper_type> __object{};
  const _Vtable* __static_vtable = nullptr;
};

template <class _Vtable, _WrapperType _Wrapper_type>
struct _Async_alloc_base : public _Alloc_base<_Vtable, _Wrapper_type>
{
  static_assert(_CUDA_VSTD::is_base_of_v<_Async_alloc_vtable, _Vtable>, "");

  _Async_alloc_base(void* __object_, const _Vtable* __static_vtabl_) noexcept
      : _Alloc_base<_Vtable, _Wrapper_type>(__object_, __static_vtabl_)
  {}

  _CCCL_NODISCARD void* allocate_async(size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return this->__static_vtable->__async_alloc_fn(this->_Get_object(), __bytes, __alignment, __stream);
  }

  _CCCL_NODISCARD void* allocate_async(size_t __bytes, ::cuda::stream_ref __stream)
  {
    return this->__static_vtable->__async_alloc_fn(this->_Get_object(), __bytes, alignof(max_align_t), __stream);
  }

  void deallocate_async(void* _Ptr, size_t __bytes, ::cuda::stream_ref __stream)
  {
    this->__static_vtable->__async_dealloc_fn(this->_Get_object(), _Ptr, __bytes, alignof(max_align_t), __stream);
  }

  void deallocate_async(void* _Ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    this->__static_vtable->__async_dealloc_fn(this->_Get_object(), _Ptr, __bytes, __alignment, __stream);
  }
};

template <class _VTable, _WrapperType _Wrapper_type>
constexpr bool _Is_resource_base_fn(const _Alloc_base<_VTable, _Wrapper_type>*) noexcept
{
  return true;
}

constexpr bool _Is_resource_base_fn(...) noexcept
{
  return false;
}

template <class _Resource>
_LIBCUDACXX_CONCEPT _Is_resource_base = _Is_resource_base_fn(static_cast<_Resource*>(nullptr));

template <_AllocType _Alloc_type, _WrapperType _Wrapper_type>
using _Resource_base =
  _CUDA_VSTD::_If<_Alloc_type == _AllocType::_Default,
                  _Alloc_base<_Alloc_vtable, _Wrapper_type>,
                  _Async_alloc_base<_Async_alloc_vtable, _Wrapper_type>>;

template <_AllocType _Alloc_type>
using _Vtable_store = _CUDA_VSTD::_If<_Alloc_type == _AllocType::_Default, _Alloc_vtable, _Async_alloc_vtable>;

template <_AllocType _Alloc_type, _WrapperType _Wrapper_type, class _Resource>
_LIBCUDACXX_INLINE_VAR constexpr _Vtable_store<_Alloc_type> __alloc_vtable =
  _Resource_vtable_builder::template _Create<_Resource, _Alloc_type, _Wrapper_type>();

struct _Resource_ref_helper
{
  //! This is used from \c basic_any_resource to make it convertible to a \c basic_resource_ref
  template <_AllocType _Alloc_type, class... _Properties>
  static basic_resource_ref<_Alloc_type, _Properties...>
  _Construct(void* __object,
             const _Vtable_store<_Alloc_type>* __static_vtable,
             _Filtered_vtable<_Properties...> __properties) noexcept
  {
    return basic_resource_ref<_Alloc_type, _Properties...>(__object, __static_vtable, __properties);
  }
};

template <_AllocType _Alloc_type, class... _Properties>
class basic_resource_ref
    : public _Resource_base<_Alloc_type, _WrapperType::_Reference>
    , private _Filtered_vtable<_Properties...>
{
private:
  static_assert(__contains_execution_space_property<_Properties...>,
                "The properties of cuda::mr::basic_resource_ref must contain at least one execution space property!");

  template <_AllocType, class...>
  friend class basic_resource_ref;

  template <class...>
  friend struct _Resource_vtable;

  friend struct _Resource_ref_helper;

  using __vtable = _Filtered_vtable<_Properties...>;

  //! @brief Checks whether \c _OtherProperties is a true superset of \c _Properties, accounting for host_accessible
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    _CUDA_VSTD::__type_set_contains<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Constructs a \c basic_resource_ref from a void*, a resource vtable ptr, and a vtable
  //! for the properties. This is used to create a \c basic_resource_ref from a \c basic_any_resource.
  explicit basic_resource_ref(
    void* __object_, const _Vtable_store<_Alloc_type>* __static_vtable, __vtable __properties) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(__object_, __static_vtable)
      , __vtable(__properties)
  {}

public:
  //! @brief Constructs a \c basic_resource_ref from a type that satisfies the \c resource or \c async_resource concept
  //! as well as all properties
  //! @param __res The resource to be wrapped within the \c basic_resource_ref
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_resource_base<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Default)
                         _LIBCUDACXX_AND resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource& __res) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(
          _CUDA_VSTD::addressof(__res), &__alloc_vtable<_Alloc_type, _WrapperType::_Reference, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Constructs a \c resource_ref from a type that satisfies the \c async_resource concept  as well as all
  //! properties. This ignores the async interface of the passed in resource
  //! @param __res The resource to be wrapped within the \c resource_ref
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_resource_base<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Async)
                         _LIBCUDACXX_AND async_resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource& __res) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(
          _CUDA_VSTD::addressof(__res), &__alloc_vtable<_Alloc_type, _WrapperType::_Reference, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Constructs a \c basic_resource_ref from a type that satisfies the \c resource or \c async_resource concept
  //! as well as all properties
  //! @param __res Pointer to a resource to be wrapped within the \c basic_resource_ref
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_resource_base<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Default)
                         _LIBCUDACXX_AND resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource* __res) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(
          __res, &__alloc_vtable<_Alloc_type, _WrapperType::_Reference, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Constructs a \c resource_ref from a type that satisfies the \c async_resource concept  as well as all
  //! properties. This ignores the async interface of the passed in resource
  //! @param __res Pointer to a resource to be wrapped within the \c resource_ref
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!_Is_resource_base<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Async)
                         _LIBCUDACXX_AND async_resource_with<_Resource, _Properties...>)
  basic_resource_ref(_Resource* __res) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(
          __res, &__alloc_vtable<_Alloc_type, _WrapperType::_Reference, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Conversion from a \c basic_resource_ref with the same set of properties but in a different order
  //! @param __ref The other \c basic_resource_ref
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES(__properties_match<_OtherProperties...>)
  basic_resource_ref(basic_resource_ref<_Alloc_type, _OtherProperties...> __ref) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(__ref.__object, __ref.__static_vtable)
      , __vtable(static_cast<const _Filtered_vtable<_OtherProperties...>&>(__ref))
  {}

  //! @brief Conversion from a \c async_resource_ref with the same set of properties but in a different order to a
  //! \c resource_ref
  //! @param __ref The other \c async_resource_ref
  _LIBCUDACXX_TEMPLATE(_AllocType _OtherAllocType, class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((_OtherAllocType == _AllocType::_Async) _LIBCUDACXX_AND(_OtherAllocType != _Alloc_type)
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  basic_resource_ref(basic_resource_ref<_OtherAllocType, _OtherProperties...> __ref) noexcept
      : _Resource_base<_Alloc_type, _WrapperType::_Reference>(__ref.__object, __ref.__static_vtable)
      , __vtable(static_cast<const _Filtered_vtable<_OtherProperties...>&>(__ref))
  {}

  //! @brief Equality comparison between two \c basic_resource_ref
  //! @param __rhs The other \c basic_resource_ref
  //! @return Checks whether both resources have the same equality function stored in their vtable and if so returns
  //! the result of that equality comparison. Otherwise returns false.
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator==(const basic_resource_ref<_Alloc_type, _OtherProperties...>& __rhs) const
  {
    // BUGBUG: comparing function pointers like this can lead to false negatives:
    return (this->__static_vtable->__equal_fn == __rhs.__static_vtable->__equal_fn)
        && this->__static_vtable->__equal_fn(this->__object, __rhs.__object);
  }

  //! @brief Inequality comparison between two \c basic_resource_ref
  //! @param __rhs The other \c basic_resource_ref
  //! @return Checks whether both resources have the same equality function stored in their vtable and if so returns
  //! the inverse result of that equality comparison. Otherwise returns true.
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator!=(const basic_resource_ref<_Alloc_type, _OtherProperties...>& __rhs) const
  {
    return !(*this == __rhs);
  }

  //! @brief Forwards the stateless properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(
    (!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::__is_included_in<_Property, _Properties...>)
  friend void get_property(const basic_resource_ref&, _Property) noexcept {}

  //! @brief Forwards the stateful properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(
    property_with_value<_Property> _LIBCUDACXX_AND _CUDA_VSTD::__is_included_in<_Property, _Properties...>)
  _CCCL_NODISCARD_FRIEND __property_value_t<_Property> get_property(const basic_resource_ref& __res, _Property) noexcept
  {
    return __res._Property_vtable<_Property>::__property_fn(__res.__object);
  }
};

//! @brief Type erased wrapper around a `resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any resource wrapped within the `resource_ref` needs to satisfy
template <class... _Properties>
using resource_ref = basic_resource_ref<_AllocType::_Default, _Properties...>;

//! @brief Type erased wrapper around a `async_resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any async resource wrapped within the `async_resource_ref` needs to satisfy
template <class... _Properties>
using async_resource_ref = basic_resource_ref<_AllocType::_Async, _Properties...>;

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_RESOURCE_REF_H
