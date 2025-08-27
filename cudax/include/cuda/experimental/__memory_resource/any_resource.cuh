//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
#define _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/__utility/basic_any.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/experimental/__memory_resource/properties.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document this

template <class _Property>
using __property_result_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_call1< //
  ::cuda::std::conditional_t<cuda::property_with_value<_Property>,
                             ::cuda::std::__type_quote1<__property_value_t>,
                             ::cuda::std::__type_always<void>>,
  _Property>;

template <class _Property>
struct __with_property
{
  template <class _Ty>
  _CCCL_PUBLIC_HOST_API static auto __get_property(const _Ty& __obj) //
    -> __property_result_t<_Property>
  {
    if constexpr (!::cuda::std::is_same_v<__property_result_t<_Property>, void>)
    {
      return get_property(__obj, _Property());
    }
    else
    {
      return void();
    }
  }

  template <class...>
  struct __iproperty : __basic_interface<__iproperty>
  {
    _CCCL_HOST_API friend auto get_property([[maybe_unused]] const __iproperty& __obj, _Property)
      -> __property_result_t<_Property>
    {
      if constexpr (!::cuda::std::is_same_v<__property_result_t<_Property>, void>)
      {
        return ::cuda::__virtcall<&__get_property<__iproperty>>(&__obj);
      }
      else
      {
        return void();
      }
    }

    template <class _Ty>
    using overrides _CCCL_NODEBUG_ALIAS = __overrides_for<_Ty, &__get_property<_Ty>>;
  };
};

template <class _Property>
using __iproperty = typename __with_property<_Property>::template __iproperty<>;

template <class... _Properties>
using __iproperty_set = ::cuda::__iset<__iproperty<_Properties>...>;

// Wrap the calls of the allocate and deallocate member functions
// because of NVBUG#4967486
// Needs to keep the _async because of fun windows macros
template <class _Resource>
_CCCL_PUBLIC_HOST_API auto
__allocate_async(_Resource& __mr, ::cuda::stream_ref __stream, size_t __bytes, size_t __alignment)
  -> decltype(__mr.allocate(__stream, __bytes, __alignment))
{
  return __mr.allocate(__stream, __bytes, __alignment);
}

template <class _Resource>
_CCCL_PUBLIC_HOST_API auto
__deallocate_async(_Resource& __mr, ::cuda::stream_ref __stream, void* __pv, size_t __bytes, size_t __alignment)
  -> decltype(__mr.deallocate(__stream, __pv, __bytes, __alignment))
{
  __mr.deallocate(__stream, __pv, __bytes, __alignment);
}

template <class...>
struct __ibasic_resource : __basic_interface<__ibasic_resource>
{
  _CCCL_PUBLIC_HOST_API void* allocate_sync(size_t __bytes, size_t __alignment = alignof(::cuda::std::max_align_t))
  {
    return ::cuda::__virtcall<&__ibasic_resource::allocate_sync>(this, __bytes, __alignment);
  }

  _CCCL_PUBLIC_HOST_API void

  deallocate_sync(void* __pv, size_t __bytes, size_t __alignment = alignof(::cuda::std::max_align_t))
  {
    return ::cuda::__virtcall<&__ibasic_resource::deallocate_sync>(this, __pv, __bytes, __alignment);
  }

  template <class _Ty>
  using overrides _CCCL_NODEBUG_ALIAS = __overrides_for<_Ty, &_Ty::allocate_sync, &_Ty::deallocate_sync>;
};

template <class...>
struct __ibasic_async_resource : __basic_interface<__ibasic_async_resource>
{
  _CCCL_PUBLIC_HOST_API void* allocate(::cuda::stream_ref __stream, size_t __bytes, size_t __alignment)
  {
    return ::cuda::__virtcall<&__allocate_async<__ibasic_async_resource>>(this, __stream, __bytes, __alignment);
  }

  _CCCL_PUBLIC_HOST_API void* allocate(::cuda::stream_ref __stream, size_t __bytes)
  {
    return ::cuda::__virtcall<&__allocate_async<__ibasic_async_resource>>(
      this, __stream, __bytes, alignof(::cuda::std::max_align_t));
  }

  _CCCL_PUBLIC_HOST_API void deallocate(::cuda::stream_ref __stream, void* __pv, size_t __bytes, size_t __alignment)
  {
    return ::cuda::__virtcall<&__deallocate_async<__ibasic_async_resource>>(this, __stream, __pv, __bytes, __alignment);
  }

  _CCCL_PUBLIC_HOST_API void deallocate(::cuda::stream_ref __stream, void* __pv, size_t __bytes)
  {
    return ::cuda::__virtcall<&__deallocate_async<__ibasic_async_resource>>(
      this, __stream, __pv, __bytes, alignof(::cuda::std::max_align_t));
  }

  template <class _Ty>
  using overrides _CCCL_NODEBUG_ALIAS = __overrides_for<_Ty, &__allocate_async<_Ty>, &__deallocate_async<_Ty>>;
};

// This is the pseudo-virtual override for getting an old-style vtable pointer
// from a new-style __basic_any resource type. It is used below by
// __iresource_ref_conversions.
template <class _Resource>
_CCCL_PUBLIC_HOST_API const ::cuda::mr::_Alloc_vtable* __get_resource_vptr(_Resource&) noexcept
{
  if constexpr (::cuda::mr::resource<_Resource>)
  {
    return &::cuda::mr::__alloc_vtable<::cuda::mr::_AllocType::_Default, ::cuda::mr::_WrapperType::_Reference, _Resource>;
  }
  else if constexpr (::cuda::mr::synchronous_resource<_Resource>)
  {
    return &::cuda::mr::
      __alloc_vtable<::cuda::mr::_AllocType::_Synchronous, ::cuda::mr::_WrapperType::_Reference, _Resource>;
  }
  else
  {
    // This branch is taken when called from the thunk of an unspecialized
    // interface; e.g., `icat<>` rather than `icat<ialley_cat<>>`. The thunks of
    // unspecialized interfaces are never called, they just need to exist. The
    // function pointer will be used as a key to look up the proper override.
    _CCCL_UNREACHABLE();
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-but-set-parameter")

// Given a list of properties and a __basic_any vptr, build a _Resource_vtable
// for the properties as cuda::mr::basic_resource_ref expects.
template <class _VPtr, class... _Properties>
_CCCL_HOST_API auto __make_resource_vtable(_VPtr __vptr, ::cuda::mr::_Resource_vtable<_Properties...>*) noexcept
  -> ::cuda::mr::_Resource_vtable<_Properties...>
{
  return {__vptr->__query_interface(__iproperty<_Properties>())->__fn_...};
}

_CCCL_DIAG_POP

// This interface provides the any_[async_]resource types with a conversion
// to the old cuda::mr::basic_resource_ref types.
template <class... _Super>
struct _CCCL_DECLSPEC_EMPTY_BASES __iresource_ref_conversions
    : __basic_interface<__iresource_ref_conversions>
    , ::cuda::mr::_Resource_ref_base
{
  using __self_t = __basic_any_from_t<__iresource_ref_conversions&>;

  template <class _Property>
  using __iprop = __rebind_interface<__iproperty<_Property>, _Super...>;

  template <::cuda::mr::_AllocType _Alloc_type>
  using __iresource = __rebind_interface<
    ::cuda::std::
      conditional_t<_Alloc_type == ::cuda::mr::_AllocType::_Synchronous, __ibasic_resource<>, __ibasic_async_resource<>>,
    _Super...>;

  _CCCL_TEMPLATE(::cuda::mr::_AllocType _Alloc_type, class... _Properties)
  _CCCL_REQUIRES(::cuda::std::derived_from<__self_t, __iresource<_Alloc_type>>
                 && (::cuda::std::derived_from<__self_t, __iprop<_Properties>> && ...))
  operator ::cuda::mr::basic_resource_ref<_Alloc_type, _Properties...>()
  {
    auto& __self = ::cuda::__basic_any_from(*this);
    auto* __vptr = ::cuda::__virtcall<&__get_resource_vptr<__iresource_ref_conversions>>(this);
    auto* __vtag = static_cast<::cuda::mr::_Filtered_vtable<_Properties...>*>(nullptr);
    auto __props = ::cuda::experimental::__make_resource_vtable(__basic_any_access::__get_vptr(__self), __vtag);

    return ::cuda::mr::_Resource_ref_helper::_Construct<_Alloc_type, _Properties...>(
      __basic_any_access::__get_optr(__self),
      static_cast<const ::cuda::mr::_Vtable_store<_Alloc_type>*>(__vptr),
      __props);
  }

  template <class _Resource>
  using overrides = __overrides_for<_Resource, &__get_resource_vptr<_Resource>>;
};

template <class... _Properties>
using __iresource _CCCL_NODEBUG_ALIAS =
  ::cuda::__iset<__ibasic_resource<>,
                 __iproperty_set<_Properties...>,
                 __iresource_ref_conversions<>,
                 ::cuda::__icopyable<>,
                 ::cuda::__iequality_comparable<>>;

template <class... _Properties>
using __iasync_resource _CCCL_NODEBUG_ALIAS = __iset<__iresource<_Properties...>, __ibasic_async_resource<>>;

template <class _Property>
using __try_property_result_t =
  ::cuda::std::conditional_t<!::cuda::std::is_same_v<__property_result_t<_Property>, void>, //
                             ::cuda::std::optional<__property_result_t<_Property>>, //
                             bool>;

template <class _Derived>
struct __with_try_get_property
{
  template <class _Property>
  [[nodiscard]] _CCCL_HOST_API friend auto try_get_property(const _Derived& __self, _Property) noexcept
    -> __try_property_result_t<_Property>
  {
    auto __prop = ::cuda::__dynamic_any_cast<const __iproperty<_Property>*>(&__self);
    if constexpr (::cuda::std::is_same_v<__property_result_t<_Property>, void>)
    {
      return __prop != nullptr;
    }
    else if (__prop)
    {
      return get_property(*__prop, _Property{});
    }
    else
    {
      return ::cuda::std::nullopt;
    }
  }
};

template <class... _Properties>
struct _CCCL_DECLSPEC_EMPTY_BASES any_resource;

template <class... _Properties>
struct _CCCL_DECLSPEC_EMPTY_BASES resource_ref;

// `any_synchronous_resource` wraps any given resource that satisfies the required
// properties. It owns the contained resource, taking care of construction /
// destruction. This makes it especially suited for use in e.g. container types
// that need to ensure that the lifetime of the container exceeds the lifetime
// of the memory resource used to allocate the storage
template <class... _Properties>
struct _CCCL_DECLSPEC_EMPTY_BASES any_synchronous_resource
    : __basic_any<__iresource<_Properties...>>
    , __with_try_get_property<any_synchronous_resource<_Properties...>>
{
  // Inherit constructors from __basic_any
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(
    any_synchronous_resource, ::cuda::__basic_any, experimental::__iresource<_Properties...>);

  // any_resource is convertible to any_synchronous_resource
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES((::cuda::std::__type_set_contains_v<::cuda::std::__type_set<_OtherProperties...>, _Properties...>) )
  any_synchronous_resource(experimental::any_resource<_OtherProperties...> __other) noexcept
      : __base(::cuda::std::move(__other.__get_base()))
  {}

  using default_queries = properties_list<_Properties...>;

private:
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::any_synchronous_resource must contain at least one execution "
                "space "
                "property!");
  using __base::interface;
};

// ``any_resource`` wraps any given resource that satisfies the
// required properties. It owns the contained resource, taking care of
// construction / destruction. This makes it especially suited for use in e.g.
// container types that need to ensure that the lifetime of the container
// exceeds the lifetime of the memory resource used to allocate the storage
template <class... _Properties>
struct _CCCL_DECLSPEC_EMPTY_BASES any_resource
    : __basic_any<__iasync_resource<_Properties...>>
    , __with_try_get_property<any_resource<_Properties...>>
{
  // Inherit constructors from __basic_any
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(any_resource, ::cuda::__basic_any, experimental::__iasync_resource<_Properties...>);

  using default_queries = properties_list<_Properties...>;

private:
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::any_resource must contain at least one execution space "
                "property!");

  template <class...>
  friend struct any_synchronous_resource;

  using __base::interface;

  __base& __get_base() noexcept
  {
    return *this;
  }
};

//! @brief Type erased wrapper around a `synchronous_resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any resource wrapped within the `synchronous_resource_ref` needs to satisfy
template <class... _Properties>
struct _CCCL_DECLSPEC_EMPTY_BASES synchronous_resource_ref
    : __basic_any<__iresource<_Properties...>&>
    , __with_try_get_property<synchronous_resource_ref<_Properties...>>
{
  // Inherit constructors from __basic_any
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(
    synchronous_resource_ref, ::cuda::__basic_any, experimental::__iresource<_Properties...>&);

  // resource_ref is convertible to synchronous_resource_ref
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES((::cuda::std::__type_set_contains_v<::cuda::std::__type_set<_OtherProperties...>, _Properties...>) )
  synchronous_resource_ref(experimental::resource_ref<_OtherProperties...> __other) noexcept
      : __base(__other.__get_base())
  {}

  // Conversions from the synchronous_resource_ref types in cuda::mr is not supported.
  template <class... _OtherProperties>
  synchronous_resource_ref(::cuda::mr::synchronous_resource_ref<_OtherProperties...>) = delete;

  template <class... _OtherProperties>
  synchronous_resource_ref(::cuda::mr::resource_ref<_OtherProperties...>) = delete;

  using default_queries = properties_list<_Properties...>;

private:
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::synchronous_resource_ref must contain at least one execution "
                "space "
                "property!");
  using __base::interface;
};

//! @brief Type erased wrapper around a `synchronous_resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any async resource wrapped within the `resource_ref` needs to satisfy
template <class... _Properties>
struct _CCCL_DECLSPEC_EMPTY_BASES resource_ref
    : __basic_any<__iasync_resource<_Properties...>&>
    , __with_try_get_property<resource_ref<_Properties...>>
{
  // Conversions from the synchronous_resource_ref types in cuda::mr is not supported.
  template <class... _OtherProperties>
  resource_ref(::cuda::mr::resource_ref<_OtherProperties...>) = delete;

  // Inherit other constructors from __basic_any
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(resource_ref, ::cuda::__basic_any, experimental::__iasync_resource<_Properties...>&);

  using default_queries = properties_list<_Properties...>;

private:
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::resource_ref must contain at least one execution space "
                "property!");

  template <class...>
  friend struct synchronous_resource_ref;

  using __base::interface;

  __base& __get_base() noexcept
  {
    return *this;
  }
};

_CCCL_TEMPLATE(class... _Properties, class _Resource)
_CCCL_REQUIRES(mr::synchronous_resource_with<_Resource, _Properties...>)
synchronous_resource_ref<_Properties...> __as_resource_ref(_Resource& __mr) noexcept
{
  return synchronous_resource_ref<_Properties...>(__mr);
}

template <class... _Properties>
synchronous_resource_ref<_Properties...> __as_resource_ref(synchronous_resource_ref<_Properties...> const __mr) noexcept
{
  return __mr;
}

template <class... _Properties>
synchronous_resource_ref<_Properties...> __as_resource_ref(resource_ref<_Properties...> const __mr) noexcept
{
  return __mr;
}

template <class... _Properties, mr::_AllocType _Alloc_type>
mr::synchronous_resource_ref<_Properties...>
__as_resource_ref(mr::basic_resource_ref<_Alloc_type, _Properties...> const __mr) noexcept
{
  return __mr;
}

#else // ^^^ !_CCCL_DOXYGEN_INVOKED ^^^ / vvv _CCCL_DOXYGEN_INVOKED vvv

enum class _ResourceKind
{
  _Synchronous,
  _Asynchronous
};

//! @rst
//! Type erased wrapper around a `resource` or a `synchronous_resource`
//! --------------------------------------------------------------------
//!
//! ``basic_any_resource`` wraps any given :ref:`resource
//! <libcudacxx-extended-api-memory-resources-resource>` that satisfies the
//! required properties. It owns the contained resource, taking care of
//! construction / destruction. This makes it especially suited for use in e.g.
//! container types that need to ensure that the lifetime of the container
//! exceeds the lifetime of the memory resource used to allocate the storage
//!
//! ``basic_any_resource`` models the ``cuda::std::regular`` concept.
//! @endrst
//!
//! @tparam _Kind Either `_ResourceKind::_Synchronous` for `any_synchronous_resource`, or
//! `_ResourceKind::_Asynchronous` for `any_resource`.
//! @tparam _Properties A pack of property types that a memory resource must
//! provide in order to be storable in instances of this `basic_any_resource`
//! type.
//!
//! @sa any_synchronous_resource
//! @sa any_resource
//! @sa synchronous_resource_ref
//! @sa resource_ref
template <_ResourceKind _Kind, class... _Properties>
class basic_any_resource
{
public:
  //! @brief Constructs a \c basic_any_resource with no value
  //! @post `has_value()` is `false`
  basic_any_resource() noexcept;

  //! @brief Move constructs a \c basic_any_resource
  //! @post `has_value()` is `true` if `__other` had a value prior to the move,
  //! and `false` otherwise. `__other.has_value()` is `false`.
  basic_any_resource(basic_any_resource&& __other) noexcept;

  //! @brief Copy constructs a \c basic_any_resource
  //! @post `has_value()` is the same as `__other.has_value()`.
  basic_any_resource(const basic_any_resource& __other);

  //! @brief Constructs a \c basic_any_resource from a type that satisfies the
  //! \c resource concept.
  //! and that supports all of the specified properties.
  //! @param __res The resource to be wrapped by the \c basic_any_resource.
  //! @pre \c _Resource is not a specialization of \c basic_any_resource or
  //! \c basic_resource_ref, or a type derived from such.
  //! @pre `synchronous_resource_with<_Resource, _Properties...>` is `true`.
  //! @pre If \c _Kind is \c _ResourceKind::_Asynchronous,
  //! `resource_with<_Resource, _Properties...>` is `true`.
  //! @post `has_value()` is `true`
  template <class _Resource>
  basic_any_resource(_Resource __res);

  //! @brief Conversion from a type-erased resource with a superset of the
  //! required properties.
  //! @param __res The object to copy from.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_OtherProperties...` is a superset of `_Properties...`.
  //! @post `has_value()` is equal to `__res.has_value()`
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  basic_any_resource(basic_any_resource<_OtherKind, _OtherProperties...> __res);

  //! @brief Deep copy from a type-erased resource reference with a superset
  //! of the required properties.
  //!
  //! The object to which \c __res refers is copied into `*this`.
  //! @param __res The reference to copy from.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_OtherProperties...` is a superset of `_Properties...`.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  basic_any_resource(basic_resource_ref<_OtherKind, _OtherProperties...> __res);

  //! @brief Move assigns a \c basic_any_resource
  //! @post `has_value()` is `true` if `__other` had a value prior to the move,
  //! and `false` otherwise.
  //! @post `__other.has_value()` is `false`.
  basic_any_resource& operator=(basic_any_resource&& __other) noexcept;

  //! @brief Copy assigns a \c basic_any_resource
  //! @post `has_value()` is the same as `__other.has_value()`.
  basic_any_resource& operator=(const basic_any_resource& __other);

  //! @brief Assigns from a type that satisfies the \c resource concept and that
  //! supports all of the specified properties.
  //! @param __res The resource to be wrapped within the \c basic_any_resource
  //! @pre \c _Resource is not a specialization of \c basic_any_resource or
  //! \c basic_resource_ref, or a type derived from such.
  //! @pre `synchronous_resource_with<_Resource, _Properties...>` is `true`.
  //! @pre If \c _Kind is \c _ResourceKind::_Asynchronous,
  //! `resource_with<_Resource, _Properties...>` is `true`.
  //! @post `has_value()` is `true`
  template <class _Resource>
  basic_any_resource& operator=(_Resource __res);

  //! @brief Assignment from a type-erased resource with a superset of the
  //! required properties.
  //! @param __res The object to copy from.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_OtherProperties...` is a superset of `_Properties...`.
  //! @post `has_value()` is equal to `__res.has_value()`.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  basic_any_resource& operator=(basic_any_resource<_OtherKind, _OtherProperties...> __res);

  //! @brief Deep copy from a type-erased resource reference with a superset of
  //! the required properties.
  //! @param __res The type-erased resource reference to copy from.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_OtherProperties...` is a superset of `_Properties...`.
  //! @post `has_value()` is `true`.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  basic_any_resource& operator=(basic_resource_ref<_OtherKind, _OtherProperties...> __res);

  //! @brief Equality comparison between two type-erased memory resource
  //! @param __rhs The type-erased resource to compare with `*this`.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_Properties...` is equal to the set `_OtherProperties...`.
  //! @return `true` if both resources hold objects of the same type and those
  //! objects compare equal, and `false` otherwise.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  [[nodiscard]] bool operator==(const basic_any_resource<_OtherKind, _OtherProperties...>& __rhs) const;

  //! @brief Equality comparison between `*this` and a type-erased resource
  //! reference.
  //! @param __rhs The type-erased resource reference to compare with `*this`.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_Properties...` is equal to the set `_OtherProperties...`.
  //! @return `true` if \c __rhs refers to an object of the same type as that
  //! wrapped by `*this` and those objects compare equal; `false` otherwise.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  [[nodiscard]] bool operator==(const basic_resource_ref<_OtherKind, _OtherProperties...>& __rhs) const;

  //! @brief Calls `allocate_sync` on the wrapped object with the specified
  //! arguments.
  //! @pre `has_value()` is `true`.
  //! @return `obj.allocate_sync(__size, __align)`, where `obj` is the wrapped
  //! object.
  [[nodiscard]] void* allocate_sync(size_t __size, size_t __align = alignof(cuda::std::max_align_t));

  //! @brief Calls `deallocate_sync` on the wrapped object with the specified
  //! arguments.
  //! @pre `has_value()` is `true`.
  //! @pre `__pv` must be a pointer that was previously returned by a call to \c
  //! allocate on the object wrapped by `*this`.
  //! @return `obj.deallocate_sync(__pv, __size, __align)`, where `obj` is the
  //! wrapped object.
  void deallocate_sync(void* __pv, size_t __size, size_t __align = alignof(cuda::std::max_align_t));

  //! @brief Calls `allocate` on the wrapped object with the specified
  //! arguments.
  //! @pre `_Kind` is `_ResourceKind::_Asynchronous`.
  //! @pre `has_value()` is `true`.
  //! @return `obj.allocate(__stream, __size, __align)`, where `obj` is
  //! the wrapped object.
  //! @warning The returned pointer is not valid until `__stream` has been
  //! synchronized.
  [[nodiscard]] void* allocate(cuda::stream_ref __stream, size_t __size, size_t __align);

  //! @brief Equivalent to `allocate(__stream, __size,
  //! alignof(::cuda::std::max_align_t))`.
  [[nodiscard]] void* allocate(cuda::stream_ref __stream, size_t __size);

  //! @brief Calls `deallocate` on the wrapped object with the specified
  //! arguments.
  //! @pre `_Kind` is `_ResourceKind::_Asynchronous`.
  //! @pre `has_value()` is `true`.
  //! @pre `__pv` must be a pointer that was previously returned by a call to
  //! \c allocate on the object wrapped by `*this`.
  //! @return `obj.deallocate(__stream, __pv, __size, __align)`, where
  //! `obj` is the wrapped object.
  void deallocate(cuda::stream_ref __stream, void* __pv, size_t __size, size_t __align);

  //! @brief Equivalent to `deallocate(__stream, __pv, __size,
  //! alignof(::cuda::std::max_align_t), __stream)`.
  void deallocate(cuda::stream_ref __stream, void* __pv, size_t __size);

  //! @brief Checks if `*this` holds a value.
  //! @return `true` if `*this` holds a value; `false` otherwise.
  [[nodiscard]] bool has_value() const noexcept;

  //! @brief Resets `*this` to the empty state.
  //! @post `has_value() == false`
  void reset() noexcept;

  //! @return A reference to the \c type_info object for the wrapped
  //! resource, or `typeid(void)` if `has_value()` is `false`.
  [[nodiscard]] const cuda::std::type_info& type() const noexcept;

  //! @brief Forwards a property query to the type-erased object.
  //! @param __res The \c basic_any_resource object
  //! @param __prop The property to query
  //! @pre The type \c _Property is one of the types in the pack
  //! `_Properties...`.
  //! @return The result of calling `get_property(__obj, __prop)`, where `__obj`
  //! is the type-erased object stored in `__res`.
  template <class _Property>
  friend decltype(auto) get_property(const basic_any_resource& __res, _Property __prop) noexcept;

  //! @brief Attempts to forward a property query to the type-erased object and
  //! returns a _`boolean-testable`_ object that contains the result, if any.
  //!
  //! @tparam _Property
  //! @param __res The \c basic_any_resource object
  //! @param __prop The property to query
  //! @pre `has_value()` is `true`.
  //! @return
  //! Let:
  //!   - \c obj be the wrapped object.
  //!   - \c ValueType be the associated value type of \c __prop.
  //!   - \c ReturnType be \c bool if \c ValueType is \c void. Otherwise,
  //!     \c ReturnType is \c cuda::std::optional<ValueType>.
  //!   - \c _OtherProperties be the pack of type parameters of the
  //!     \c basic_any_resource object that first type-erased \c obj. [_Note:_
  //!     `_OtherProperties` is different than `_Properties` when \c *this is
  //!     the result of a conversion from a different \c __basic_any type. -- end
  //!     note]
  //!   .
  //! `try_get_property(__res, __prop)` has type \c ReturnType. If \c _Property
  //! is not in the pack \c _OtherProperties, returns `ReturnType()`.
  //! Otherwise:
  //!   - Returns \c true if \c ValueType is \c void.
  //!   - Returns `ReturnType(get_property(obj, __prop))` otherwise.
  template <class _Property>
  friend auto try_get_property(const basic_any_resource& __res, _Property __prop) noexcept;
};

//! @brief Type erased wrapper around a reference to an object that satisfies
//! the \c resource concept and that provides the requested \c _Properties.
//! @tparam _Properties The properties that any resource wrapped within the
//! `basic_resource_ref` needs to provide.
//!
//! ``basic_resource_ref`` models the ``cuda::std::copyable`` and
//! ``cuda::std::equality_comparable`` concepts.
template <_ResourceKind _Kind, class... _Properties>
class basic_resource_ref
{
public:
  //! @brief Copy constructs a \c basic_resource_ref
  //! @post `*this` and `__other` both refer to the same resource object.
  basic_resource_ref(const basic_resource_ref& __other);

  //! @brief Constructs a \c basic_resource_ref from a reference to a type that
  //! satisfies the \c resource concept and that supports all of the specified
  //! properties.
  //! @param __res The resource reference to be wrapped.
  //! @pre `synchronous_resource_with<_Resource, _Properties...>` is `true`.
  //! @pre If \c _Kind is \c _ResourceKind::_Asynchronous,
  //! `resource_with<_Resource, _Properties...>` is `true`.
  //! @pre If \c __res refers to a specialization of \c basic_any_resource or
  //! a type derived from such, `__res.has_value()` is `true`.
  template <class _Resource>
  basic_resource_ref(_Resource& __res);

  //! @brief Conversion from type-erased resource reference with a superset
  //! of the required properties.
  //! @param __res The other type-erased resource reference to copy from.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_OtherProperties...` is a superset of `_Properties...`.
  //! @post `*this` and `__res` both refer to the same resource object.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  basic_resource_ref(basic_resource_ref<_OtherKind, _OtherProperties...> __res);

  //! @brief Rebinds `*this` to refer to the object to which `__other` refers.
  //! @post `*this` and `__other` both refer to the same resource object.
  basic_resource_ref& operator=(const basic_resource_ref& __other);

  //! @brief Rebinds the wrapped reference to an object whose type satisfies the
  //! \c resource concept and that supports all of the specified properties.
  //! @param __res The reference to the resource to be wrapped by the \c
  //! basic_resource_ref.
  //! @pre `synchronous_resource_with<_Resource, _Properties...>` is `true`.
  //! @pre If \c _Kind is \c _ResourceKind::_Asynchronous,
  //! `synchronous_resource_with<_Resource, _Properties...>` is `true`.
  //! @pre If \c __res refers to a specialization of \c basic_any_resource or a
  //! type derived from such, `__res.has_value()` is `true`.
  template <class _Resource>
  basic_resource_ref& operator=(_Resource& __res);

  //! @brief Rebinds `*this` to refer to the object to which `__other` refers.
  //! @param __res The other type-erased resource reference to copy from.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_OtherProperties...` is a superset of `_Properties...`.
  //! @post `*this` and `__res` both refer to the same resource object.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  basic_resource_ref& operator=(basic_resource_ref<_OtherKind, _OtherProperties...> __res);

  //! @brief Equality comparison between two type-erased resource references.
  //! @param __rhs The other type-erased resource reference.
  //! @pre \c _OtherKind is equal to either \c _Kind or
  //! \c _ResourceKind::_Asynchronous.
  //! @pre The set `_Properties...` is equal to the set `_OtherProperties...`.
  //! @return `true` if both resources refer to objects of the same type and
  //! those objects compare equal. Otherwise, returns `false`.
  template <_ResourceKind _OtherKind, class... _OtherProperties>
  [[nodiscard]] bool operator==(const basic_resource_ref<_OtherKind, _OtherProperties...>& __rhs) const;

  //! @brief Calls `allocate_sync` on the wrapped reference with the specified
  //! arguments.
  //! @return `obj.allocate_sync(__size, __align)`, where `obj` is the wrapped
  //! reference.
  [[nodiscard]] void* allocate_sync(size_t __size, size_t __align = alignof(cuda::std::max_align_t));

  //! @brief Calls `deallocate_sync` on the wrapped reference with the specified
  //! arguments.
  //! @pre `__pv` must be a pointer that was previously returned by a call to
  //! \c allocate on the object referenced by `*this`.
  //! @return `obj.deallocate_sync(__pv, __size, __align)`, where `obj` is the
  //! wrapped reference.
  void deallocate_sync(void* __pv, size_t __size, size_t __align = alignof(cuda::std::max_align_t));

  //! @brief Calls `allocate` on the wrapped reference with the specified
  //! arguments.
  //! @pre `_Kind` is `_ResourceKind::_Asynchronous`.
  //! @return `obj.allocate(__stream, __size, __align)`, where `obj` is
  //! the wrapped reference.
  //! @warning The returned pointer is not valid until `__stream` has been
  //! synchronized.
  [[nodiscard]] void* allocate(cuda::stream_ref __stream, size_t __size, size_t __align);

  //! @brief Equivalent to `allocate(__stream, __size,
  //! alignof(::cuda::std::max_align_t))`.
  [[nodiscard]] void* allocate(cuda::stream_ref __stream, size_t __size);

  //! @brief Calls `deallocate` on the wrapped reference with the specified
  //! arguments.
  //! @pre `_Kind` is `_ResourceKind::_Asynchronous`.
  //! @pre `__pv` must be a pointer that was previously returned by a call to \c
  //! allocate on the object referenced by `*this`.
  //! @return `obj.deallocate(__stream, __pv, __size, __align)`, where
  //! `obj` is the wrapped reference.
  void deallocate(cuda::stream_ref __stream, void* __pv, size_t __size, size_t __align);

  //! @brief Equivalent to `deallocate(__stream, __pv, __size,
  //! alignof(::cuda::std::max_align_t), __stream)`.
  void deallocate(cuda::stream_ref __stream, void* __pv, size_t __size);

  //! @return A reference to the \c type_info object for the type of the object
  //! to which `*this` refers.
  [[nodiscard]] const cuda::std::type_info& type() const noexcept;

  //! @brief Forwards a property query to the type-erased reference.
  //! @tparam _Property
  //! @param __res The \c basic_resource_ref object
  //! @param __prop The property to query
  //! @pre \c _Property is a type in `_Properties...`.
  //! @return The result of calling `get_property(__obj, __prop)`, where `__obj`
  //! is the type-erased reference stored in `__res`.
  template <class _Property>
  friend decltype(auto) get_property(const basic_resource_ref& __res, _Property __prop) noexcept;

  //! @brief Attempts to forward a property query to the type-erased object and
  //! returns a _`boolean-testable`_ object that contains the result, if any.
  //!
  //! @tparam _Property
  //! @param __res The \c any_synchronous_resource object
  //! @param __prop The property to query
  //! @pre `has_value()` is `true`.
  //! @return
  //! Let:
  //!   - \c obj be the wrapped reference.
  //!   - \c ValueType be the associated value type of \c __prop.
  //!   - \c ReturnType be \c bool if \c ValueType is \c void. Otherwise,
  //!     \c ReturnType is \c cuda::std::optional<ValueType>.
  //!   - \c _OtherProperties be the pack of type parameters of the wrapper type
  //!     that first type-erased \c obj. [_Note:_ `_OtherProperties` is
  //!     different than `_Properties` when \c *this is the result of an
  //!     interface-narrowing conversion. -- end note]
  //!   .
  //! `try_get_property(__res, __prop)` has type \c ReturnType. If \c _Property
  //! is not in the pack \c _OtherProperties, returns `ReturnType()`.
  //! Otherwise:
  //!   - Returns \c true if \c ValueType is \c void.
  //!   - Returns `ReturnType(get_property(obj, __prop))` otherwise.
  template <class _Property>
  friend auto try_get_property(const basic_resource_ref& __res, _Property __prop) noexcept;
};

//! @rst
//! .. _cudax-memory-resource-any-resource:
//!
//! Type erased wrapper around a `synchronous_resource`
//! ----------------------------------------------------
//!
//! ``any_synchronous_resource`` wraps any given :ref:`synchronous_resource
//! <libcudacxx-extended-api-memory-resources-resource>` that satisfies the
//! required properties. It owns the contained resource, taking care of
//! construction / destruction. This makes it especially suited for use in e.g.
//! container types that need to ensure that the lifetime of the container
//! exceeds the lifetime of the memory resource used to allocate the storage
//!
//! ``any_synchronous_resource`` models the ``cuda::std::regular`` concept.
//!
//! @endrst
template <class... _Properties>
using any_synchronous_resource = basic_any_resource<_ResourceKind::_Synchronous, _Properties...>;

//! @rst
//! .. _cudax-memory-resource-any-async-resource:
//!
//! Type erased wrapper around an `resource`
//! ----------------------------------------------
//!
//! ``any_resource`` wraps any given :ref:`resource
//! <libcudacxx-extended-api-memory-resources-resource>` that satisfies the
//! required properties. It owns the contained resource, taking care of
//! construction / destruction. This makes it especially suited for use in e.g.
//! container types that need to ensure that the lifetime of the container
//! exceeds the lifetime of the memory resource used to allocate the storage
//!
//! ``any_resource`` models the ``cuda::std::regular`` concept.
//!
//! @endrst
template <class... _Properties>
using any_resource = basic_any_resource<_ResourceKind::_Asynchronous, _Properties...>;

//! @brief Type erased wrapper around a `synchronous_resource` that satisfies \c
//! _Properties.
//! @tparam _Properties The properties that any resource wrapped within the
//! `synchronous_resource_ref` needs to satisfy
template <class... _Properties>
using synchronous_resource_ref = basic_resource_ref<_ResourceKind::_Synchronous, _Properties...>;

//! @brief Type erased wrapper around a `resource` that satisfies \c
//! _Properties
//! @tparam _Properties The properties that any resource wrapped within
//! the `resource_ref` needs to satisfy
template <class... _Properties>
using resource_ref = basic_resource_ref<_ResourceKind::_Asynchronous, _Properties...>;

#endif // _CCCL_DOXYGEN_INVOKED

//! @rst
//! .. _cudax-memory-resource-make-any-resource:
//!
//! Factory function for `any_synchronous_resource` objects
//! -------------------------------------------------------
//!
//! ``make_any_synchronous_resource`` constructs an :ref:`any_synchronous_resource
//! <cudax-memory-resource-any-resource>` object that wraps a newly constructed
//! instance of the given resource type. The resource type must satisfy the
//! ``cuda::mr::synchronous_resource`` concept and provide all of the properties specified
//! in the template parameter pack.
//!
//! @param __args The arguments used to construct the instance of the resource
//! type.
//!
//! @endrst
template <class _Resource, class... _Properties, class... _Args>
auto make_any_synchronous_resource(_Args&&... __args) -> any_synchronous_resource<_Properties...>
{
  static_assert(::cuda::mr::synchronous_resource<_Resource>,
                "_Resource does not satisfy the cuda::mr::synchronous_resource concept");
  static_assert(::cuda::mr::synchronous_resource_with<_Resource, _Properties...>,
                "The provided _Resource type does not support the requested properties");
  return any_synchronous_resource<_Properties...>{
    ::cuda::std::in_place_type<_Resource>, ::cuda::std::forward<_Args>(__args)...};
}

//! @rst
//! .. _cudax-memory-resource-make-any-async-resource:
//!
//! Factory function for `any_resource` objects
//! -------------------------------------------------
//!
//! ``make_any_resource`` constructs an :ref:`any_resource
//! <cudax-memory-resource-any-async-resource>` object that wraps a newly
//! constructed instance of the given resource type. The resource type must
//! satisfy the ``cuda::mr::resource`` concept and provide all of the
//! properties specified in the template parameter pack.
//!
//! @param __args The arguments used to construct the instance of the resource
//! type.
//!
//! @endrst
template <class _Resource, class... _Properties, class... _Args>
auto make_any_resource(_Args&&... __args) -> any_resource<_Properties...>
{
  static_assert(::cuda::mr::resource<_Resource>, "_Resource does not satisfy the cuda::mr::resource concept");
  static_assert(::cuda::mr::resource_with<_Resource, _Properties...>,
                "The provided _Resource type does not support the requested properties");
  return any_resource<_Properties...>{::cuda::std::in_place_type<_Resource>, ::cuda::std::forward<_Args>(__args)...};
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
