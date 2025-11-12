/*
 *  Copyright 2008-2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file
 *  \brief A pointer to a variable which resides in memory associated with a
 *  system.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/reference_forward_declaration.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traversal_tags.h>

#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstddef>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif // !_CCCL_COMPILER(NVRTC)

THRUST_NAMESPACE_BEGIN
template <typename Element, typename Tag, typename Reference = use_default, typename Derived = use_default>
class pointer;
THRUST_NAMESPACE_END

// Specialize `std::iterator_traits` (picked up by cuda::std::iterator_traits) to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type. We do this before pointer is defined so the specialization
// is correctly used inside the definition.
template <typename Element, typename Tag, typename Reference, typename Derived>
struct std::iterator_traits<THRUST_NS_QUALIFIER::pointer<Element, Tag, Reference, Derived>>
{
  using pointer           = THRUST_NS_QUALIFIER::pointer<Element, Tag, Reference, Derived>;
  using iterator_category = typename pointer::iterator_category;
  using value_type        = typename pointer::value_type;
  using difference_type   = typename pointer::difference_type;
  using reference         = typename pointer::reference;
};

THRUST_NAMESPACE_BEGIN
namespace detail
{
// this metafunction computes the type of iterator_adaptor thrust::pointer should inherit from
template <typename Element, typename Tag, typename Reference, typename Derived>
struct pointer_base
{
  // void pointers should have no element type
  // note that we remove_cv from the Element type to get the value_type
  using value_type = typename eval_if<::cuda::std::is_void_v<::cuda::std::remove_cvref_t<Element>>,
                                      ::cuda::std::type_identity<void>,
                                      ::cuda::std::remove_cv<Element>>::type;

  // if no Derived type is given, just use pointer
  using derived_type =
    typename eval_if<::cuda::std::is_same_v<Derived, use_default>,
                     ::cuda::std::type_identity<pointer<Element, Tag, Reference, Derived>>,
                     ::cuda::std::type_identity<Derived>>::type;

  // void pointers should have no reference type
  // if no Reference type is given, just use reference
  using reference_type =
    typename eval_if<::cuda::std::is_void_v<::cuda::std::remove_cvref_t<Element>>,
                     ::cuda::std::type_identity<void>,
                     eval_if<::cuda::std::is_same_v<Reference, use_default>,
                             ::cuda::std::type_identity<reference<Element, derived_type>>,
                             ::cuda::std::type_identity<Reference>>>::type;

  using type =
    iterator_adaptor<derived_type, Element*, value_type, Tag, random_access_traversal_tag, reference_type, std::ptrdiff_t>;
};
} // namespace detail

/*! \addtogroup memory_management Memory Management
 *  \{
 */

// the base type for all of thrust's tagged pointers.
// for reasonable pointer-like semantics, derived types should reimplement the following:
// 1. no-argument constructor
// 2. constructor from OtherElement *
// 3. constructor from OtherPointer related by convertibility
// 4. constructor from OtherPointer to void
// 5. assignment from OtherPointer related by convertibility
// These should just call the corresponding members of pointer.

/*! \p pointer stores a pointer to an object allocated in memory. Like \p device_ptr, this
 *  type ensures type safety when dispatching standard algorithms on ranges resident in memory.
 *
 *  \p pointer generalizes \p device_ptr by relaxing the backend system associated with the \p pointer.
 *  Instead of the backend system specified by \p THRUST_DEVICE_SYSTEM, \p pointer's
 *  system is given by its second template parameter, \p Tag. For the purpose of Thrust dispatch,
 *  <tt>device_ptr<Element></tt> and <tt>pointer<Element,device_system_tag></tt> are considered equivalent.
 *
 *  The raw pointer encapsulated by a \p pointer may be obtained through its <tt>get</tt> member function
 *  or the \p raw_pointer_cast free function.
 *
 *  \tparam Element specifies the type of the pointed-to object.
 *
 *  \tparam Tag specifies the system with which this \p pointer is associated. This may be any Thrust
 *          backend system, or a user-defined tag.
 *
 *  \tparam Reference allows the client to specify the reference type returned upon derereference.
 *          By default, this type is <tt>reference<Element,pointer></tt>.
 *
 *  \tparam Derived allows the client to specify the name of the derived type when \p pointer is used as
 *          a base class. This is useful to ensure that arithmetic on values of the derived type return
 *          values of the derived type as a result. By default, this type is <tt>pointer<Element,Tag,Reference></tt>.
 *
 *  \note \p pointer is not a smart pointer; it is the client's responsibility to deallocate memory
 *        pointer to by \p pointer.
 *
 *  \see device_ptr
 *  \see reference
 *  \see raw_pointer_cast
 */
template <typename Element, typename Tag, typename Reference, typename Derived>
class pointer : public detail::pointer_base<Element, Tag, Reference, Derived>::type
{
private:
  using super_t      = typename detail::pointer_base<Element, Tag, Reference, Derived>::type;
  using derived_type = typename detail::pointer_base<Element, Tag, Reference, Derived>::derived_type;

  // friend iterator_core_access to give it access to dereference
  friend class iterator_core_access;

  template <typename SuperRef = typename super_t::reference>
  _CCCL_HOST_DEVICE SuperRef dereference() const
  {
    const auto& derived_ptr = static_cast<const derived_type&>(*this);
    if constexpr (::cuda::std::is_reference_v<SuperRef>)
    {
      // Implementation for dereference() when Reference is Element&, e.g. cuda's managed_memory_pointer
      return *derived_ptr.get();
    }
    else
    {
      // Implementation for pointers with proxy references:
      return SuperRef(derived_ptr);
    }
  }

  // don't provide access to this part of super_t's interface
  using super_t::base;
  using typename super_t::base_type;

public:
  /*! The type of the raw pointer
   */
  using raw_pointer = typename super_t::base_type;

  /*! \p pointer's default constructor initializes its encapsulated pointer to \c 0
   */
  _CCCL_HOST_DEVICE pointer()
      : super_t(static_cast<Element*>(nullptr))
  {}

  // NOTE: This is needed so that Thrust smart pointers can be used in `std::unique_ptr`.
  _CCCL_HOST_DEVICE pointer(::cuda::std::nullptr_t)
      : super_t(static_cast<Element*>(nullptr))
  {}

  /*! This constructor allows construction of a <tt>pointer<const T, ...></tt> from a <tt>T*</tt>.
   *
   *  \param ptr A raw pointer to copy from, presumed to point to a location in \p Tag's memory.
   *  \tparam OtherElement \p OtherElement shall be convertible to \p Element.
   */
  // XXX consider making the pointer implementation a template parameter which defaults to Element *
  template <typename OtherElement>
  _CCCL_HOST_DEVICE explicit pointer(OtherElement* ptr)
      : super_t(ptr)
  {}

  /*! This constructor allows initialization from another pointer-like object.
   *
   *  \param other The \p OtherPointer to copy.
   *
   *  \tparam OtherPointer The tag associated with \p OtherPointer shall be convertible to \p Tag,
   *                       and its element type shall be convertible to \p Element.
   */
  template <typename OtherPointer,
            typename detail::enable_if_pointer_is_convertible<OtherPointer, pointer>::type* = nullptr>
  _CCCL_HOST_DEVICE pointer(const OtherPointer& other)
      : super_t(detail::pointer_traits<OtherPointer>::get(other))
  {}

#ifndef _CCCL_DOXYGEN_INVOKED // Doxygen cannot handle this constructor and creates a duplicate ID with the ctor above
  // OtherPointer's element_type shall be void
  // OtherPointer's system shall be convertible to Tag
  template <typename OtherPointer,
            typename detail::enable_if_void_pointer_is_system_convertible<OtherPointer, pointer>::type* = nullptr>
  _CCCL_HOST_DEVICE explicit pointer(const OtherPointer& other)
      : super_t(static_cast<Element*>(detail::pointer_traits<OtherPointer>::get(other)))
  {}
#endif

  // assignment

  // NOTE: This is needed so that Thrust smart pointers can be used in `std::unique_ptr`.
  _CCCL_HOST_DEVICE derived_type& operator=(::cuda::std::nullptr_t)
  {
    super_t::base_reference() = nullptr;
    return static_cast<derived_type&>(*this);
  }

  /*! Assignment operator allows assigning from another pointer-like object whose element type
   *  is convertible to \c Element.
   *
   *  \param other The other pointer-like object to assign from.
   *  \return <tt>*this</tt>
   *
   *  \tparam OtherPointer The tag associated with \p OtherPointer shall be convertible to \p Tag,
   *                       and its element type shall be convertible to \p Element.
   */
  template <typename OtherPointer>
  _CCCL_HOST_DEVICE typename detail::enable_if_pointer_is_convertible<OtherPointer, pointer, derived_type&>::type
  operator=(const OtherPointer& other)
  {
    super_t::base_reference() = detail::pointer_traits<OtherPointer>::get(other);
    return static_cast<derived_type&>(*this);
  }

  // observers

  /*! \p get returns this \p pointer's encapsulated raw pointer.
   *  \return This \p pointer's raw pointer.
   */
  _CCCL_HOST_DEVICE Element* get() const
  {
    return super_t::base();
  }

  _CCCL_HOST_DEVICE Element* operator->() const
  {
    return super_t::base();
  }

  // NOTE: This is needed so that Thrust smart pointers can be used in `std::unique_ptr`.
  _CCCL_HOST_DEVICE explicit operator bool() const
  {
    return bool(get());
  }

  _CCCL_HOST_DEVICE static derived_type
  pointer_to(typename detail::pointer_traits_detail::pointer_to_param<Element>::type r)
  {
    return detail::pointer_traits<derived_type>::pointer_to(r);
  }

#if !_CCCL_COMPILER(NVRTC)
  template <typename charT, typename traits>
  _CCCL_HOST friend std::basic_ostream<charT, traits>&
  operator<<(std::basic_ostream<charT, traits>& os, const pointer& p)
  {
    return os << p.get();
  }
#endif // !_CCCL_COMPILER(NVRTC)

  // NOTE: This is needed so that Thrust smart pointers can be used in `std::unique_ptr`.
  _CCCL_HOST_DEVICE friend bool operator==(::cuda::std::nullptr_t, pointer p)
  {
    return nullptr == p.get();
  }

  _CCCL_HOST_DEVICE friend bool operator==(pointer p, ::cuda::std::nullptr_t)
  {
    return nullptr == p.get();
  }

  _CCCL_HOST_DEVICE friend bool operator!=(::cuda::std::nullptr_t, pointer p)
  {
    return !(nullptr == p);
  }

  _CCCL_HOST_DEVICE friend bool operator!=(pointer p, ::cuda::std::nullptr_t)
  {
    return !(nullptr == p);
  }
};

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
