/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

// allocator_traits::rebind_alloc and allocator::rebind_traits are from libc++,
// dual licensed under the MIT and the University of Illinois Open Source
// Licenses.

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_deduction.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/has_member_function.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits/is_call_possible.h>
#include <thrust/detail/type_traits/pointer_traits.h>

#include <cuda/std/__cccl/memory_wrapper.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/limits>

#include <new>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// forward declaration for has_member_system
template <typename Alloc>
struct allocator_system;

template <typename Alloc>
struct allocator_traits;

namespace allocator_traits_detail
{
__THRUST_DEFINE_HAS_NESTED_TYPE(has_value_type, value_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_pointer, pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_pointer, const_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_reference, reference)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_reference, const_reference)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_void_pointer, void_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_void_pointer, const_void_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_difference_type, difference_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_size_type, size_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_copy_assignment, propagate_on_container_copy_assignment)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_move_assignment, propagate_on_container_move_assignment)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_swap, propagate_on_container_swap)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_system_type, system_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_is_always_equal, is_always_equal)
__THRUST_DEFINE_HAS_MEMBER_FUNCTION(has_member_system_impl, system)

template <typename Alloc, typename U, typename = void>
inline constexpr bool has_rebind = false;

template <typename Alloc, typename U>
inline constexpr bool has_rebind<Alloc, U, ::cuda::std::void_t<decltype(U::template rebind<U>::other)>> = true;

_CCCL_SUPPRESS_DEPRECATED_PUSH

// The following fields of std::allocator have been deprecated (since C++17).
// There's no way to detect it other than explicit specialization.
#define THRUST_SPECIALIZE_DEPRECATED(trait_name)    \
  template <typename T>                             \
  struct trait_name<std::allocator<T>> : false_type \
  {};

THRUST_SPECIALIZE_DEPRECATED(has_is_always_equal)
THRUST_SPECIALIZE_DEPRECATED(has_pointer)
THRUST_SPECIALIZE_DEPRECATED(has_const_pointer)
THRUST_SPECIALIZE_DEPRECATED(has_reference)
THRUST_SPECIALIZE_DEPRECATED(has_const_reference)

#undef THRUST_SPECIALIZE_DEPRECATED

template <typename T, typename U>
inline constexpr bool has_rebind<std::allocator<T>, U, void> = false;

template <typename T>
struct nested_pointer
{
  using type = typename T::pointer;
};

template <typename T>
struct nested_const_pointer
{
  using type = typename T::const_pointer;
};

template <typename T>
struct nested_reference
{
  using type = typename T::reference;
};

template <typename T>
struct nested_const_reference
{
  using type = typename T::const_reference;
};

template <typename T>
struct nested_void_pointer
{
  using type = typename T::void_pointer;
};

template <typename T>
struct nested_const_void_pointer
{
  using type = typename T::const_void_pointer;
};

template <typename T>
struct nested_difference_type
{
  using type = typename T::difference_type;
};

template <typename T>
struct nested_size_type
{
  using type = typename T::size_type;
};

template <typename T>
struct nested_propagate_on_container_copy_assignment
{
  using type = typename T::propagate_on_container_copy_assignment;
};

template <typename T>
struct nested_propagate_on_container_move_assignment
{
  using type = typename T::propagate_on_container_move_assignment;
};

template <typename T>
struct nested_propagate_on_container_swap
{
  using type = typename T::propagate_on_container_swap;
};

template <typename T>
struct nested_is_always_equal
{
  using type = typename T::is_always_equal;
};

template <typename T>
struct nested_system_type
{
  using type = typename T::system_type;
};

template <typename Alloc>
struct has_member_system
{
  using system_type = typename allocator_system<Alloc>::type;

  using type              = typename has_member_system_impl<Alloc, system_type&()>::type;
  static const bool value = type::value;
};

_CCCL_SUPPRESS_DEPRECATED_POP

template <class Alloc, class U, bool = has_rebind<Alloc, U>>
struct rebind_alloc
{
  using type = typename Alloc::template rebind<U>::other;
};

template <template <typename, typename...> class Alloc, typename T, typename... Args, typename U>
struct rebind_alloc<Alloc<T, Args...>, U, true>
{
  using type = typename Alloc<T, Args...>::template rebind<U>::other;
};

template <template <typename, typename...> class Alloc, typename T, typename... Args, typename U>
struct rebind_alloc<Alloc<T, Args...>, U, false>
{
  using type = Alloc<U, Args...>;
};

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_allocate_with_hint_impl, allocate)

template <typename Alloc>
class has_member_allocate_with_hint
{
  using pointer            = typename allocator_traits<Alloc>::pointer;
  using size_type          = typename allocator_traits<Alloc>::size_type;
  using const_void_pointer = typename allocator_traits<Alloc>::const_void_pointer;

public:
  using type = typename has_member_allocate_with_hint_impl<Alloc, pointer(size_type, const_void_pointer)>::type;
  static const bool value = type::value;
};

template <typename Alloc>
_CCCL_HOST_DEVICE typename allocator_traits<Alloc>::pointer
allocate(Alloc& a,
         typename allocator_traits<Alloc>::size_type n,
         [[maybe_unused]] typename allocator_traits<Alloc>::const_void_pointer hint)
{
  if constexpr (has_member_allocate_with_hint<Alloc>::value)
  {
    return a.allocate(n, hint);
  }
  else
  {
    return a.allocate(n);
  }
}

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_construct1_impl, construct)

template <typename Alloc, typename T>
struct has_member_construct1 : has_member_construct1_impl<Alloc, void(T*)>
{};

_CCCL_EXEC_CHECK_DISABLE
template <typename Alloc, typename T>
_CCCL_HOST_DEVICE void construct(Alloc& a, T* p)
{
  if constexpr (has_member_construct1<Alloc, T>::value)
  {
    a.construct(p);
  }
  else
  {
    ::new (static_cast<void*>(p)) T();
  }
}

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_construct2_impl, construct)

template <typename Alloc, typename T, typename Arg1>
struct has_member_construct2 : has_member_construct2_impl<Alloc, void(T*, const Arg1&)>
{};

_CCCL_EXEC_CHECK_DISABLE
template <typename Alloc, typename T, typename Arg1>
_CCCL_HOST_DEVICE void construct(Alloc& a, T* p, const Arg1& arg1)
{
  if constexpr (has_member_construct2<Alloc, T, Arg1>::value)
  {
    a.construct(p, arg1);
  }
  else
  {
    ::new (static_cast<void*>(p)) T(arg1);
  }
}

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_constructN_impl, construct)

template <typename Alloc, typename T, typename... Args>
struct has_member_constructN : has_member_constructN_impl<Alloc, void(T*, Args...)>
{};

_CCCL_EXEC_CHECK_DISABLE
template <typename Alloc, typename T, typename... Args>
inline _CCCL_HOST_DEVICE void construct([[maybe_unused]] Alloc& a, T* p, Args&&... args)
{
  if constexpr (has_member_constructN<Alloc, T, Args...>::value)
  {
    a.construct(p, THRUST_FWD(args)...);
  }
  else
  {
    ::new (static_cast<void*>(p)) T(THRUST_FWD(args)...);
  }
}

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_destroy_impl, destroy)

template <typename Alloc, typename T>
struct has_member_destroy : has_member_destroy_impl<Alloc, void(T*)>
{};

_CCCL_EXEC_CHECK_DISABLE
template <typename Alloc, typename T>
_CCCL_HOST_DEVICE void destroy([[maybe_unused]] Alloc& a, T* p)
{
  if constexpr (has_member_destroy<Alloc, T>::value)
  {
    a.destroy(p);
  }
  else
  {
    p->~T();
  }
}

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_max_size_impl, max_size)

template <typename Alloc>
class has_member_max_size
{
  using size_type = typename allocator_traits<Alloc>::size_type;

public:
  using type              = typename has_member_max_size_impl<Alloc, size_type()>::type;
  static const bool value = type::value;
};

template <typename Alloc>
_CCCL_HOST_DEVICE typename allocator_traits<Alloc>::size_type max_size([[maybe_unused]] const Alloc& a)
{
  if constexpr (has_member_max_size<Alloc>::value)
  {
    return a.max_size();
  }
  else
  {
    using size_type = typename allocator_traits<Alloc>::size_type;
    return ::cuda::std::numeric_limits<size_type>::max();
  }
}

// TODO(bgruber): can be return decltype(auto) here?
template <typename Alloc>
_CCCL_HOST_DEVICE ::cuda::std::
  _If<has_member_system<Alloc>::value, typename allocator_system<Alloc>::type&, typename allocator_system<Alloc>::type>
  system(Alloc& a)
{
  if constexpr (has_member_system<Alloc>::value)
  {
    // return the allocator's system
    return a.system();
  }
  else
  {
    // return a copy of a value-initialized system
    return typename allocator_system<Alloc>::type{};
  }
}
} // namespace allocator_traits_detail

template <typename Alloc>
struct allocator_traits
{
  using allocator_type = Alloc;

  using value_type = typename allocator_type::value_type;

  using pointer = typename eval_if<allocator_traits_detail::has_pointer<allocator_type>::value,
                                   allocator_traits_detail::nested_pointer<allocator_type>,
                                   ::cuda::std::type_identity<value_type*>>::type;

private:
  template <typename T>
  struct rebind_pointer
  {
    using type = typename pointer_traits<pointer>::template rebind<T>::other;
  };

public:
  using const_pointer =
    typename eval_if<allocator_traits_detail::has_const_pointer<allocator_type>::value,
                     allocator_traits_detail::nested_const_pointer<allocator_type>,
                     rebind_pointer<const value_type>>::type;

  using void_pointer =
    typename eval_if<allocator_traits_detail::has_void_pointer<allocator_type>::value,
                     allocator_traits_detail::nested_void_pointer<allocator_type>,
                     rebind_pointer<void>>::type;

  using const_void_pointer =
    typename eval_if<allocator_traits_detail::has_const_void_pointer<allocator_type>::value,
                     allocator_traits_detail::nested_const_void_pointer<allocator_type>,
                     rebind_pointer<const void>>::type;

  using difference_type =
    typename eval_if<allocator_traits_detail::has_difference_type<allocator_type>::value,
                     allocator_traits_detail::nested_difference_type<allocator_type>,
                     pointer_difference<pointer>>::type;

  using size_type = typename eval_if<allocator_traits_detail::has_size_type<allocator_type>::value,
                                     allocator_traits_detail::nested_size_type<allocator_type>,
                                     ::cuda::std::make_unsigned<difference_type>>::type;

  using propagate_on_container_copy_assignment =
    typename eval_if<allocator_traits_detail::has_propagate_on_container_copy_assignment<allocator_type>::value,
                     allocator_traits_detail::nested_propagate_on_container_copy_assignment<allocator_type>,
                     ::cuda::std::type_identity<false_type>>::type;

  using propagate_on_container_move_assignment =
    typename eval_if<allocator_traits_detail::has_propagate_on_container_move_assignment<allocator_type>::value,
                     allocator_traits_detail::nested_propagate_on_container_move_assignment<allocator_type>,
                     ::cuda::std::type_identity<false_type>>::type;

  using propagate_on_container_swap =
    typename eval_if<allocator_traits_detail::has_propagate_on_container_swap<allocator_type>::value,
                     allocator_traits_detail::nested_propagate_on_container_swap<allocator_type>,
                     ::cuda::std::type_identity<false_type>>::type;

  using is_always_equal =
    typename eval_if<allocator_traits_detail::has_is_always_equal<allocator_type>::value,
                     allocator_traits_detail::nested_is_always_equal<allocator_type>,
                     ::cuda::std::is_empty<allocator_type>>::type;

  using system_type =
    typename eval_if<allocator_traits_detail::has_system_type<allocator_type>::value,
                     allocator_traits_detail::nested_system_type<allocator_type>,
                     thrust::iterator_system<pointer>>::type;

  // XXX rebind and rebind_traits are alias templates
  //     and so are omitted while c++11 is unavailable

  template <typename U>
  using rebind_alloc = typename allocator_traits_detail::rebind_alloc<allocator_type, U>::type;

  template <typename U>
  using rebind_traits = allocator_traits<rebind_alloc<U>>;

  // We define this nested type alias for compatibility with the C++03-style
  // rebind_* mechanisms.
  using other = allocator_traits;

  // Deprecated std::allocator aliases that we need:
  using reference       = typename pointer_traits<pointer>::reference;
  using const_reference = typename pointer_traits<const_pointer>::reference;

  inline _CCCL_HOST_DEVICE static pointer allocate(allocator_type& a, size_type n)
  {
    struct workaround_warnings
    {
      _CCCL_EXEC_CHECK_DISABLE
      static _CCCL_HOST_DEVICE pointer allocate(Alloc& a, size_type n)
      {
        return a.allocate(n);
      }
    };

    return workaround_warnings::allocate(a, n);
  }

  inline _CCCL_HOST_DEVICE static pointer allocate(allocator_type& a, size_type n, const_void_pointer hint)
  {
    return allocator_traits_detail::allocate(a, n, hint);
  }

  inline _CCCL_HOST_DEVICE static void deallocate(allocator_type& a, pointer p, size_type n) noexcept
  {
    struct workaround_warnings
    {
      _CCCL_EXEC_CHECK_DISABLE
      static _CCCL_HOST_DEVICE void deallocate(Alloc& a, pointer p, size_type n) noexcept
      {
        return a.deallocate(p, n);
      }
    };

    return workaround_warnings::deallocate(a, p, n);
  }

  // XXX should probably change T* to pointer below and then relax later

  template <typename T>
  _CCCL_HOST_DEVICE static void construct(allocator_type& a, T* p)
  {
    return allocator_traits_detail::construct(a, p);
  }

  template <typename T, typename Arg1>
  _CCCL_HOST_DEVICE static void construct(allocator_type& a, T* p, const Arg1& arg1)
  {
    return allocator_traits_detail::construct(a, p, arg1);
  }

  template <typename T, typename... Args>
  _CCCL_HOST_DEVICE static void construct(allocator_type& a, T* p, Args&&... args)
  {
    return allocator_traits_detail::construct(a, p, THRUST_FWD(args)...);
  }

  template <typename T>
  _CCCL_HOST_DEVICE static void destroy(allocator_type& a, T* p) noexcept
  {
    return allocator_traits_detail::destroy(a, p);
  }

  _CCCL_HOST_DEVICE static size_type max_size(const allocator_type& a)
  {
    return allocator_traits_detail::max_size(a);
  }
}; // end allocator_traits

// std::allocator's member functions are deprecated in C++17 and removed in
// C++20, so we can't just use the generic implementation for allocator_traits
// that calls the allocator's member functions.
// Instead, specialize allocator_traits for std::allocator and defer to
// std::allocator_traits<std::allocator> and let the STL do whatever it needs
// to for the current c++ version. Manually forward the calls to suppress
// host/device warnings.
template <typename T>
struct allocator_traits<std::allocator<T>> : public std::allocator_traits<std::allocator<T>>
{
private:
  using superclass = std::allocator_traits<std::allocator<T>>;

public:
  using allocator_type                         = typename superclass::allocator_type;
  using value_type                             = typename superclass::value_type;
  using pointer                                = typename superclass::pointer;
  using const_pointer                          = typename superclass::const_pointer;
  using void_pointer                           = typename superclass::void_pointer;
  using const_void_pointer                     = typename superclass::const_void_pointer;
  using difference_type                        = typename superclass::difference_type;
  using size_type                              = typename superclass::size_type;
  using propagate_on_container_swap            = typename superclass::propagate_on_container_swap;
  using propagate_on_container_copy_assignment = typename superclass::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment = typename superclass::propagate_on_container_move_assignment;

  // std::allocator_traits added this in C++17, but thrust::allocator_traits defines
  // it unconditionally.
  using is_always_equal =
    typename eval_if<allocator_traits_detail::has_is_always_equal<allocator_type>::value,
                     allocator_traits_detail::nested_is_always_equal<allocator_type>,
                     ::cuda::std::is_empty<allocator_type>>::type;

  // std::allocator_traits doesn't provide these, but
  // thrust::detail::allocator_traits does. These used to be part of the
  // std::allocator API but were deprecated in C++17.
  using reference       = typename pointer_traits<pointer>::reference;
  using const_reference = typename pointer_traits<const_pointer>::reference;

  template <typename U>
  using rebind_alloc = std::allocator<U>;
  template <typename U>
  using rebind_traits = allocator_traits<std::allocator<U>>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE static pointer allocate(allocator_type& a, size_type n)
  {
    return superclass::allocate(a, n);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE static pointer allocate(allocator_type& a, size_type n, const_void_pointer hint)
  {
    return superclass::allocate(a, n, hint);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE static void deallocate(allocator_type& a, pointer p, size_type n) noexcept
  {
    superclass::deallocate(a, p, n);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <typename U, typename... Args>
  _CCCL_HOST_DEVICE static void construct(allocator_type& a, U* p, Args&&... args)
  {
    superclass::construct(a, p, THRUST_FWD(args)...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <typename U>
  _CCCL_HOST_DEVICE static void destroy(allocator_type& a, U* p) noexcept
  {
    superclass::destroy(a, p);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE static size_type max_size(const allocator_type& a)
  {
    return superclass::max_size(a);
  }
};

// we consider a type an allocator if T::value_type exists
// it doesn't make much sense (containers, which are not allocators, will fulfill this requirement),
// but allocator_traits is specified to work for any type with that nested alias
template <typename T>
struct is_allocator : allocator_traits_detail::has_value_type<T>
{};

// XXX consider moving this non-standard functionality inside allocator_traits
template <typename Alloc>
struct allocator_system
{
  // the type of the allocator's system
  using type = typename eval_if<allocator_traits_detail::has_system_type<Alloc>::value,
                                allocator_traits_detail::nested_system_type<Alloc>,
                                thrust::iterator_system<typename allocator_traits<Alloc>::pointer>>::type;

  // the type that get returns
  using get_result_type =
    typename eval_if<allocator_traits_detail::has_member_system<Alloc>::value,
                     ::cuda::std::add_lvalue_reference<type>,
                     ::cuda::std::type_identity<type>>::type;

  _CCCL_HOST_DEVICE inline static get_result_type get(Alloc& a)
  {
    return allocator_traits_detail::system(a);
  }
};
} // namespace detail
THRUST_NAMESPACE_END
