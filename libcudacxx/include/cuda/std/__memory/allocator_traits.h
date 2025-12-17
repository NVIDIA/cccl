// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_ALLOCATOR_TRAITS_H
#define _CUDA_STD___MEMORY_ALLOCATOR_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/allocator.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstring>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NV_DIAG_SUPPRESS(1215)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(NAME, PROPERTY) \
  template <class _Tp, class = void>                         \
  inline constexpr bool NAME##_v = false;                    \
  template <class _Tp>                                       \
  inline constexpr bool NAME##_v<_Tp, void_t<typename _Tp::PROPERTY>> = true;

// __pointer
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_pointer, pointer)
template <class _Tp, class _Alloc, class _RawAlloc = remove_reference_t<_Alloc>, bool = __has_pointer_v<_RawAlloc>>
struct __pointer
{
  using type _CCCL_NODEBUG_ALIAS = typename _RawAlloc::pointer;
};
template <class _Tp, class _Alloc, class _RawAlloc>
struct __pointer<_Tp, _Alloc, _RawAlloc, false>
{
  using type _CCCL_NODEBUG_ALIAS = _Tp*;
};

// __const_pointer
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_const_pointer, const_pointer)
template <class _Tp, class _Ptr, class _Alloc, bool = __has_const_pointer_v<_Alloc>>
struct __const_pointer
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::const_pointer;
};
template <class _Tp, class _Ptr, class _Alloc>
struct __const_pointer<_Tp, _Ptr, _Alloc, false>
{
  using type _CCCL_NODEBUG_ALIAS = typename pointer_traits<_Ptr>::template rebind<const _Tp>;
};

// __void_pointer
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_void_pointer, void_pointer)
template <class _Ptr, class _Alloc, bool = __has_void_pointer_v<_Alloc>>
struct __void_pointer
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::void_pointer;
};
template <class _Ptr, class _Alloc>
struct __void_pointer<_Ptr, _Alloc, false>
{
  using type _CCCL_NODEBUG_ALIAS = typename pointer_traits<_Ptr>::template rebind<void>;
};

// __const_void_pointer
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_const_void_pointer, const_void_pointer)
template <class _Ptr, class _Alloc, bool = __has_const_void_pointer_v<_Alloc>>
struct __const_void_pointer
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::const_void_pointer;
};
template <class _Ptr, class _Alloc>
struct __const_void_pointer<_Ptr, _Alloc, false>
{
  using type _CCCL_NODEBUG_ALIAS = typename pointer_traits<_Ptr>::template rebind<const void>;
};

// __size_type
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_size_type, size_type)
template <class _Alloc, class _DiffType, bool = __has_size_type_v<_Alloc>>
struct __size_type : make_unsigned<_DiffType>
{};
template <class _Alloc, class _DiffType>
struct __size_type<_Alloc, _DiffType, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::size_type;
};

// __alloc_traits_difference_type
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_alloc_traits_difference_type, difference_type)
template <class _Alloc, class _Ptr, bool = __has_alloc_traits_difference_type_v<_Alloc>>
struct __alloc_traits_difference_type
{
  using type _CCCL_NODEBUG_ALIAS = typename pointer_traits<_Ptr>::difference_type;
};
template <class _Alloc, class _Ptr>
struct __alloc_traits_difference_type<_Alloc, _Ptr, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::difference_type;
};

// __propagate_on_container_copy_assignment
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_propagate_on_container_copy_assignment,
                                     propagate_on_container_copy_assignment)
template <class _Alloc, bool = __has_propagate_on_container_copy_assignment_v<_Alloc>>
struct __propagate_on_container_copy_assignment : false_type
{};
template <class _Alloc>
struct __propagate_on_container_copy_assignment<_Alloc, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::propagate_on_container_copy_assignment;
};

// __propagate_on_container_move_assignment
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_propagate_on_container_move_assignment,
                                     propagate_on_container_move_assignment)
template <class _Alloc, bool = __has_propagate_on_container_move_assignment_v<_Alloc>>
struct __propagate_on_container_move_assignment : false_type
{};
template <class _Alloc>
struct __propagate_on_container_move_assignment<_Alloc, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::propagate_on_container_move_assignment;
};

// __propagate_on_container_swap
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_propagate_on_container_swap, propagate_on_container_swap)
template <class _Alloc, bool = __has_propagate_on_container_swap_v<_Alloc>>
struct __propagate_on_container_swap : false_type
{};
template <class _Alloc>
struct __propagate_on_container_swap<_Alloc, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::propagate_on_container_swap;
};

// __is_always_equal
_LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX(__has_is_always_equal, is_always_equal)
template <class _Alloc, bool = __has_is_always_equal_v<_Alloc>>
struct __is_always_equal : is_empty<_Alloc>
{};
template <class _Alloc>
struct __is_always_equal<_Alloc, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc::is_always_equal;
};

// __allocator_traits_rebind
template <class _Tp, class _Up>
_CCCL_CONCEPT __has_member_rebind_other =
  _CCCL_REQUIRES_EXPR((_Tp, _Up))(typename(typename _Tp::template rebind<_Up>::other));

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _Up, bool = __has_member_rebind_other<_Tp, _Up>>
struct __allocator_traits_rebind
{
  static_assert(__has_member_rebind_other<_Tp, _Up>, "This allocator has to implement rebind");
  using type _CCCL_NODEBUG_ALIAS = typename _Tp::template rebind<_Up>::other;
};
template <template <class, class...> class _Alloc, class _Tp, class... _Args, class _Up>
struct __allocator_traits_rebind<_Alloc<_Tp, _Args...>, _Up, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Alloc<_Tp, _Args...>::template rebind<_Up>::other;
};
template <template <class, class...> class _Alloc, class _Tp, class... _Args, class _Up>
struct __allocator_traits_rebind<_Alloc<_Tp, _Args...>, _Up, false>
{
  using type _CCCL_NODEBUG_ALIAS = _Alloc<_Up, _Args...>;
};

template <class _Alloc, class _Tp>
using __allocator_traits_rebind_t = typename __allocator_traits_rebind<_Alloc, _Tp>::type;

// __has_allocate_hint
template <class _Alloc, class _SizeType, class _ConstVoidPtr, class = void>
inline constexpr bool __has_allocate_hint = false;

template <class _Alloc, class _SizeType, class _ConstVoidPtr>
inline constexpr bool __has_allocate_hint<
  _Alloc,
  _SizeType,
  _ConstVoidPtr,
  void_t<decltype((void) ::cuda::std::declval<_Alloc>().allocate(
    ::cuda::std::declval<_SizeType>(), ::cuda::std::declval<_ConstVoidPtr>()))>> = true;

// __has_construct
template <class, class _Alloc, class... _Args>
inline constexpr bool __has_construct_impl = false;

template <class _Alloc, class... _Args>
inline constexpr bool
  __has_construct_impl<decltype((void) ::cuda::std::declval<_Alloc>().construct(::cuda::std::declval<_Args>()...)),
                       _Alloc,
                       _Args...> = true;

template <class _Alloc, class... _Args>
inline constexpr bool __has_construct = __has_construct_impl<void, _Alloc, _Args...>;

// __has_destroy
template <class _Alloc, class _Pointer, class = void>
inline constexpr bool __has_destroy = false;

template <class _Alloc, class _Pointer>
inline constexpr bool
  __has_destroy<_Alloc,
                _Pointer,
                void_t<decltype((void) ::cuda::std::declval<_Alloc>().destroy(::cuda::std::declval<_Pointer>()))>> =
    true;

// __has_max_size
template <class _Alloc, class = void>
inline constexpr bool __has_max_size = false;

template <class _Alloc>
inline constexpr bool __has_max_size<_Alloc, void_t<decltype((void) ::cuda::std::declval<_Alloc&>().max_size())>> =
  true;

// __has_select_on_container_copy_construction
template <class _Alloc, class = void>
inline constexpr bool __has_select_on_container_copy_construction = false;

template <class _Alloc>
inline constexpr bool __has_select_on_container_copy_construction<
  _Alloc,
  void_t<decltype((void) ::cuda::std::declval<_Alloc>().select_on_container_copy_construction())>> = true;

template <class _Tp>
_CCCL_API constexpr _Tp* __to_raw_pointer(_Tp* __p) noexcept
{
  return __p;
}

#if _CCCL_STD_VER <= 2017
template <class _Pointer>
_CCCL_API inline typename pointer_traits<_Pointer>::element_type* __to_raw_pointer(_Pointer __p) noexcept
{
  return ::cuda::std::__to_raw_pointer(__p.operator->());
}
#else // ^^^ C++17 ^^^ / vvv C++20 vvv
template <class _Pointer>
_CCCL_API inline auto __to_raw_pointer(const _Pointer& __p) noexcept
  -> decltype(pointer_traits<_Pointer>::to_address(__p))
{
  return pointer_traits<_Pointer>::to_address(__p);
}

template <class _Pointer, class... _None>
_CCCL_API inline auto __to_raw_pointer(const _Pointer& __p, _None...) noexcept
{
  return ::cuda::std::__to_raw_pointer(__p.operator->());
}
#endif // _CCCL_STD_VER >= 2020

// __is_cpp17_move_insertable
template <class _Alloc, class = void>
inline constexpr bool __is_cpp17_move_insertable = is_move_constructible_v<typename _Alloc::value_type>;

template <class _Alloc>
inline constexpr bool __is_cpp17_move_insertable<
  _Alloc,
  enable_if_t<__is_cuda_std_allocator_v<_Alloc>
              && __has_construct<_Alloc, typename _Alloc::value_type*, typename _Alloc::value_type&&>>> = true;

// __is_cpp17_copy_insertable
template <class _Alloc, class = void>
inline constexpr bool __is_cpp17_copy_insertable =
  is_copy_constructible_v<typename _Alloc::value_type> && __is_cpp17_move_insertable<_Alloc>;

template <class _Alloc>
inline constexpr bool __is_cpp17_copy_insertable<
  _Alloc,
  enable_if_t<!__is_cuda_std_allocator_v<_Alloc>
              && __has_construct<_Alloc, typename _Alloc::value_type*, const typename _Alloc::value_type&>>> =
  __is_cpp17_move_insertable<_Alloc>;

template <class _Alloc>
struct _CCCL_TYPE_VISIBILITY_DEFAULT allocator_traits
{
  using allocator_type     = _Alloc;
  using value_type         = typename allocator_type::value_type;
  using pointer            = typename __pointer<value_type, allocator_type>::type;
  using const_pointer      = typename __const_pointer<value_type, pointer, allocator_type>::type;
  using void_pointer       = typename __void_pointer<pointer, allocator_type>::type;
  using const_void_pointer = typename __const_void_pointer<pointer, allocator_type>::type;
  using difference_type    = typename __alloc_traits_difference_type<allocator_type, pointer>::type;
  using size_type          = typename __size_type<allocator_type, difference_type>::type;
  using propagate_on_container_copy_assignment =
    typename __propagate_on_container_copy_assignment<allocator_type>::type;
  using propagate_on_container_move_assignment =
    typename __propagate_on_container_move_assignment<allocator_type>::type;
  using propagate_on_container_swap = typename __propagate_on_container_swap<allocator_type>::type;
  using is_always_equal             = typename __is_always_equal<allocator_type>::type;

  template <class _Tp>
  using rebind_alloc = __allocator_traits_rebind_t<allocator_type, _Tp>;
  template <class _Tp>
  using rebind_traits = allocator_traits<rebind_alloc<_Tp>>;

  [[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static pointer allocate(allocator_type& __a, size_type __n)
  {
    return __a.allocate(__n);
  }

  [[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static pointer
  allocate(allocator_type& __a, size_type __n, [[maybe_unused]] const_void_pointer __hint)
  {
    if constexpr (__has_allocate_hint<_Alloc, size_type, const_void_pointer>)
    {
      return __a.allocate(__n, __hint);
    }
    else
    {
      return __a.allocate(__n);
    }
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static void deallocate(allocator_type& __a, pointer __p, size_type __n) noexcept
  {
    __a.deallocate(__p, __n);
  }

  template <class _Tp, class... _Args>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static void
  construct([[maybe_unused]] allocator_type& __a, _Tp* __p, _Args&&... __args)
  {
    if constexpr (__has_construct<allocator_type, _Tp*, _Args...>)
    {
      __a.construct(__p, ::cuda::std::forward<_Args>(__args)...);
    }
    else
    {
      ::cuda::std::__construct_at(__p, ::cuda::std::forward<_Args>(__args)...);
    }
  }

  template <class _Tp>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static void destroy([[maybe_unused]] allocator_type& __a, _Tp* __p) noexcept
  {
    if constexpr (__has_destroy<allocator_type, _Tp*>)
    {
      __a.destroy(__p);
    }
    else
    {
      ::cuda::std::__destroy_at(__p);
    }
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static size_type max_size([[maybe_unused]] const allocator_type& __a) noexcept
  {
    if constexpr (__has_max_size<const _Alloc>)
    {
      return __a.max_size();
    }
    else
    {
      return numeric_limits<size_type>::max() / sizeof(value_type);
    }
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static allocator_type
  select_on_container_copy_construction(const allocator_type& __a)
  {
    if constexpr (__has_select_on_container_copy_construction<const _Alloc>)
    {
      return __a.select_on_container_copy_construction();
    }
    else
    {
      return __a;
    }
  }

  template <class _Ptr>
  _CCCL_API inline static void
  __construct_forward_with_exception_guarantees(allocator_type& __a, _Ptr __begin1, _Ptr __end1, _Ptr& __begin2)
  {
    static_assert(__is_cpp17_move_insertable<allocator_type>,
                  "The specified type does not meet the requirements of Cpp17MoveInsertible");
    for (; __begin1 != __end1; ++__begin1, (void) ++__begin2)
    {
      construct(__a,
                ::cuda::std::__to_raw_pointer(__begin2),
#if !_CCCL_HAS_EXCEPTIONS()
                ::cuda::std::move(*__begin1)
#else // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^ / vvv _CCCL_HAS_EXCEPTIONS() vvv
                ::cuda::std::move_if_noexcept(*__begin1)
#endif // _CCCL_HAS_EXCEPTIONS()
      );
    }
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((__is_cuda_std_allocator_v<allocator_type>
                  || !__has_construct<allocator_type, _Tp*, _Tp>) _CCCL_AND is_trivially_move_constructible_v<_Tp>)
  _CCCL_API inline static void
  __construct_forward_with_exception_guarantees(allocator_type&, _Tp* __begin1, _Tp* __end1, _Tp*& __begin2)
  {
    ptrdiff_t _Np = __end1 - __begin1;
    if (_Np > 0)
    {
      ::cuda::std::memcpy(__begin2, __begin1, _Np * sizeof(_Tp));
      __begin2 += _Np;
    }
  }

  template <class _Iter, class _Ptr>
  _CCCL_API inline static void
  __construct_range_forward(allocator_type& __a, _Iter __begin1, _Iter __end1, _Ptr& __begin2)
  {
    for (; __begin1 != __end1; ++__begin1, (void) ++__begin2)
    {
      construct(__a, ::cuda::std::__to_raw_pointer(__begin2), *__begin1);
    }
  }

  _CCCL_TEMPLATE(class _SourceTp,
                 class _DestTp,
                 class _RawSourceTp = remove_const_t<_SourceTp>,
                 class _RawDestTp   = remove_const_t<_DestTp>)
  _CCCL_REQUIRES(is_trivially_move_constructible_v<_DestTp> _CCCL_AND is_same_v<_RawSourceTp, _RawDestTp> _CCCL_AND(
    __is_cuda_std_allocator_v<allocator_type> || !__has_construct<allocator_type, _DestTp*, _SourceTp&>))
  _CCCL_API inline static void
  __construct_range_forward(allocator_type&, _SourceTp* __begin1, _SourceTp* __end1, _DestTp*& __begin2)
  {
    ptrdiff_t _Np = __end1 - __begin1;
    if (_Np > 0)
    {
      ::cuda::std::memcpy(const_cast<_RawDestTp*>(__begin2), __begin1, _Np * sizeof(_DestTp));
      __begin2 += _Np;
    }
  }

  template <class _Ptr>
  _CCCL_API inline static void
  __construct_backward_with_exception_guarantees(allocator_type& __a, _Ptr __begin1, _Ptr __end1, _Ptr& __end2)
  {
    static_assert(__is_cpp17_move_insertable<allocator_type>,
                  "The specified type does not meet the requirements of Cpp17MoveInsertable");
    while (__end1 != __begin1)
    {
      construct(__a,
                ::cuda::std::__to_raw_pointer(__end2 - 1),
#if !_CCCL_HAS_EXCEPTIONS()
                ::cuda::std::move(*--__end1)
#else // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^ / vvv _CCCL_HAS_EXCEPTIONS() vvv
                ::cuda::std::move_if_noexcept(*--__end1)
#endif // _CCCL_HAS_EXCEPTIONS()
      );
      --__end2;
    }
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((__is_cuda_std_allocator_v<allocator_type>
                  || !__has_construct<allocator_type, _Tp*, _Tp>) _CCCL_AND is_trivially_move_constructible_v<_Tp>)
  _CCCL_API inline static void
  __construct_backward_with_exception_guarantees(allocator_type&, _Tp* __begin1, _Tp* __end1, _Tp*& __end2)
  {
    ptrdiff_t _Np = __end1 - __begin1;
    __end2 -= _Np;
    if (_Np > 0)
    {
      ::cuda::std::memcpy(__end2, __begin1, _Np * sizeof(_Tp));
    }
  }
};
_CCCL_SUPPRESS_DEPRECATED_POP

template <class _Traits, class _Tp>
using __rebind_alloc _CCCL_NODEBUG_ALIAS = typename _Traits::template rebind_alloc<_Tp>;

template <class _Traits, class _Tp>
struct __rebind_alloc_helper
{
  using type _CCCL_NODEBUG_ALIAS = typename _Traits::template rebind_alloc<_Tp>;
};

#undef _LIBCUDACXX_ALLOCATOR_TRAITS_HAS_XXX

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_END_NV_DIAG_SUPPRESS()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_ALLOCATOR_TRAITS_H
