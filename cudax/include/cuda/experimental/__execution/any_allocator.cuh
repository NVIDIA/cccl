//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_ANY_ALLOCATOR
#define __CUDAX_EXECUTION_ANY_ALLOCATOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/__utility/basic_any.h>
#include <cuda/std/__fwd/optional.h>
#include <cuda/std/__memory/allocator.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__type_traits/decay.h>

#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Value>
struct any_allocator;

namespace __detail
{
template <class _Allocator, class _Value = typename _Allocator::value_type>
_CCCL_PUBLIC_API auto __any_allocator_allocate(_Allocator& __alloc, size_t __count) -> _Value*
{
  return ::cuda::std::allocator_traits<_Allocator>::allocate(__alloc, __count);
}

template <class _Allocator, class _Value = typename _Allocator::value_type>
_CCCL_PUBLIC_API void __any_allocator_deallocate(_Allocator& __alloc, _Value* __ptr, size_t __count) noexcept
{
  ::cuda::std::allocator_traits<_Allocator>::deallocate(__alloc, static_cast<_Value*>(__ptr), __count);
}

template <class...>
struct __iallocator : __basic_interface<__iallocator, ::cuda::__extends<::cuda::__icopyable<>>>
{
  using value_type = ::cuda::std::byte;

  template <class _Other>
  struct rebind
  {
    static_assert(__same_as<_Other, value_type>);
    using other = __iallocator;
  };

  _CCCL_API auto allocate(size_t __bytes) -> value_type*
  {
    constexpr auto __allocate_vfn = &__any_allocator_allocate<__iallocator<>>;
    return ::cuda::__virtcall<__allocate_vfn>(this, __bytes);
  }

  _CCCL_API void deallocate(value_type* __ptr, size_t __bytes) noexcept
  {
    constexpr auto __deallocate_vfn = &__any_allocator_deallocate<__iallocator<>>;
    ::cuda::__virtcall<__deallocate_vfn>(this, __ptr, __bytes);
  }

  template <class _Allocator>
  using overrides =
    __overrides_for<_Allocator, &__any_allocator_allocate<_Allocator>, &__any_allocator_deallocate<_Allocator>>;
};

using __any_allocator = ::cuda::__basic_any<__iallocator<>>;

template <class _Allocator>
_CCCL_CONCEPT __is_any_allocator = __is_specialization_of_v<_Allocator, execution::any_allocator>;
} // namespace __detail

template <class _Value>
struct any_allocator : private __detail::__any_allocator
{
  using value_type = _Value;

  template <class _Other>
  struct rebind
  {
    using other = any_allocator<_Other>;
  };

  _CCCL_API any_allocator(::cuda::std::allocator<void>) noexcept
      : __detail::__any_allocator{::cuda::std::allocator<::cuda::std::byte>{}}
  {}

  _CCCL_TEMPLATE(class _Allocator)
  _CCCL_REQUIRES((!__detail::__is_any_allocator<_Allocator>) //
                 _CCCL_AND(!::cuda::std::__is_cuda_std_optional_v<_Allocator>)
                   _CCCL_AND ::cuda::__satisfies<_Allocator, __detail::__iallocator<>>)
  _CCCL_API any_allocator(_Allocator __alloc)
      : __detail::__any_allocator{__byte_allocator_t<_Allocator>(static_cast<_Allocator&&>(__alloc))}
  {}

  _CCCL_TEMPLATE(class _OtherValue)
  _CCCL_REQUIRES(__not_same_as<_OtherValue, _Value>)
  _CCCL_API any_allocator(any_allocator<_OtherValue> __other) noexcept
      : __detail::__any_allocator{static_cast<__detail::__any_allocator&&>(__other)}
  {}

  _CCCL_API auto allocate(size_t __count) -> _Value*
  {
    return reinterpret_cast<_Value*>(this->__basic_any::allocate(__count * sizeof(_Value)));
  }

  _CCCL_API void deallocate(_Value* __ptr, size_t __count) noexcept
  {
    this->__basic_any::deallocate(reinterpret_cast<::cuda::std::byte*>(__ptr), __count * sizeof(_Value));
  }

private:
  template <class>
  friend struct any_allocator;

  template <class _Allocator>
  using __byte_allocator_t = ::cuda::std::__rebind_alloc<::cuda::std::allocator_traits<_Allocator>, ::cuda::std::byte>;
};

template <class _Allocator>
_CCCL_HOST_DEVICE any_allocator(_Allocator) -> any_allocator<typename _Allocator::value_type>;

_CCCL_HOST_DEVICE any_allocator(::cuda::std::allocator<void>) -> any_allocator<::cuda::std::byte>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_ANY_ALLOCATOR
