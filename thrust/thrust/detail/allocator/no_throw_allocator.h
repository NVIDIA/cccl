// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename BaseAllocator>
struct no_throw_allocator : BaseAllocator
{
private:
  using super_t = BaseAllocator;

public:
  inline _CCCL_HOST_DEVICE no_throw_allocator(const BaseAllocator& other = BaseAllocator())
      : super_t(other)
  {}

  template <typename U>
  struct rebind
  {
    using other = no_throw_allocator<typename super_t::template rebind<U>::other>;
  }; // end rebind

  _CCCL_HOST_DEVICE void deallocate(typename super_t::pointer p, typename super_t::size_type n) noexcept
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      (try { super_t::deallocate(p, n); } // end try
       catch (...){
         // catch anything
       } // end catch
       ),
      (super_t::deallocate(p, n);));
  } // end deallocate()

  inline _CCCL_HOST_DEVICE bool operator==(no_throw_allocator const& other)
  {
    return super_t::operator==(other);
  }

  inline _CCCL_HOST_DEVICE bool operator!=(no_throw_allocator const& other)
  {
    return super_t::operator!=(other);
  }
}; // end no_throw_allocator
} // namespace detail
THRUST_NAMESPACE_END
