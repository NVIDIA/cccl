// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
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

#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename Allocator, template <typename> class BaseSystem>
struct execute_with_allocator : BaseSystem<execute_with_allocator<Allocator, BaseSystem>>
{
private:
  using super_t = BaseSystem<execute_with_allocator<Allocator, BaseSystem>>;

  Allocator alloc;

public:
  _CCCL_HOST_DEVICE execute_with_allocator(super_t const& super, Allocator alloc_)
      : super_t(super)
      , alloc(alloc_)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE execute_with_allocator(Allocator alloc_)
      : alloc(alloc_)
  {}

  _CCCL_HOST_DEVICE ::cuda::std::remove_reference_t<Allocator>& get_allocator()
  {
    return alloc;
  }
};
} // namespace detail

THRUST_NAMESPACE_END
