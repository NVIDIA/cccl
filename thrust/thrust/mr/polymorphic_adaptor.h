// SPDX-FileCopyrightText: Copyright (c) 2018-2019, NVIDIA Corporation. All rights reserved.
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

#include <thrust/mr/memory_resource.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{
template <typename Pointer = void*>
class polymorphic_adaptor_resource final : public memory_resource<Pointer>
{
public:
  polymorphic_adaptor_resource(memory_resource<Pointer>* t)
      : upstream_resource(t)
  {}

  virtual Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    return upstream_resource->allocate(bytes, alignment);
  }

  virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
  {
    return upstream_resource->deallocate(p, bytes, alignment);
  }

  _CCCL_HOST_DEVICE virtual bool do_is_equal(const memory_resource<Pointer>& other) const noexcept override
  {
    return upstream_resource->is_equal(other);
  }

private:
  memory_resource<Pointer>* upstream_resource;
};
} // namespace mr
THRUST_NAMESPACE_END
