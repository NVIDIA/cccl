// SPDX-FileCopyrightText: Copyright (c) 2018, NVIDIA Corporation. All rights reserved.
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
#include <thrust/mr/validator.h>

#include <cuda/std/__memory/pointer_traits.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{
template <typename Upstream, typename Pointer>
class fancy_pointer_resource final
    : public memory_resource<Pointer>
    , private validator<Upstream>
{
public:
  fancy_pointer_resource()
      : m_upstream(get_global_resource<Upstream>())
  {}

  fancy_pointer_resource(Upstream* upstream)
      : m_upstream(upstream)
  {}

  [[nodiscard]] virtual Pointer
  do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    return static_cast<Pointer>(m_upstream->do_allocate(bytes, alignment));
  }

  virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
  {
    return m_upstream->do_deallocate(
      static_cast<typename Upstream::pointer>(::cuda::std::to_address(p)), bytes, alignment);
  }

private:
  Upstream* m_upstream;
};
} // namespace mr
THRUST_NAMESPACE_END
