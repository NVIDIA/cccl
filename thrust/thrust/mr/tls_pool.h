// SPDX-FileCopyrightText: Copyright (c) 2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file tls_pool.h
 *  \brief A function wrapping a thread local instance of a \p unsynchronized_pool_resource.
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
#include <thrust/mr/pool.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{
/*! \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! Potentially constructs, if not yet created, and then returns the address of a thread-local \p
 * unsynchronized_pool_resource,
 *
 *  \tparam Upstream the template argument to the pool template
 *  \param upstream the argument to the constructor, if invoked
 */
template <typename Upstream, typename Bookkeeper>
_CCCL_HOST thrust::mr::unsynchronized_pool_resource<Upstream>& tls_pool(Upstream* upstream = nullptr)
{
  static thread_local auto adaptor = [&] {
    assert(upstream);
    return thrust::mr::unsynchronized_pool_resource<Upstream>(upstream);
  }();

  return adaptor;
}

/*! \}
 */
} // namespace mr
THRUST_NAMESPACE_END
