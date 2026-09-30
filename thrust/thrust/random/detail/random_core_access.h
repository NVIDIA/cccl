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

THRUST_NAMESPACE_BEGIN

namespace random::detail
{
struct random_core_access
{
  template <typename OStream, typename EngineOrDistribution>
  static OStream& stream_out(OStream& os, const EngineOrDistribution& x)
  {
    return x.stream_out(os);
  }

  template <typename IStream, typename EngineOrDistribution>
  static IStream& stream_in(IStream& is, EngineOrDistribution& x)
  {
    return x.stream_in(is);
  }

  template <typename EngineOrDistribution>
  _CCCL_HOST_DEVICE static bool equal(const EngineOrDistribution& lhs, const EngineOrDistribution& rhs)
  {
    return lhs.equal(rhs);
  }

}; // end random_core_access
} // namespace random::detail

THRUST_NAMESPACE_END
