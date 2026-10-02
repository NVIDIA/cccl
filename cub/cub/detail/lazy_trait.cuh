// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// use this whenever you need to lazily evaluate a trait. E.g., as an alternative in replace_if_use_default.
template <template <typename...> typename Trait, typename... Args>
struct lazy_trait
{
  using type = Trait<Args...>;
};
} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
