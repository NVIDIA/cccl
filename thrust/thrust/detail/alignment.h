// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// \file alignment.h
/// \brief Type-alignment utilities.

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
namespace detail
{
/// \p aligned_reinterpret_cast `reinterpret_cast`s \p u of type \p U to `void*`
/// and then `reinterpret_cast`s the result to \p T. The indirection through
/// `void*` suppresses compiler warnings when the alignment requirement of \p *u
/// is less than the alignment requirement of \p *t. The caller of
/// \p aligned_reinterpret_cast is responsible for ensuring that the alignment
/// requirements are actually satisfied.
template <typename T, typename U>
_CCCL_HOST_DEVICE T aligned_reinterpret_cast(U u)
{
  return reinterpret_cast<T>(reinterpret_cast<void*>(u));
}
} // end namespace detail
THRUST_NAMESPACE_END
