//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_DYNAMIC_ANY_CAST_H
#define __CUDAX_DETAIL_BASIC_ANY_DYNAMIC_ANY_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__utility/basic_any/access.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>

namespace cuda::experimental
{
///
/// __dynamic_any_cast_fn
///
template <class _DstInterface>
struct __dynamic_any_cast_fn
{
  /// @throws bad_any_cast when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface>&& __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(_CUDA_VSTD::move(__src), __dst);
    return __dst;
  }

  /// @throws bad_any_cast when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>&, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface>& __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }

  /// @throws bad_any_cast when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const&, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface> const& __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }

  /// @returns \c nullptr when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>*, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface>* __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }

  /// @returns \c nullptr when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const*, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface> const* __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }
};

///
/// dynamic_any_cast
///
/// @throws bad_any_cast when \c from cannot be dynamically cast to a
/// \c basic_any<_DstInterface>
///
template <class _DstInterface>
inline constexpr __dynamic_any_cast_fn<_DstInterface> dynamic_any_cast{};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_DYNAMIC_ANY_CAST_H
