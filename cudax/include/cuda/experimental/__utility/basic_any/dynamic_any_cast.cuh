//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__utility/basic_any/access.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! \brief Casts one basic_any reference type to another basic_any type using
//! runtime information to determine the validity of the conversion.
//!
//! \throws bad_any_cast when \c __src cannot be dynamically cast to a
//! \c basic_any<_DstInterface>.
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<basic_any<_SrcInterface>, basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_HOST_API auto dynamic_any_cast(basic_any<_SrcInterface>&& __src) -> basic_any<_DstInterface>
{
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(_CUDA_VSTD::move(__src), __dst);
  return __dst;
}

//! \overload
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<basic_any<_SrcInterface>&, basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_HOST_API auto dynamic_any_cast(basic_any<_SrcInterface>& __src) -> basic_any<_DstInterface>
{
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

//! \overload
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const&, basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_HOST_API auto dynamic_any_cast(basic_any<_SrcInterface> const& __src) -> basic_any<_DstInterface>
{
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

//! \brief Casts a `basic_any<_SrcInterface>*` into a `basic_any<_DstInterface>`
//! using runtime information to determine the validity of the conversion.
//!
//! \pre \c _DstInterface must be a pointer type.
//!
//! \returns \c nullptr when \c __src cannot be dynamically cast to a
//! \c basic_any<_DstInterface>.
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<basic_any<_SrcInterface>*, basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_HOST_API auto dynamic_any_cast(basic_any<_SrcInterface>* __src) -> basic_any<_DstInterface>
{
  static_assert(_CUDA_VSTD::is_pointer_v<_DstInterface>,
                "when dynamic_any_cast-ing from a pointer to a basic_any, the destination type must be a pointer to an "
                "interface type.");
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

//! \overload
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const*, basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_HOST_API auto dynamic_any_cast(basic_any<_SrcInterface> const* __src) -> basic_any<_DstInterface>
{
  static_assert(_CUDA_VSTD::is_pointer_v<_DstInterface>,
                "when dynamic_any_cast-ing from a pointer to a basic_any, the destination type must be a pointer to an "
                "interface type.");
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_BASIC_ANY_DYNAMIC_ANY_CAST_H
