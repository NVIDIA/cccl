//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_DEVICE_REF
#define _CUDAX__DEVICE_DEVICE_REF

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>

namespace cuda::experimental
{
class device;

namespace detail
{
template <::cudaDeviceAttr _Attr>
struct __dev_attr;
} // namespace detail

//! @brief A non-owning representation of a CUDA device
class device_ref
{
  friend class device;

  int __id_ = 0;

public:
  //! @brief Create a `device_ref` object from a native device ordinal.
  /*implicit*/ constexpr device_ref(int __id) noexcept
      : __id_(__id)
  {}

  //! @brief Retrieve the native ordinal of the `device_ref`
  //!
  //! @return int The native device ordinal held by the `device_ref` object
  _CCCL_NODISCARD constexpr int get() const noexcept
  {
    return __id_;
  }

  //! @brief Retrieve the specified attribute for the device
  //!
  //! @param __attr The attribute to query. See `device::attrs` for the available
  //!        attributes.
  //!
  //! @throws cuda_error if the attribute query fails
  //!
  //! @sa device::attrs
  template <::cudaDeviceAttr _Attr>
  _CCCL_NODISCARD auto attr([[maybe_unused]] detail::__dev_attr<_Attr> __attr) const
  {
    int __value = 0;
    _CCCL_TRY_CUDA_API(::cudaDeviceGetAttribute, "failed to get device attribute", &__value, _Attr, get());
    return static_cast<typename detail::__dev_attr<_Attr>::type>(__value);
  }

  //! @overload
  template <::cudaDeviceAttr _Attr>
  _CCCL_NODISCARD auto attr() const
  {
    return attr(detail::__dev_attr<_Attr>());
  }
};

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE_REF
