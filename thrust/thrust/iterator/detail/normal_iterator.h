// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! \file normal_iterator.h
//! \brief Defines the interface to an iterator class which adapts a pointer type.

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
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/pointer_traits.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename Pointer>
class normal_iterator : public iterator_adaptor<normal_iterator<Pointer>, Pointer>
{
  using super_t = iterator_adaptor<normal_iterator<Pointer>, Pointer>;

public:
  _CCCL_HIDE_FROM_ABI normal_iterator() = default;

  _CCCL_HOST_DEVICE normal_iterator(Pointer p)
      : super_t(p)
  {}

  template <typename OtherPointer>
  _CCCL_HOST_DEVICE
  normal_iterator(const normal_iterator<OtherPointer>& other, enable_if_convertible_t<OtherPointer, Pointer>* = nullptr)
      : super_t(other.base())
  {}
};

template <typename Pointer>
_CCCL_HOST_DEVICE normal_iterator<Pointer> make_normal_iterator(Pointer ptr)
{
  return normal_iterator<Pointer>(ptr);
}
} // namespace detail

template <typename Pointer>
struct proclaim_contiguous_iterator<detail::normal_iterator<Pointer>> : true_type
{};

THRUST_NAMESPACE_END

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Specialize pointer traits for everything that has the raw_pointer alias
template <class Pointer>
struct pointer_traits<THRUST_NS_QUALIFIER::detail::normal_iterator<Pointer>>
{
  using pointer         = THRUST_NS_QUALIFIER::detail::normal_iterator<Pointer>;
  using element_type    = Pointer;
  using difference_type = ptrdiff_t;

  template <typename U>
  using rebind = typename THRUST_NS_QUALIFIER::detail::rebind_pointer<pointer, U>::type;

  [[nodiscard]] _CCCL_API inline static pointer pointer_to(Pointer& r) noexcept(noexcept(::cuda::std::addressof(r)))
  {
    return static_cast<element_type*>(::cuda::std::addressof(r));
  }

  //! @brief Retrieve the address of the element pointed at by an thrust pointer
  //! @param iter A thrust::detail::normal_iterator
  //! @return A pointer to the element pointed to by the thrust pointer
  [[nodiscard]] _CCCL_API static constexpr auto* to_address(const pointer iter) noexcept
  {
    return ::cuda::std::to_address(iter.base());
  }
};

_CCCL_END_NAMESPACE_CUDA_STD
