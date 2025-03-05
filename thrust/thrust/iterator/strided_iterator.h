// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA Corporation
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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
struct deref
{
  template <typename It>
  _CCCL_HOST_DEVICE auto operator()(It it) const -> it_reference_t<It>
  {
    return *it;
  }
};
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

template <typename Iterator, typename Step = detail::empty>
using strided_iterator =
  transform_iterator<detail::deref, counting_iterator<Iterator, use_default, use_default, use_default, Step>>;

//! Constructs a strided_iterator with a runtime stride
template <typename Iterator, typename Stride>
_CCCL_HOST_DEVICE auto make_strided_iterator(Iterator it, Stride stride)
{
  using CI = counting_iterator<Iterator, use_default, use_default, use_default, detail::runtime_step_holder<Stride>>;
  return strided_iterator<Iterator, detail::runtime_step_holder<Stride>>(CI{it, {stride}}, detail::deref{});
}

//! Constructs a strided_iterator with a compile-time stride
template <auto Value, typename Iterator>
_CCCL_HOST_DEVICE auto make_ct_strided_iterator(Iterator it)
{
  using CI =
    counting_iterator<Iterator,
                      use_default,
                      use_default,
                      use_default,
                      detail::compile_time_step_holder<decltype(Value), Value>>;
  return strided_iterator<Iterator, detail::compile_time_step_holder<decltype(Value), Value>>(
    CI{it, {}}, detail::deref{});
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
