// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/tabulate_output_iterator.h>

THRUST_NAMESPACE_BEGIN

template <typename BinaryFunction, typename System, typename DifferenceT>
class tabulate_output_iterator;

namespace detail
{

// Proxy reference that invokes a BinaryFunction with the index of the dereferenced iterator and the assigned value
template <typename BinaryFunction, typename DifferenceT>
class tabulate_output_iterator_proxy
{
public:
  _CCCL_HOST_DEVICE tabulate_output_iterator_proxy(BinaryFunction fun, DifferenceT index)
      : fun(fun)
      , index(index)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T>
  _CCCL_HOST_DEVICE tabulate_output_iterator_proxy operator=(const T& x)
  {
    fun(index, x);
    return *this;
  }

private:
  BinaryFunction fun;
  DifferenceT index;
};

// Alias template for the iterator_adaptor instantiation to be used for tabulate_output_iterator
template <typename BinaryFunction, typename System, typename DifferenceT>
using tabulate_output_iterator_base =
  thrust::iterator_adaptor<tabulate_output_iterator<BinaryFunction, System, DifferenceT>,
                           counting_iterator<DifferenceT>,
                           thrust::use_default,
                           System,
                           thrust::use_default,
                           tabulate_output_iterator_proxy<BinaryFunction, DifferenceT>>;

// Register tabulate_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
template <class BinaryFunction, class OutputIterator>
struct is_proxy_reference<tabulate_output_iterator_proxy<BinaryFunction, OutputIterator>>
    : public thrust::detail::true_type
{};

} // namespace detail
THRUST_NAMESPACE_END
