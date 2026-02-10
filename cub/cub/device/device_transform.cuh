// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/__functional/address_stability.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <typename T>
struct __return_constant
{
  T value;

  template <typename... Args>
  _CCCL_HOST_DEVICE auto operator()(Args&&...) const -> T
  {
    return value;
  }
};
} // namespace detail
CUB_NAMESPACE_END

template <typename T>
struct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type
{};

CUB_NAMESPACE_BEGIN
namespace detail::transform
{
struct get_tuning_query_t
{};
} // namespace detail::transform

//! DeviceTransform provides device-wide, parallel operations for transforming elements tuple-wise from multiple input
//! sequences into an output sequence.
struct DeviceTransform
{
private:
  template <detail::transform::requires_stable_address StableAddress = detail::transform::requires_stable_address::no,
            typename... RandomAccessIteratorsIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Predicate,
            typename TransformOp,
            typename Env>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformInternal(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    Predicate predicate,
    TransformOp transform_op,
    Env env)
  {
    using choose_offset_t = detail::choose_signed_offset<NumItemsT>;
    using offset_t        = typename choose_offset_t::type;

    // Check if the number of items exceeds the range covered by the selected signed offset type
    if (const cudaError_t error = choose_offset_t::is_exceeding_offset_type(num_items); error != cudaSuccess)
    {
      return error;
    }

    using tuning_env_t =
      ::cuda::__call_result_or_t<::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>, Env>;
    using transform_tuning_t = ::cuda::__call_result_or_t<detail::transform::get_tuning_query_t, int, tuning_env_t>;

    if constexpr (!::cuda::std::is_same_v<transform_tuning_t, int>)
    {
      return detail::transform::dispatch<StableAddress>(
        ::cuda::std::move(inputs),
        ::cuda::std::move(output),
        static_cast<offset_t>(num_items),
        ::cuda::std::move(predicate),
        ::cuda::std::move(transform_op),
        ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env).get(),
        transform_tuning_t{});
    }
    else
    {
      return detail::transform::dispatch<StableAddress>(
        ::cuda::std::move(inputs),
        ::cuda::std::move(output),
        static_cast<offset_t>(num_items),
        ::cuda::std::move(predicate),
        ::cuda::std::move(transform_op),
        ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env).get());
    }
  }

  // TODO(bgruber): we want to eventually forward the output tuple to the kernel and optimize writing multiple streams
  template <detail::transform::requires_stable_address StableAddress = detail::transform::requires_stable_address::no,
            typename... RandomAccessIteratorsIn,
            typename... RandomAccessIteratorsOut,
            typename NumItemsT,
            typename Predicate,
            typename TransformOp,
            typename Env>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformInternal(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    ::cuda::std::tuple<RandomAccessIteratorsOut...> outputs,
    NumItemsT num_items,
    Predicate predicate,
    TransformOp transform_op,
    Env env)
  {
    return TransformInternal<StableAddress>(
      ::cuda::std::move(inputs),
      ::cuda::make_zip_iterator(::cuda::std::move(outputs)),
      num_items,
      ::cuda::std::move(predicate),
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

public:
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Transforms many input sequences into many output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the tuple result to the corresponding output elements. No guarantee is given on the
  //! identity (i.e. address) of the objects passed to the call operator of the transformation operation.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-many-many
  //!     :end-before: example-end transform-many-many
  //!
  //! @endrst
  //!
  //! @param inputs A tuple of iterators to the input sequences where num_items elements are read from each. The
  //! iterators' value types must be trivially relocatable.
  //! @param outputs A tuple of iterators to the output sequences where num_items results are written to each. Each
  //! sequence may point to the beginning of one of the input sequences, performing the transformation inplace. Any
  //! output sequence must not overlap with any of the input sequence in any other way.
  //! @param num_items The number of elements in each input and output sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be a tuple where each tuple element is assignable to the corresponding dereferenced output
  //! iterators.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename... RandomAccessIteratorsIn,
            typename... RandomAccessIteratorsOut,
            typename NumItemsT,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    ::cuda::std::tuple<RandomAccessIteratorsOut...> outputs,
    NumItemsT num_items,
    TransformOp transform_op,
    Env env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Transform");
    return TransformInternal(
      ::cuda::std::move(inputs),
      ::cuda::std::move(outputs),
      num_items,
      detail::transform::always_true_predicate{},
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename... RandomAccessIteratorsIn,
            typename... RandomAccessIteratorsOut,
            typename NumItemsT,
            typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    ::cuda::std::tuple<RandomAccessIteratorsOut...> outputs,
    NumItemsT num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      ::cuda::std::move(inputs), ::cuda::std::move(outputs), num_items, ::cuda::std::move(transform_op), stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Transforms many input sequences into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. No guarantee is given on the identity
  //! (i.e. address) of the objects passed to the call operator of the transformation operation.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-many
  //!     :end-before: example-end transform-many
  //!
  //! @endrst
  //!
  //! @param inputs A tuple of iterators to the input sequences where num_items elements are read from each.
  //! @param output An iterator to the output sequence where num_items results are written to. May point to the
  //! beginning of one of the input sequences, performing the transformation inplace. The output sequence must not
  //! overlap with any of the input sequence in any other way.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename... RandomAccessIteratorsIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    Env env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Transform");
    return TransformInternal(
      ::cuda::std::move(inputs),
      ::cuda::std::move(output),
      num_items,
      detail::transform::always_true_predicate{},
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename NumItemsT, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      ::cuda::std::move(inputs), ::cuda::std::move(output), num_items, ::cuda::std::move(transform_op), stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Transforms one input sequence into one output sequence, by applying a transformation operation on each input
  //! element and writing the result to the corresponding output element. No guarantee is given on the identity (i.e.
  //! address) of the objects passed to the call operator of the transformation operation.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //! @endrst
  //!
  //! @param input An iterator to the input sequence where num_items elements are read from.
  //! @param output An iterator to the output sequence where num_items results are written to. May point to the same
  //! sequence as \p input, performing the transformation inplace. The output sequence must not overlap with the
  //! input sequence in any other way.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op A unary function object. The input iterator's value type must be convertible to the parameter
  //! of the function object's call operator. The return type of the call operator must be assignable to the
  //! dereferenced output iterator.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename RandomAccessIteratorIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    Env env = {})
  {
    return Transform(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename NumItemsT, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Fills the output sequence by invoking a generator operation for each output element and writing the result to it.
  //! This is effectively calling Transform with no input sequences.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //! @endrst
  //!
  //! @param output An iterator to the output sequence where num_items results are written to.
  //! @param num_items The number of elements to write to the output sequence.
  //! @param generator A nullary function object. The return type of the call operator must be assignable to the
  //! dereferenced output iterator.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Generator,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Generate(RandomAccessIteratorOut output, NumItemsT num_items, Generator generator, Env env = {})
  {
    static_assert(::cuda::std::is_invocable_v<Generator>, "The passed generator must be a nullary function object");
    static_assert(
      ::cuda::std::is_assignable_v<detail::it_reference_t<RandomAccessIteratorOut>,
                                   ::cuda::std::invoke_result_t<Generator>>,
      "The return value of the generator's call operator must be assignable to the dereferenced output iterator");

    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Generate");
    return TransformInternal(
      ::cuda::std::make_tuple(),
      ::cuda::std::move(output),
      num_items,
      detail::transform::always_true_predicate{},
      ::cuda::std::move(generator),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename RandomAccessIteratorOut, typename NumItemsT, typename Generator>
  CUB_RUNTIME_FUNCTION static cudaError_t Generate(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    Generator generator,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Generate(::cuda::std::move(output), num_items, ::cuda::std::move(generator), stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Fills the output sequence by writing the provided value to each element of the output sequence.
  //! This is effectively calling Generate with a functor returning that value.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //! @endrst
  //!
  //! @param output An iterator to the output sequence where num_items results are written to.
  //! @param num_items The number of elements to write to the output sequence.
  //! @param value The value to write. Must be assignable to the dereferenced output iterator.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Value,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Fill(RandomAccessIteratorOut output, NumItemsT num_items, Value value, Env env = {})
  {
    static_assert(::cuda::std::is_assignable_v<detail::it_reference_t<RandomAccessIteratorOut>, Value>,
                  "The passed value must be assignable to the dereferenced output iterator");

    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Fill");
    return TransformInternal(
      ::cuda::std::make_tuple(),
      ::cuda::std::move(output),
      num_items,
      detail::transform::always_true_predicate{},
      detail::__return_constant<Value>{::cuda::std::move(value)},
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename RandomAccessIteratorOut, typename NumItemsT, typename Value>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Fill(void* d_temp_storage,
       size_t& temp_storage_bytes,
       RandomAccessIteratorOut output,
       NumItemsT num_items,
       Value value,
       cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Fill(::cuda::std::move(output), num_items, ::cuda::std::move(value), stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Selectively transforms many input sequences into one output sequence, by applying a transformation operation on
  //! corresponding input elements, if a given predicate is true, and writing the result to the corresponding output
  //! element. No guarantee is given on the identity (i.e. address) of the objects passed to the call operator of the
  //! predicate and transformation operation. Output elements for which the predicate returns false are not written to.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-if
  //!     :end-before: example-end transform-if
  //!
  //! @endrst
  //!
  //! @param inputs A tuple of iterators to the input sequences where num_items elements are read from each.
  //! @param output An iterator to the output sequence where num_items results are written to. May point to the
  //! beginning of one of the input sequences, performing the transformation inplace. The output sequence must not
  //! overlap with any of the input sequence in any other way.
  //! @param num_items The number of elements in each input sequence.
  //! @param predicate An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator, which must return a boolean
  //! value.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator. Will only be invoked if \p predicate returns
  //! true.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename... RandomAccessIteratorsIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Predicate,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformIf(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    Predicate predicate,
    TransformOp transform_op,
    Env env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceTransform::TransformIf");
    return TransformInternal(
      ::cuda::std::move(inputs),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(predicate),
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename... RandomAccessIteratorsIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Predicate,
            typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    Predicate predicate,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformIf(
      ::cuda::std::move(inputs),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(predicate),
      ::cuda::std::move(transform_op),
      stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Selectively transforms one input sequence into one output sequence, by applying a transformation operation on each
  //! input element, if a given predicate is true, and writing the result to the corresponding output element. No
  //! guarantee is given on the identity (i.e. address) of the objects passed to the call operator of the predicate and
  //! transformation operation. Output elements for which the predicate returns false are not written to.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-if
  //!     :end-before: example-end transform-if
  //!
  //! @endrst
  //!
  //! @param input An iterator to the input sequence where num_items elements are read from.
  //! @param output An iterator to the output sequence where num_items results are written to. May point to the same
  //! sequence as \p input, performing the transformation inplace. The output sequence must not overlap with the
  //! input sequence in any other way.
  //! @param num_items The number of elements in each input sequence.
  //! @param predicate A unary function objects returning \p bool. The input iterators' value types must be convertible
  //! to the parameters of the function object's call operator.
  //! @param transform_op A unary function object. The input iterator's value type must be convertible to the
  //! parameter of the function object's call operator. The return type of the call operator must be assignable to the
  //! dereferenced output iterator. Will only be invoked if \p predicate returns true.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename RandomAccessIteratorIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Predicate,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformIf(
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    Predicate predicate,
    TransformOp transform_op,
    Env env = {})
  {
    return TransformIf(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(predicate),
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  // Overload with additional parameters to specify temporary storage. Provided for compatibility with other CUB APIs.
  template <typename RandomAccessIteratorIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename Predicate,
            typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    Predicate predicate,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformIf(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(predicate),
      ::cuda::std::move(transform_op),
      stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Transforms many input sequences into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. The objects passed to the call operator
  //! of the transformation operation are guaranteed to reside in the input sequences and are never copied.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-many-stable
  //!     :end-before: example-end transform-many-stable
  //!
  //! @endrst
  //!
  //! @param inputs A tuple of iterators to the input sequences where num_items elements are read from each.
  //! @param output An iterator to the output sequence where num_items results are written to. May point to the
  //! beginning of one of the input sequences, performing the transformation inplace. The output sequence must not
  //! overlap with any of the input sequence in any other way.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename... RandomAccessIteratorsIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    Env env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceTransform::TransformStableArgumentAddresses");
    return TransformInternal<detail::transform::requires_stable_address::yes>(
      ::cuda::std::move(inputs),
      ::cuda::std::move(output),
      num_items,
      detail::transform::always_true_predicate{},
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename NumItemsT, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformStableArgumentAddresses(
      ::cuda::std::move(inputs), ::cuda::std::move(output), num_items, ::cuda::std::move(transform_op), stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Transforms one input sequence into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. The objects passed to the call operator
  //! of the transformation operation are guaranteed to reside in the input sequences and are never copied.
  //!
  //! .. versionadded:: 2.8.0
  //!    First appears in CUDA Toolkit 12.9.
  //! @endrst
  //!
  //! @param input An iterator to the input sequence where num_items elements are read from.
  //! @param output An iterator to the output sequence where num_items results are written to. May point to the
  //! beginning of one of the input sequences, performing the transformation inplace. The output sequence must not
  //! overlap with any of the input sequence in any other way.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param env Execution environment, or cudaStream_t. Default is ``cuda::std::execution::env{}``, which will run on
  //! stream\ :sub:`0`
  template <typename RandomAccessIteratorIn,
            typename RandomAccessIteratorOut,
            typename NumItemsT,
            typename TransformOp,
            typename Env = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    Env env = {})
  {
    return TransformStableArgumentAddresses(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      ::cuda::std::move(env));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename NumItemsT, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    NumItemsT num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformStableArgumentAddresses(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      stream);
  }
#endif // _CCCL_DOXYGEN_INVOKED
};

CUB_NAMESPACE_END
