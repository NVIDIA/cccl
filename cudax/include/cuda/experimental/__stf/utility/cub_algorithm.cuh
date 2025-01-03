//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of CUB algorithm with STF facilities
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/cub.cuh>

namespace cuda::experimental::stf::reserved
{

/**
 * This functor transforms a 1D index into the result of the transformation
 */
// Args... is for example slice<double>, slice<int>...
template <typename TransformOp, typename shape_t, typename... Args>
struct IndexToTransformedValue
{
  IndexToTransformedValue(TransformOp _op, shape_t s, ::std::tuple<Args...> _targs)
      : op(mv(_op))
      , shape(mv(s))
      , targs(mv(_targs))
  {}

  __host__ __device__ __forceinline__ auto operator()(const size_t& index) const
  {
    const auto explode_args = [&](auto&&... data) {
      CUDASTF_NO_DEVICE_STACK
      auto const explode_coords = [&](auto&&... coords) {
        return op(coords..., data...);
      };
      return ::std::apply(explode_coords, shape.index_to_coords(index));
    };
    return ::std::apply(explode_args, targs);
  }

  TransformOp op;
  const shape_t shape;
  ::std::tuple<Args...> targs;
};

/**
 * @brief Helper to transform an extended device lambda into a functor that we can use in CUB
 */
template <typename BinaryOp>
struct LambdaOpWrapper
{
  LambdaOpWrapper(BinaryOp _op)
      : op(mv(_op))
  {}

  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return op(a, b);
  }

  BinaryOp op;
};

#if 0
// A helper trait to detect if the expression in the operator->* body is valid
template <typename F1, typename F2, typename = void>
struct is_arrow_star_valid : ::std::false_type
{};

// Specialization: If the expression compiles, this specialization is chosen
template <typename F1, typename F2>
struct is_arrow_star_valid<F1, F2, ::std::void_t<decltype(std::declval<F1>()(std::declval<F2>()))>> : ::std::true_type
{};

// Helper variable template for convenience
template <typename F1, typename F2>
constexpr bool is_arrow_star_valid_v = is_arrow_star_valid<F1, F2>::value;

// Define operator->* only if the body of the operator would compile
template <class F1, class F2, ::std::enable_if_t<is_arrow_star_valid_v<F1, F2>, int> = 0>
decltype(auto) operator->*(F1&& f1, F2&& f2)
{
  static_assert(!::std::is_lvalue_reference_v<F1>, "Left-hand side of operator->* must be an rvalue.");
  return ::std::forward<F1>(f1)(::std::forward<F2>(f2));
}

template <typename Ctx, typename shape_t, typename OutT, typename... Args>
auto stf_transform_reduce(Ctx& ctx, shape_t s, OutT init_val, const logical_data<Args>&... args)
{
  auto args_tuple = ::std::forward_as_tuple(args...);

  // We are going to enumerate the entries in the shape, and create an counting
  // iterator that will be combined with a functor that converts the
  // corresponding 1D index in the shape to the result of the transform
  // operation.
  // The lambda we return here takes the transform lambda and applies it.
  return [&ctx, args_tuple = mv(args_tuple), s = mv(s), init_val = mv(init_val)](auto&& transform_op) mutable {
    // This lambda takes the reduction operator and finalizes the operation.
    // We can capture most parameters by reference here because the parent object will
    // be alive during the call.
    return
      [&ctx,
       args_tuple = mv(args_tuple),
       //         transform_op = mv(transform_op),
       &transform_op,
       // due to mutable ?
       //         s        = mv(s),
       //         init_val = mv(init_val)
       &s,
       &init_val](auto&& reduce_op) {
        // This will be the ultimate result of the transform followed by reduce
        auto result = ctx.logical_data(shape_of<scalar_view<OutT>>());

        auto t = ::std::apply(
          [&](auto&&... a) {
            return ctx.task(result.write(), a.read()...);
          },
          args_tuple);

        t->*[&](cudaStream_t stream, scalar_view<OutT> res, auto... deps) {
          using TransformOp = ::std::remove_reference_t<decltype(transform_op)>;

          auto deps_tuple = ::std::forward_as_tuple(deps...);

          // This is a functor that transforms a 1D index in the shape into the result
          // of the transform operation
          //
          // Note that we pass a tuple where we have removed the first argument (we do
          // not consider the logical data of the result during the transform)
          using ConversionOp_t = IndexToTransformedValue<TransformOp, shape_t, Args...>;

          auto count_it = cub::CountingInputIterator<size_t>(0);

          // This is an iterator that combines the counting iterator with the functor
          // to apply the transformation, it thus outputs the different values produced
          // by the transformation over the shape
          cub::TransformInputIterator<OutT, ConversionOp_t, decltype(count_it)> transform_output_it(
            count_it, ConversionOp_t(transform_op, s, deps_tuple));

          using BinaryOp = ::std::remove_reference_t<decltype(reduce_op)>;
          // Determine temporary device storage requirements
          void* d_temp_storage      = nullptr;
          size_t temp_storage_bytes = 0;
          auto wrapped_reducer      = LambdaOpWrapper<BinaryOp>(reduce_op);

          cub::DeviceReduce::Reduce(
            d_temp_storage,
            temp_storage_bytes,
            transform_output_it,
            (OutT*) res.addr, // output into a scalar_view<OutT>
            s.size(),
            wrapped_reducer,
            init_val,
            stream);

          cuda_safe_call(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

          cub::DeviceReduce::Reduce(
            d_temp_storage,
            temp_storage_bytes,
            transform_output_it,
            (OutT*) res.addr, // output into a scalar_view<OutT>
            s.size(),
            mv(wrapped_reducer),
            init_val,
            stream);

          cuda_safe_call(cudaFreeAsync(d_temp_storage, stream));
        };

        return mv(result);
      };
  };
}
#endif

template <typename Ctx, typename shape_t, typename OutT, typename... Args>
class stf_transform_reduce_scope
{
public:
  stf_transform_reduce_scope(Ctx ctx, shape_t s, OutT init_val, logical_data<Args>... args)
      : ctx(mv(ctx))
      , s(mv(s))
      , init_val(mv(init_val))
      , args_tuple(mv(args)...)
  {}

  stf_transform_reduce_scope(const stf_transform_reduce_scope&)            = delete;
  stf_transform_reduce_scope(stf_transform_reduce_scope&&)                 = default;
  stf_transform_reduce_scope& operator=(const stf_transform_reduce_scope&) = delete;

  template <typename TransformOp>
  class reduce_scope
  {
  public:
    reduce_scope(stf_transform_reduce_scope& parent_scope, TransformOp&& transform_op)
        : parent(&parent_scope)
        , transform_op(mv(transform_op))
    {}

    reduce_scope(const reduce_scope&)            = delete;
    reduce_scope(reduce_scope&&)                 = default;
    reduce_scope& operator=(const reduce_scope&) = delete;

    template <typename ReduceOp>
    auto operator->*(ReduceOp&& reduce_op)
    {
      // This will be the ultimate result of the transform followed by reduce
      auto result = parent->ctx.logical_data(shape_of<scalar_view<OutT>>());

      auto t = ::std::apply(
        [&](auto&&... a) {
          return parent->ctx.task(result.write(), a.read()...);
        },
        parent->args_tuple);

      t->*[&](cudaStream_t stream, scalar_view<OutT> res, auto... deps) {
        // using TransformOp = ::std::remove_reference_t<decltype(transform_op)>;

        auto deps_tuple = ::std::forward_as_tuple(deps...);

        // This is a functor that transforms a 1D index in the shape into the result
        // of the transform operation
        //
        // Note that we pass a tuple where we have removed the first argument (we do
        // not consider the logical data of the result during the transform)
        using ConversionOp_t = IndexToTransformedValue<TransformOp, shape_t, Args...>;

        auto count_it = cub::CountingInputIterator<size_t>(0);

        // This is an iterator that combines the counting iterator with the functor
        // to apply the transformation, it thus outputs the different values produced
        // by the transformation over the shape
        cub::TransformInputIterator<OutT, ConversionOp_t, decltype(count_it)> transform_output_it(
          count_it, ConversionOp_t(transform_op, parent->s, deps_tuple));

        using BinaryOp = ::std::remove_reference_t<decltype(reduce_op)>;
        // Determine temporary device storage requirements
        void* d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        auto wrapped_reducer      = LambdaOpWrapper<BinaryOp>(reduce_op);

        cub::DeviceReduce::Reduce(
          d_temp_storage,
          temp_storage_bytes,
          transform_output_it,
          (OutT*) res.addr, // output into a scalar_view<OutT>
          parent->s.size(),
          wrapped_reducer,
          parent->init_val,
          stream);

        cuda_safe_call(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

        cub::DeviceReduce::Reduce(
          d_temp_storage,
          temp_storage_bytes,
          transform_output_it,
          (OutT*) res.addr, // output into a scalar_view<OutT>
          parent->s.size(),
          mv(wrapped_reducer),
          parent->init_val,
          stream);

        cuda_safe_call(cudaFreeAsync(d_temp_storage, stream));
      };

      return result;
    }

  private:
    stf_transform_reduce_scope* parent = nullptr;
    TransformOp transform_op;
  };

  template <typename TransformOp>
  auto operator->*(TransformOp&& transform_op)
  {
    return reduce_scope<::std::remove_reference_t<TransformOp>>(*this, mv(transform_op));
  }

private:
  Ctx ctx;
  shape_t s;
  OutT init_val;
  ::std::tuple<logical_data<::std::decay_t<Args>>...> args_tuple;
};

template <typename Ctx, typename shape_t, typename OutT, typename... Args>
class stf_transform_exclusive_scan_scope
{
public:
  stf_transform_exclusive_scan_scope(Ctx& ctx, shape_t s, OutT init_val, const logical_data<Args>&... args)
      : ctx(mv(ctx))
      , s(mv(s))
      , init_val(mv(init_val))
      , args_tuple(args...)
  {}

  stf_transform_exclusive_scan_scope(const stf_transform_exclusive_scan_scope&)            = delete;
  stf_transform_exclusive_scan_scope(stf_transform_exclusive_scan_scope&&)                 = default;
  stf_transform_exclusive_scan_scope& operator=(const stf_transform_exclusive_scan_scope&) = delete;

  template <typename TransformOp>
  class reduce_scope
  {
  public:
    reduce_scope(stf_transform_exclusive_scan_scope& parent_scope, TransformOp&& transform_op)
        : parent(&parent_scope)
        , transform_op(mv(transform_op))
    {}

    reduce_scope(const reduce_scope&)            = delete;
    reduce_scope(reduce_scope&&)                 = default;
    reduce_scope& operator=(const reduce_scope&) = delete;

    template <typename BinaryOp>
    auto operator->*(BinaryOp&& binary_op)
    {
      // This will be the ultimate result of the transform followed by scan
      auto result = parent->ctx.logical_data(shape_of<slice<OutT>>(parent->s.size()));

      auto t = ::std::apply(
        [&](auto&&... a) {
          return parent->ctx.task(result.write(), a.read()...);
        },
        parent->args_tuple);

      t->*
        [transform_op = mv(transform_op),
         binary_op    = mv(binary_op),
         s            = mv(parent->s),
         init_val     = mv(parent->init_val)](cudaStream_t stream, slice<OutT> res, auto... deps) {
          auto deps_tuple = ::std::forward_as_tuple(deps...);

          // This is a functor that transforms a 1D index in the shape into the result
          // of the transform operation
          //
          // Note that we pass a tuple where we have removed the first argument (we do
          // not consider the logical data of the result during the transform)
          using ConversionOp_t = IndexToTransformedValue<TransformOp, shape_t, Args...>;

          auto count_it = cub::CountingInputIterator<size_t>(0);

          //          using BinaryOp = ::std::remove_reference_t<decltype(binary_op)>;

          // This is an iterator that combines the counting iterator with the functor
          // to apply the transformation, it thus outputs the different values produced
          // by the transformation over the shape
          cub::TransformInputIterator<OutT, ConversionOp_t, decltype(count_it)> transform_output_it(
            count_it, ConversionOp_t(transform_op, s, deps_tuple));

          // Determine temporary device storage requirements
          void* d_temp_storage      = nullptr;
          size_t temp_storage_bytes = 0;
          auto wrapped_binary_op    = LambdaOpWrapper<BinaryOp>(binary_op);

          cub::DeviceScan::ExclusiveScan(
            d_temp_storage,
            temp_storage_bytes,
            transform_output_it,
            (OutT*) res.data_handle(),
            wrapped_binary_op,
            init_val,
            s.size(),
            stream);

          cuda_safe_call(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

          cub::DeviceScan::ExclusiveScan(
            d_temp_storage,
            temp_storage_bytes,
            transform_output_it,
            (OutT*) res.data_handle(),
            wrapped_binary_op,
            init_val,
            s.size(),
            stream);

          cuda_safe_call(cudaFreeAsync(d_temp_storage, stream));
        };

      return mv(result);
    }

  private:
    stf_transform_exclusive_scan_scope* parent = nullptr;
    TransformOp transform_op;
  };

  template <typename TransformOp>
  auto operator->*(TransformOp&& transform_op)
  {
    return reduce_scope<::std::remove_reference_t<TransformOp>>(*this, mv(transform_op));
  }

private:
  Ctx ctx;
  shape_t s;
  OutT init_val;
  ::std::tuple<logical_data<::std::decay_t<Args>>...> args_tuple;
};

} // end namespace cuda::experimental::stf::reserved
