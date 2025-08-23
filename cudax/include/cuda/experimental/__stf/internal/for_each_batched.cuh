//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief for_each_batched experimental mechanism
 */

#pragma once

#include <cuda/experimental/__stf/internal/algorithm.cuh>
#include <cuda/experimental/__stf/internal/context.cuh>

namespace cuda::experimental::stf
{

template <typename... Deps>
class for_each_batched
{
public:
  for_each_batched(
    context ctx, size_t cnt, size_t batch_size, ::std::function<::std::tuple<task_dep<Deps>...>(size_t)> df)
      : cnt(cnt)
      , batch_size(batch_size)
      , df(mv(df))
      , ctx(mv(ctx))
  {}

  // Create a batch operation that computes fun(start), fun(start+1), ... f(end-1)
  template <typename Fun>
  void batched_iterations(Fun&& fun, size_t start, size_t end)
  {
    // Create "untyped" dependencies
    task_dep_vector_untyped deps;
    for (size_t i = start; i < end; i++)
    {
      ::std::apply(
        [&deps](auto&&... args) {
          // Call the method on each tuple element
          (deps.push_back(::std::forward<decltype(args)>(args)), ...);
        },
        df(i));
    }

    // templated by Fun
    static algorithm batch_alg;

    auto fn = [this, start, end, &fun](context gctx, stream_task<> t) {
      // How many logical data per iteration ?
      [[maybe_unused]] constexpr size_t data_per_iteration = ::std::tuple_size<decltype(df(0))>::value;

      auto logify = [](auto& dest_ctx, auto x) {
        return dest_ctx.logical_data(to_rw_type_of(x), exec_place::current_device().affine_data_place());
      };

      for (size_t i = start; i < end; i++)
      {
        // Compute a tuple of all instances (e.g. tuple<slice<double>, slice<double>>)

        // Transform the tuple by applying a lambda to each element
        auto instance_tuple =
          tuple_transform(df(i), [&t, i, start, data_per_iteration](auto&& item, std::size_t arg_ind) {
            // Get the arg_ind-th element of the i-th batch.
            // Its type is the same as the arg_ind-th entry of
            // df(i)
            //
            // For example : if df(i) is tuple(lX.read(),
            // lY.rw()), the second entry of the batch has the
            // same type as the lY interface
            using arg_type = typename ::std::decay_t<decltype(item)>::data_t;
            return t.template get<arg_type>((i - start) * data_per_iteration + arg_ind);
          });

        // Logify all these instances (create temporary aliases)
        // Returns eg. a tuple<logical_data<slice<double>>, logical_data<slice<double>>>
        auto logified_instances_tuple = ::std::apply(
          [&logify, &gctx](auto&&... args) {
            return ::std::make_tuple(logify(gctx, args)...);
          },
          instance_tuple);

        ::std::apply(fun, ::std::tuple_cat(::std::make_tuple(context(gctx), i), logified_instances_tuple));
      }
    };

    // Launch the fn method as a task which takes an untyped vector of dependencies
    batch_alg.run_as_task_dynamic(fn, ctx, deps);
  }

  template <typename Fun>
  void operator->*(Fun&& fun)
  {
    // Process in batches
    for (size_t start = 0; start < cnt; start += batch_size)
    {
      size_t end = ::std::min(start + batch_size, cnt);
      batched_iterations(fun, start, end);
    }
  }

private:
  size_t cnt;
  size_t batch_size;
  ::std::function<::std::tuple<task_dep<Deps>...>(size_t)> df;
  context ctx;
};

} // namespace cuda::experimental::stf
