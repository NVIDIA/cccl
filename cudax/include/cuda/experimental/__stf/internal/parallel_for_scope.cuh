//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/execution_space.h>

#include <cuda/experimental/__stf/internal/backend_ctx.cuh> // for null_partition
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/task_statistics.cuh>

namespace cuda::experimental::stf
{

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

class stream_ctx;
class graph_ctx;

namespace reserved
{

/*
 * @brief A CUDA kernel for executing a function `f` in parallel over `n` threads.
 *
 * This kernel takes a function `f` and its parameters `p`, then executes `f(i, p...)`
 * for each thread index `i` from 0 through `n-1`.
 *
 * @tparam F The type of the function to execute.
 * @tparam P The types of additional parameters to pass to the function.
 *
 * @param n The number of times to execute the function in parallel.
 * @param f The function to execute.
 * @param p The additional parameters to pass to the function `f`.
 */
template <typename F, typename shape_t, typename tuple_args>
__global__ void loop(const _CCCL_GRID_CONSTANT size_t n, shape_t shape, F f, tuple_args targs)
{
  size_t i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;

  // This will explode the targs tuple into a pack of data

  // Help the compiler which may not detect that a device lambda is calling a device lambda
  CUDASTF_NO_DEVICE_STACK
  auto explode_args = [&](auto... data) {
    // For every linearized index in the shape
    for (; i < n; i += step)
    {
      CUDASTF_NO_DEVICE_STACK
      auto explode_coords = [&](auto... coords) {
        f(coords..., data...);
      };
      ::std::apply(explode_coords, shape.index_to_coords(i));
    }
  };
  ::std::apply(explode_args, mv(targs));
}

/**
 * @brief Supporting class for the parallel_for construct
 *
 * This is used to implement operators such as ->* on the object produced by `ctx.parallel_for`
 *
 * @tparam deps_t
 */
template <typename context, typename shape_t, typename partitioner_t, typename... deps_t>
class parallel_for_scope
{
public:
  /// @brief Constructor
  /// @param ctx Reference to context (it will not be copied, so careful with lifetimes)
  /// @param e_place Execution place for this parallel_for
  /// @param shape Shape to iterate
  /// @param ...deps Dependencies
  parallel_for_scope(context& ctx, exec_place e_place, shape_t shape, task_dep<deps_t>... deps)
      : deps{deps...}
      , ctx(ctx)
      , e_place(mv(e_place))
      , shape(mv(shape))
  {
    dump_hooks = reserved::get_dump_hooks(&ctx, deps...);
  }

  parallel_for_scope(const parallel_for_scope&)            = delete;
  parallel_for_scope(parallel_for_scope&&)                 = default;
  parallel_for_scope& operator=(const parallel_for_scope&) = delete;

  /**
   * @brief Retrieves the task dependencies in an untyped vector.
   *
   * @return The task dependencies as a `task_dep_vector_untyped` object.
   */
  const task_dep_vector_untyped& get_task_deps() const
  {
    return deps;
  }

  /**
   * @brief Retrieves the symbol associated with the task.
   *
   * @return A constant reference to the symbol string.
   */
  const ::std::string& get_symbol() const
  {
    return symbol;
  }

  /**
   * @brief Sets the symbol associated with the task.
   *
   * This method uses a custom move function `mv` to handle the transfer of ownership.
   *
   * @param s The new symbol string.
   * @return A reference to the current object, allowing for method chaining.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  /**
   * @brief Overloads the `operator->*` to perform parallel computations using a user-defined function or lambda.
   *
   * This method initializes various runtime entities (task, CUDA events, etc.) and applies the parallel_for construct
   * over the task's shape. It can work with different execution places and also supports partitioning the shape based
   * on a given partitioner.
   *
   * @tparam Fun The type of the user-defined function or lambda.
   *
   * @param f The user-defined function or lambda that is to be applied in parallel.
   *
   * @note
   * The method will perform several operations such as:
   *   - Initialize and populate the task with the required dependencies and symbol.
   *   - Create CUDA events and record them if needed.
   *   - Determine whether the operation is to be performed on device or host and proceed accordingly.
   *   - If there is a partitioner, the method can also break the task's shape into sub-shapes and perform
   * computations in parts.
   *   - Log task time if statistics gathering is enabled.
   *
   * @pre `e_place.affine_data_place()` should match `t.get_affine_data_place()` where `t` is the task generated from
   * `ctx.task(e_place)`.
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();
    auto t           = ctx.task(e_place);

    assert(e_place.affine_data_place() == t.get_affine_data_place());

    // If there is a partitioner, we ensure there is a proper affine data place for this execution place
    if constexpr (!::std::is_same_v<partitioner_t, null_partition>)
    {
      // This is only meaningful for grid of places
      if (e_place.is_grid())
      {
        // Create a composite data place defined by the grid of places + the partitioning function
        t.set_affine_data_place(data_place::composite(partitioner_t(), e_place.as_grid()));
      }
    }

    t.add_post_submission_hook(dump_hooks);

    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    cudaEvent_t start_event, end_event;

    const bool record_time = t.schedule_task() || statistics.is_calibrating_to_file();

    nvtx_range nr(t.get_symbol().c_str());
    t.start();

    int device = -1;

    SCOPE(exit)
    {
      t.end_uncleared();
      if constexpr (::std::is_same_v<context, stream_ctx>)
      {
        if (record_time)
        {
          cuda_safe_call(cudaEventRecord(end_event, t.get_stream()));
          cuda_safe_call(cudaEventSynchronize(end_event));

          float milliseconds = 0;
          cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

          if (dot.is_tracing())
          {
            dot.template add_vertex_timing<typename context::task_type>(t, milliseconds, device);
          }

          if (statistics.is_calibrating())
          {
            statistics.log_task_time(t, milliseconds);
          }
        }
      }

      t.clear();
    };

    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaGetDevice(&device)); // We will use this to force it during the next run
        // Events must be created here to avoid issues with multi-gpu
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    if (dot.is_tracing())
    {
      dot.template add_vertex<typename context::task_type, logical_data_untyped>(t);
    }

#  if __NVCOMPILER
    // With nvc++, all lambdas can run on host and device.
    static constexpr bool is_extended_host_device_lambda_closure_type = true,
                          is_extended_device_lambda_closure_type      = false;
#  else
    // With nvcpp, dedicated traits tell how a lambda can be executed.
    static constexpr bool is_extended_host_device_lambda_closure_type =
                            __nv_is_extended_host_device_lambda_closure_type(Fun),
                          is_extended_device_lambda_closure_type = __nv_is_extended_device_lambda_closure_type(Fun);
#  endif

    if constexpr (is_extended_host_device_lambda_closure_type)
    {
      // Can run on both - decide dynamically
      if (e_place == exec_place::host)
      {
        return do_parallel_for_host(::std::forward<Fun>(f), shape, t);
      }
      // Fall through for the device implementation
    }
    else if constexpr (is_extended_device_lambda_closure_type)
    {
      // Lambda can run only on device - make sure they're not trying it on the host
      EXPECT(e_place != exec_place::host, "Attempt to run a device function on the host.");
      // Fall through for the device implementation
    }
    else
    {
      // Lambda can run only on the host - make sure they're not trying it elsewhere
      EXPECT(e_place == exec_place::host, "Attempt to run a host function on a device.");
      return do_parallel_for_host(::std::forward<Fun>(f), shape, t);
    }

    // Device land. Must use the supplemental if constexpr below to avoid compilation errors.
    if constexpr (is_extended_host_device_lambda_closure_type || is_extended_device_lambda_closure_type)
    {
      if (!e_place.is_grid())
      {
        // Apply the parallel_for construct over the entire shape on the
        // execution place of the task
        do_parallel_for(f, e_place, shape, t);
      }
      else
      {
        if constexpr (::std::is_same_v<partitioner_t, null_partition>)
        {
          fprintf(stderr, "Fatal: Grid execution requires a partitioner.\n");
          abort();
        }
        else
        {
          size_t grid_size = t.grid_dims().size();
          for (size_t i = 0; i < grid_size; i++)
          {
            t.set_current_place(pos4(i));
            const auto sub_shape = partitioner_t::apply(shape, pos4(i), t.grid_dims());
            do_parallel_for(f, t.get_current_place(), sub_shape, t);
            t.unset_current_place();
          }
        }
      }
    }
    else
    {
      // This point is never reachable, but we can't prove that statically.
      assert(!"Internal CUDASTF error.");
    }
  }

  // Executes the loop on a device, or use the host implementation
  template <typename Fun, typename sub_shape_t>
  void do_parallel_for(
    Fun&& f, const exec_place& sub_exec_place, const sub_shape_t& sub_shape, typename context::task_type& t)
  {
    // parallel_for never calls this function with a host.
    _CCCL_ASSERT(sub_exec_place != exec_place::host, "Internal CUDASTF error.");

    if (sub_exec_place == exec_place::device_auto)
    {
      // We have all latitude - recurse with the current device.
      return do_parallel_for(::std::forward<Fun>(f), exec_place::current_device(), sub_shape, t);
    }

    using Fun_no_ref = ::std::remove_reference_t<Fun>;

    static const auto conf = [] {
      // We are using int instead of size_t because CUDA API uses int for occupancy calculations
      int min_grid_size = 0, max_block_size = 0, block_size_limit = 0;
      // compute_kernel_limits will return the min number of blocks/max
      // block size to optimize occupancy, as well as some block size
      // limit. We choose to dimension the kernel of the parallel loop to
      // optimize occupancy.
      reserved::compute_kernel_limits(
        &reserved::loop<Fun_no_ref, sub_shape_t, ::std::tuple<deps_t...>>,
        min_grid_size,
        max_block_size,
        0,
        false,
        block_size_limit);
      return ::std::pair(size_t(min_grid_size), size_t(max_block_size));
    }();

    const auto [block_size, min_blocks] = conf;
    size_t n                            = sub_shape.size();

    // If there is no item in that shape, no need to launch a kernel !
    if (n == 0)
    {
      // fprintf(stderr, "Empty shape, no kernel ...\n");
      return;
    }

    // max_blocks is computed so we have one thread per element processed
    const auto max_blocks = (n + block_size - 1) / block_size;

    // TODO: improve this
    size_t blocks = ::std::min(min_blocks * 3 / 2, max_blocks);

    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      reserved::loop<Fun_no_ref><<<static_cast<int>(blocks), static_cast<int>(block_size), 0, t.get_stream()>>>(
        static_cast<int>(n), sub_shape, mv(f), deps.instance(t));
    }
    else if constexpr (::std::is_same_v<context, graph_ctx>)
    {
      // Put this kernel node in the child graph that implements the graph_task<>
      cudaKernelNodeParams kernel_params;

      kernel_params.func = (void*) reserved::loop<Fun_no_ref, sub_shape_t, ::std::tuple<deps_t...>>;

      kernel_params.gridDim  = dim3(static_cast<int>(blocks));
      kernel_params.blockDim = dim3(static_cast<int>(block_size));

      auto arg_instances = deps.instance(t);
      // It is ok to use reference to local variables because the arguments
      // will be used directly when calling cudaGraphAddKernelNode
      void* kernelArgs[]         = {&n, const_cast<void*>(static_cast<const void*>(&sub_shape)), &f, &arg_instances};
      kernel_params.kernelParams = kernelArgs;
      kernel_params.extra        = nullptr;

      kernel_params.sharedMemBytes = 0;

      // This task corresponds to a single graph node, so we set that
      // node instead of creating an child graph. Input and output
      // dependencies will be filled later.
      cuda_safe_call(cudaGraphAddKernelNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &kernel_params));
      // fprintf(stderr, "KERNEL NODE => graph %p, gridDim %d blockDim %d (n %ld)\n", t.get_graph(),
      // kernel_params.gridDim.x, kernel_params.blockDim.x, n);
    }
    else
    {
      fprintf(stderr, "Internal error.\n");
      abort();
    }
  }

  // Executes loop on the host.
  template <typename Fun, typename sub_shape_t>
  void do_parallel_for_host(Fun&& f, const sub_shape_t& shape, typename context::task_type& t)
  {
    const size_t n = shape.size();

    // Wrap this for_each_n call in a host callback launched in CUDA stream associated with that task
    // To do so, we pack all argument in a dynamically allocated tuple
    // that will be deleted by the callback
    auto args =
      new ::std::tuple<decltype(deps.instance(t)), size_t, Fun&&, sub_shape_t>(deps.instance(t), n, mv(f), shape);

    // The function which the host callback will execute
    auto host_func = [](void* untyped_args) {
      auto p = static_cast<decltype(args)>(untyped_args);
      SCOPE(exit)
      {
        delete p;
      };

      auto& data               = ::std::get<0>(*p);
      const size_t n           = ::std::get<1>(*p);
      Fun&& f                  = mv(::std::get<2>(*p));
      const sub_shape_t& shape = ::std::get<3>(*p);

      auto explode_coords = [&](size_t i, deps_t... data) {
        auto h = [&](auto... coords) {
          f(coords..., data...);
        };
        ::std::apply(h, shape.index_to_coords(i));
      };

      // Finally we get to do the workload on every 1D item of the shape
      for (size_t i = 0; i < n; ++i)
      {
        ::std::apply(explode_coords, ::std::tuple_cat(::std::make_tuple(i), data));
      }
    };

    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      cuda_safe_call(cudaLaunchHostFunc(t.get_stream(), host_func, args));
    }
    else if constexpr (::std::is_same_v<context, graph_ctx>)
    {
      cudaHostNodeParams params;
      params.userData = args;
      params.fn       = host_func;

      // Put this host node into the child graph that implements the graph_task<>
      cuda_safe_call(cudaGraphAddHostNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &params));
    }
    else
    {
      fprintf(stderr, "Internal error.\n");
      abort();
    }
  }

private:
  task_dep_vector<deps_t...> deps;
  context& ctx;
  exec_place e_place;
  ::std::string symbol;
  shape_t shape;

  ::std::vector<::std::function<void()>> dump_hooks;
};
} // end namespace reserved

#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

} // end namespace cuda::experimental::stf
