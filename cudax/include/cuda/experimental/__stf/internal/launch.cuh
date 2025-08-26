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

#include <cuda/experimental/__stf/internal/execution_policy.cuh> // launch_impl() uses execution_policy
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/task_statistics.cuh>
#include <cuda/experimental/__stf/internal/thread_hierarchy.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh> // graph_launch_impl() uses SCOPE

namespace cuda::experimental::stf
{

// This feature requires a CUDA compiler
#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

class stream_ctx;
template <typename...>
class stream_task;

namespace reserved
{

template <typename Fun, typename Arg>
__global__ void launch_kernel(Fun f, Arg arg)
{
  ::std::apply(mv(f), mv(arg));
}

template <typename interpreted_spec, typename Fun, typename Stream_t>
void cuda_launcher(interpreted_spec interpreted_policy, Fun&& f, void** args, Stream_t& stream)
{
  const ::std::array<size_t, 3> config     = interpreted_policy.get_config();
  const ::std::array<size_t, 3> mem_config = interpreted_policy.get_mem_config();

  bool cooperative_kernel = interpreted_policy.need_cooperative_kernel_launch();

  cudaLaunchAttribute attrs[1];
  attrs[0].id              = cudaLaunchAttributeCooperative;
  attrs[0].val.cooperative = cooperative_kernel ? 1 : 0;

  cudaLaunchConfig_t lconfig;
  lconfig.gridDim          = static_cast<int>(config[1]);
  lconfig.blockDim         = static_cast<int>(config[2]);
  lconfig.attrs            = attrs;
  lconfig.numAttrs         = 1;
  lconfig.dynamicSmemBytes = mem_config[2];
  lconfig.stream           = stream;

  cuda_safe_call(cudaLaunchKernelExC(&lconfig, (void*) f, args));
}

template <typename interpreted_spec, typename Fun>
void cuda_launcher_graph(interpreted_spec interpreted_policy, Fun&& f, void** args, cudaGraph_t& g, cudaGraphNode_t& n)
{
  const ::std::array<size_t, 3> config     = interpreted_policy.get_config();
  const ::std::array<size_t, 3> mem_config = interpreted_policy.get_mem_config();

  cudaKernelNodeParams kconfig;
  kconfig.gridDim        = static_cast<int>(config[1]);
  kconfig.blockDim       = static_cast<int>(config[2]);
  kconfig.extra          = nullptr;
  kconfig.func           = (void*) f;
  kconfig.kernelParams   = args;
  kconfig.sharedMemBytes = static_cast<int>(mem_config[2]);

  cuda_safe_call(cudaGraphAddKernelNode(&n, g, nullptr, 0, &kconfig));

  // Enable cooperative kernel if necessary by updating the node attributes

  bool cooperative_kernel = interpreted_policy.need_cooperative_kernel_launch();

  cudaKernelNodeAttrValue val;
  val.cooperative = cooperative_kernel ? 1 : 0;
  cuda_safe_call(cudaGraphKernelNodeSetAttribute(n, cudaKernelNodeAttributeCooperative, &val));
}

template <typename Fun, typename interpreted_spec, typename Arg>
void launch_impl(interpreted_spec interpreted_policy, exec_place& p, Fun f, Arg arg, cudaStream_t stream, size_t rank)
{
  assert(!p.is_grid());

  p->*[&] {
    auto th = thread_hierarchy(static_cast<int>(rank), interpreted_policy);

    void* th_dev_tmp_ptr = nullptr;

    /* Allocate temporary device memory */
    auto th_mem_config = interpreted_policy.get_mem_config();
    if (th_mem_config[0] > 0)
    {
      // Lazily initialize system memory if needed
      void* sys_mem = interpreted_policy.get_system_mem();
      if (!sys_mem)
      {
        sys_mem = allocateManagedMemory(th_mem_config[0]);
        interpreted_policy.set_system_mem(sys_mem);
      }

      assert(sys_mem);
      th.set_system_tmp(sys_mem);
    }

    if (th_mem_config[1] > 0)
    {
      cuda_safe_call(cudaMallocAsync(&th_dev_tmp_ptr, th_mem_config[1], stream));
      th.set_device_tmp(th_dev_tmp_ptr);
    }

    auto kernel_args = tuple_prepend(mv(th), mv(arg));
    using args_type  = decltype(kernel_args);
    void* all_args[] = {&f, &kernel_args};

    cuda_launcher(interpreted_policy, reserved::launch_kernel<Fun, args_type>, all_args, stream);

    if (th_mem_config[1] > 0)
    {
      cuda_safe_call(cudaFreeAsync(th_dev_tmp_ptr, stream));
    }
  };
}

template <typename task_t, typename Fun, typename interpreted_spec, typename Arg>
void graph_launch_impl(task_t& t, interpreted_spec interpreted_policy, exec_place& p, Fun f, Arg arg, size_t rank)
{
  assert(!p.is_grid());

  auto kernel_args = tuple_prepend(thread_hierarchy(static_cast<int>(rank), interpreted_policy), mv(arg));
  using args_type  = decltype(kernel_args);
  void* all_args[] = {&f, &kernel_args};

  p->*[&] {
    cuda_launcher_graph(
      interpreted_policy, reserved::launch_kernel<Fun, args_type>, all_args, t.get_ctx_graph(), t.get_node());
  };
}

/**
 * @brief a free-function implementation of the launch mechanism which can be
 * used multiple times in a task, or possibly outside tasks
 */
template <typename spec_t, typename Arg>
class launch
{
public:
  /**
   * @brief Constructor to initialize a light-weight launch
   * @param ctx Execution context.
   * @param spec The specification of the thread hierarchy.
   * @param e_place Execution place (e.g., device, host).
   * @param arg Arguments to be passed to the kernel.
   */
  launch(spec_t spec, exec_place e_place, ::std::vector<cudaStream_t> streams, Arg arg)
      : arg(mv(arg))
      , e_place(mv(e_place))
      , spec(mv(spec))
      , streams(mv(streams))
  {}

  launch(exec_place e_place, ::std::vector<cudaStream_t> streams, Arg arg)
      : launch(spec_t(), mv(e_place), mv(arg), mv(streams))
  {}

  template <typename Fun>
  void operator->*(Fun&& f)
  {
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

    static_assert(is_extended_host_device_lambda_closure_type || is_extended_device_lambda_closure_type,
                  "Cannot run launch() on the host");

    EXPECT(e_place != exec_place::host(), "Attempt to run a launch on the host.");

    const size_t grid_size = e_place.size();

    using th_t     = typename spec_t::thread_hierarchy_t;
    using arg_type = decltype(tuple_prepend(th_t(), arg));

    auto interpreted_policy = interpreted_execution_policy(spec, e_place, reserved::launch_kernel<Fun, arg_type>);

    SCOPE(exit)
    {
      /* If there was managed memory allocated we need to deallocate it */
      void* sys_mem = interpreted_policy.get_system_mem();
      if (sys_mem)
      {
        auto th_mem_config = interpreted_policy.get_mem_config();
        deallocateManagedMemory(sys_mem, th_mem_config[0], streams[0]);
      }

      unsigned char* hostMemoryArrivedList = interpreted_policy.cg_system.get_arrived_list();
      if (hostMemoryArrivedList)
      {
        deallocateManagedMemory(hostMemoryArrivedList, grid_size, streams[0]);
      }
    };

    /* Should only be allocated / deallocated if the last level used is system wide. Unnecessary and wasteful
     * otherwise. */
    if (grid_size > 1)
    {
      if (interpreted_policy.last_level_scope() == hw_scope::device)
      {
        auto hostMemoryArrivedList = (unsigned char*) allocateManagedMemory(grid_size - 1);
        // printf("About to allocate hostmemarrivedlist : %lu bytes\n", grid_size - 1);
        memset(hostMemoryArrivedList, 0, grid_size - 1);
        interpreted_policy.cg_system = reserved::cooperative_group_system(hostMemoryArrivedList);
      }
    }

    // t.get_stream_grid should return the stream from get_stream if this is not a grid ?
    size_t p_rank = 0;
    for (auto&& p : e_place)
    {
      launch_impl(interpreted_policy, p, f, arg, streams[p_rank], p_rank);
      p_rank++;
    }
  }

private:
  template <typename Fun>
  void run_on_host(Fun&& f)
  {
    assert(!"Not yet implemented");
    abort();
  }

  Arg arg;
  exec_place e_place;
  ::std::string symbol;
  spec_t spec;
  ::std::vector<cudaStream_t> streams;
};

/**
 * @brief A class template for launching tasks with dependencies and execution context.
 * @tparam Ctx Type of the execution context.
 * @tparam Deps Types of dependencies.
 */
template <typename Ctx, typename thread_hierarchy_spec_t, typename... Deps>
class launch_scope
{
public:
  /**
   * @brief Constructor to initialize a launch_scope.
   * @param ctx Execution context.
   * @param spec The specification of the thread hierarchy.
   * @param e_place Execution place (e.g., device, host).
   * @param deps Dependencies for the task to be launched.
   */
  launch_scope(Ctx& ctx, thread_hierarchy_spec_t spec, exec_place e_place, task_dep<Deps>... deps)
      : deps(mv(deps)...)
      , ctx(ctx)
      , e_place(mv(e_place))
      , spec(mv(spec))
  {}

  /// Deleted copy constructor and copy assignment operator.
  launch_scope(const launch_scope&)            = delete;
  launch_scope& operator=(const launch_scope&) = delete;

  /// Move constructor
  launch_scope(launch_scope&&) = default;

  /**
   * @brief Set the symbol for the task.
   * @param s The symbol string.
   * @return Reference to the current object for chaining.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  /**
   * @brief Operator to launch the task.
   * @tparam Fun Type of the function or lambda to be executed.
   * @param f Function or lambda to be executed.
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
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

    static_assert(is_extended_device_lambda_closure_type || is_extended_host_device_lambda_closure_type,
                  "Cannot run launch() on the host");

    EXPECT(e_place != exec_place::host(), "Attempt to run a launch on the host.");

    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    auto t = ctx.task(e_place);

    assert(e_place.affine_data_place() == t.get_affine_data_place());

    /*
     * If we have a grid of places, the implicit affine partitioner is the blocked_partition.
     *
     * An explicit composite data place is required per data dependency to customize this behaviour.
     */
    if (e_place.is_grid())
    {
      // Create a composite data place defined by the grid of places + the partitioning function
      t.set_affine_data_place(data_place::composite(blocked_partition(), e_place.as_grid()));
    }

    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    bool record_time = t.schedule_task();
    // Execution place may have changed during scheduling task
    e_place = t.get_exec_place();

    if (statistics.is_calibrating_to_file())
    {
      record_time = true;
    }

    nvtx_range nr(t.get_symbol().c_str());
    t.start();

    if (dot.is_tracing())
    {
      dot.template add_vertex<typename Ctx::task_type, logical_data_untyped>(t);
    }

    int device;
    cudaEvent_t start_event, end_event;

    if constexpr (::std::is_same_v<Ctx, stream_ctx>)
    {
      if (record_time)
      {
        cudaGetDevice(&device); // We will use this to force it during the next run
        // Events must be created here to avoid issues with multi-gpu
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    const size_t grid_size = e_place.size();

    // Put all data instances in a tuple
    auto args = data2inst<decltype(t), Deps...>(t);

    using th_t      = typename thread_hierarchy_spec_t::thread_hierarchy_t;
    using args_type = decltype(tuple_prepend(th_t(), args));

    auto interpreted_policy = interpreted_execution_policy(spec, e_place, reserved::launch_kernel<Fun, args_type>);

    SCOPE(exit)
    {
      t.end_uncleared();

      if constexpr (::std::is_same_v<Ctx, stream_ctx>)
      {
        /* If there was managed memory allocated we need to deallocate it */
        void* sys_mem = interpreted_policy.get_system_mem();
        if (sys_mem)
        {
          auto th_mem_config = interpreted_policy.get_mem_config();
          deallocateManagedMemory(sys_mem, th_mem_config[0], t.get_stream());
        }

        unsigned char* hostMemoryArrivedList = interpreted_policy.cg_system.get_arrived_list();
        if (hostMemoryArrivedList)
        {
          deallocateManagedMemory(hostMemoryArrivedList, grid_size, t.get_stream());
        }

        if (record_time)
        {
          cuda_safe_call(cudaEventRecord(end_event, t.get_stream()));
          cuda_safe_call(cudaEventSynchronize(end_event));

          float milliseconds = 0;
          cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

          if (dot.is_tracing())
          {
            dot.template add_vertex_timing<stream_task<>>(t, milliseconds, device);
          }

          if (statistics.is_calibrating())
          {
            statistics.log_task_time(t, milliseconds);
          }
        }
      }

      t.clear();
    };

    /* Should only be allocated / deallocated if the last level used is system wide. Unnecessary and wasteful
     * otherwise. */
    if (grid_size > 1)
    {
      if (interpreted_policy.last_level_scope() == hw_scope::device)
      {
        unsigned char* hostMemoryArrivedList;
        hostMemoryArrivedList = (unsigned char*) allocateManagedMemory(grid_size - 1);
        memset(hostMemoryArrivedList, 0, grid_size - 1);
        interpreted_policy.cg_system = reserved::cooperative_group_system(hostMemoryArrivedList);
      }
    }

    size_t p_rank = 0;
    for (auto p : e_place)
    {
      if constexpr (::std::is_same_v<Ctx, stream_ctx>)
      {
        reserved::launch_impl(interpreted_policy, p, f, args, t.get_stream(p_rank), p_rank);
      }
      else
      {
        reserved::graph_launch_impl(t, interpreted_policy, p, f, args, p_rank);
      }
      p_rank++;
    }
  }

private:
  /**
   * @brief Converts a set of slices to a tuple of instances.
   * @tparam T Type of the task.
   * @tparam S Type of the slice.
   * @tparam MoreSlices Types of more slices.
   * @param t The task object.
   * @param i Index (used internally for recursion).
   * @return A tuple containing instances corresponding to slices.
   */
  template <typename T, typename S, typename... MoreSlices>
  auto data2inst(T& t, size_t i = 0)
  {
    S s = t.template get<S>(i);
    if constexpr (sizeof...(MoreSlices) == 0)
    {
      return ::std::make_tuple(s);
    }
    else
    {
      return tuple_prepend(s, data2inst<T, MoreSlices...>(t, i + 1));
    }
  }

  task_dep_vector<Deps...> deps;
  Ctx& ctx;
  exec_place e_place;
  ::std::string symbol;
  thread_hierarchy_spec_t spec;
};

} // namespace reserved

#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
} // end namespace cuda::experimental::stf
