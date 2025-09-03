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
 *
 * @brief Implementation of the cuda_kernel and cuda_kernel_chain constructs
 *
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

#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/task_statistics.cuh>

namespace cuda::experimental::stf
{

class graph_ctx;
class stream_ctx;

namespace reserved
{

template <typename T>
inline constexpr bool is_cufunction_or_cukernel_v = ::std::is_same_v<T, CUfunction> || ::std::is_same_v<T, CUkernel>;

} // end namespace reserved

/**
 * @brief Description of a CUDA kernel
 *
 * This is used to describe kernels passed to the `ctx.cuda_kernel` and
 * `ctx.cuda_kernel_chain` API calls.
 */
struct cuda_kernel_desc
{
  cuda_kernel_desc() = default;

  template <typename Fun, typename... Args>
  cuda_kernel_desc(Fun func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, Args... args)
  {
    configure(mv(func), gridDim_, blockDim_, sharedMem_, mv(args)...);
  }

  template <typename Fun, typename... Args>
  void configure(Fun func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, Args... args)
  {
    // Ensure we are packing arguments of the proper types to call func (only
    // valid with the runtime API)
    static_assert(reserved::is_cufunction_or_cukernel_v<Fun> || ::std::is_invocable_v<Fun, Args...>);

    using TupleType = ::std::tuple<::std::decay_t<Args>...>;

    _CCCL_ASSERT(!configured, "cuda_kernel_desc was already configured");

    func_variant = store_func(mv(func));
    gridDim      = gridDim_;
    blockDim     = blockDim_;
    sharedMem    = sharedMem_;

    // We first copy all arguments into a tuple because the kernel
    // implementation needs pointers to the argument, so we cannot use
    // directly those passed in the pack of arguments
    auto arg_tuple = ::std::make_shared<TupleType>(mv(args)...);

    // Get the address of every tuple entry
    ::std::apply(
      [this](auto&... elems) {
        // Push back the addresses of each tuple element into the args vector
        (args_ptr.push_back(&elems), ...);
      },
      *arg_tuple);

    // Save the tuple in a typed erased value
    arg_tuple_type_erased = mv(arg_tuple);

    configured = true;
  }

  // It is the responsibility of the caller to unsure arguments are valid until
  // the CUDA kernel construct ends
  template <typename Fun>
  void configure_raw(Fun func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, int arg_cnt, const void** args)
  {
    _CCCL_ASSERT(!configured, "cuda_kernel_desc was already configured");

    func_variant = store_func(mv(func));
    gridDim      = gridDim_;
    blockDim     = blockDim_;
    sharedMem    = sharedMem_;

    for (int i = 0; i < arg_cnt; i++)
    {
      // We can safely forget the const here because CUDA will not modify the
      // argument
      args_ptr.push_back(const_cast<void*>(args[i]));
    }

    configured = true;
  }

  /* CUfunction/CUkernel (CUDA driver API) or __global__ function (CUDA runtime API) */
  using func_variant_t = ::std::variant<CUfunction, CUkernel, const void*>;
  func_variant_t func_variant;
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem = 0;

  // Vector of pointers to the arg_tuple which saves arguments in a typed-erased way
  // Mutable so that launch can be const
  mutable ::std::vector<void*> args_ptr;

  // Helper to launch the kernel using CUDA stream based API
  void launch(cudaStream_t stream) const
  {
    _CCCL_ASSERT(func_variant.index() != ::std::variant_npos, "uninitialized variant");

    if (auto* f = ::std::get_if<const void*>(&func_variant))
    {
      cuda_safe_call(cudaLaunchKernel(*f, gridDim, blockDim, args_ptr.data(), sharedMem, stream));
    }
    else
    {
      auto* ker_ptr = ::std::get_if<CUfunction>(&func_variant);
      if (!ker_ptr)
      {
        // If this is a CUkernel, the cast to a CUfunction is sufficient
        ker_ptr = reinterpret_cast<const CUfunction*>(::std::get_if<CUkernel>(&func_variant));
      }

      cuda_safe_call(cuLaunchKernel(
        *ker_ptr,
        gridDim.x,
        gridDim.y,
        gridDim.z,
        blockDim.x,
        blockDim.y,
        blockDim.z,
        sharedMem,
        stream,
        args_ptr.data(),
        nullptr));
    }
  }

  void launch_in_graph(cudaGraphNode_t& node, cudaGraph_t& graph) const
  {
    _CCCL_ASSERT(func_variant.index() != ::std::variant_npos, "uninitialized variant");

    if (auto* f = ::std::get_if<const void*>(&func_variant))
    {
      cudaKernelNodeParams params{
        .func           = const_cast<void*>(*f),
        .gridDim        = gridDim,
        .blockDim       = blockDim,
        .sharedMemBytes = static_cast<unsigned>(sharedMem),
        .kernelParams   = args_ptr.data(),
        .extra          = nullptr};
      cuda_safe_call(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
      return;
    }

    if (auto* func_ptr = ::std::get_if<CUfunction>(&func_variant))
    {
      CUDA_KERNEL_NODE_PARAMS params{
        .func           = *func_ptr,
        .gridDimX       = gridDim.x,
        .gridDimY       = gridDim.y,
        .gridDimZ       = gridDim.z,
        .blockDimX      = blockDim.x,
        .blockDimY      = blockDim.y,
        .blockDimZ      = blockDim.z,
        .sharedMemBytes = static_cast<unsigned>(sharedMem),
        .kernelParams   = const_cast<void**>(args_ptr.data()),
        .extra          = nullptr,
        .kern           = nullptr,
        .ctx            = nullptr};
      cuda_safe_call(cuGraphAddKernelNode(&node, graph, nullptr, 0, &params));
      return;
    }

    auto* ker_ptr = ::std::get_if<CUkernel>(&func_variant);
    _CCCL_ASSERT(ker_ptr, "invalid function");

    CUDA_KERNEL_NODE_PARAMS params{
      .func           = nullptr,
      .gridDimX       = gridDim.x,
      .gridDimY       = gridDim.y,
      .gridDimZ       = gridDim.z,
      .blockDimX      = blockDim.x,
      .blockDimY      = blockDim.y,
      .blockDimZ      = blockDim.z,
      .sharedMemBytes = static_cast<unsigned>(sharedMem),
      .kernelParams   = const_cast<void**>(args_ptr.data()),
      .extra          = nullptr,
      .kern           = *ker_ptr,
      // ctx=nullptr means current context
      .ctx = nullptr};
    cuda_safe_call(cuGraphAddKernelNode(&node, graph, nullptr, 0, &params));
  }

  // Utility to query the number of registers used by this kernel
  int get_num_registers() const
  {
    _CCCL_ASSERT(func_variant.index() != ::std::variant_npos, "uninitialized variant");

    if (auto* f = ::std::get_if<const void*>(&func_variant))
    {
      cudaFuncAttributes func_attr{};
      cuda_safe_call(cudaFuncGetAttributes(&func_attr, *f));
      return func_attr.numRegs;
    }

    auto* fun_ptr = ::std::get_if<CUfunction>(&func_variant);
    if (fun_ptr)
    {
      return cuda_try<cuFuncGetAttribute>(CU_FUNC_ATTRIBUTE_NUM_REGS, *fun_ptr);
    }

    auto* ker_ptr = ::std::get_if<CUkernel>(&func_variant);
    _CCCL_ASSERT(ker_ptr, "invalid kernel");

    auto current_dev = cuda_try<cuCtxGetDevice>();
    return cuda_try<cuKernelGetAttribute>(CU_FUNC_ATTRIBUTE_NUM_REGS, *ker_ptr, current_dev);
  }

private:
  // This type-erased smart pointer keeps the argument tuple valid until the
  // object is destroyed, so that the pointer to these arguments remain valid
  ::std::shared_ptr<void> arg_tuple_type_erased;

  static func_variant_t store_func(CUfunction f)
  {
    return f;
  }

  static func_variant_t store_func(CUkernel k)
  {
    return k;
  }

  template <typename T>
  static func_variant_t store_func(T* f)
  {
    return reinterpret_cast<const void*>(f);
  }

  // We can only configure the kernel descriptor once
  bool configured = false;
};

namespace reserved
{

/**
 * @brief Implementation of the CUDA kernel construct
 *
 * If the chained flag is set, we expect to have a chain of kernels, otherwise a single kernel
 */
template <typename Ctx, bool chained, typename... Deps>
class cuda_kernel_scope
{
public:
  cuda_kernel_scope(Ctx& ctx, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
  {}

  // Provide an explicit execution place
  cuda_kernel_scope(Ctx& ctx, exec_place e_place, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
      , e_place(mv(e_place))
  {}

  cuda_kernel_scope(const cuda_kernel_scope&)            = delete;
  cuda_kernel_scope& operator=(const cuda_kernel_scope&) = delete;
  // move-constructible
  cuda_kernel_scope(cuda_kernel_scope&&) = default;

  auto& set_exec_place(exec_place e_place_)
  {
    e_place = mv(e_place_);
    return *this;
  }

  /// Add a set of dependencies
  template <typename... Pack>
  void add_deps(task_dep_untyped first, Pack&&... pack)
  {
    dynamic_deps.push_back(mv(first));
    if constexpr (sizeof...(Pack) > 0)
    {
      add_deps(::std::forward<Pack>(pack)...);
    }
  }

  template <typename T>
  decltype(auto) get(size_t submitted_index) const
  {
    _CCCL_ASSERT(support_task.has_value(), "uninitialized task");
    return support_task->template get<T>(submitted_index);
  }

  /**
   * @brief Sets the symbol for this object.
   *
   * This method moves the provided string into the internal symbol member and returns a reference to the current
   * object, allowing for method chaining.
   *
   * @param s The string to set as the symbol.
   * @return A reference to the current object.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  auto& start()
  {
    // If a place is specified, use it
    support_task = e_place ? ctx.task(e_place.value()) : ctx.task();

    auto& t = *support_task;

    t.add_deps(deps);

    // Append all dynamic deps
    for (auto& d : dynamic_deps)
    {
      t.add_deps(mv(d));
    }
    dynamic_deps.clear();

    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    // Do we need to measure the duration of the kernel(s) ?
    auto& statistics   = reserved::task_statistics::instance();
    record_time        = t.schedule_task() || statistics.is_calibrating_to_file();
    record_time_device = -1;

    t.start();

    if constexpr (::std::is_same_v<Ctx, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaGetDevice(&record_time_device)); // We will use this to force it during the next run
        // Events must be created here to avoid issues with multi-gpu
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    auto& dot = *ctx.get_dot();
    if (dot.is_tracing())
    {
      dot.template add_vertex<typename Ctx::task_type, logical_data_untyped>(t);
    }

    return *this;
  }

  auto& end()
  {
    auto& t = *support_task;

    // Do submit kernels
    launch_kernels();

    // We need to access the task structures (eg. to get the stream) so we do
    // not clear all its resources yet.
    t.end_uncleared();

    if constexpr (::std::is_same_v<Ctx, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaEventRecord(end_event, t.get_stream()));
        cuda_safe_call(cudaEventSynchronize(end_event));

        float milliseconds = 0;
        cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

        auto& dot = *ctx.get_dot();
        if (dot.is_tracing())
        {
          dot.template add_vertex_timing<typename Ctx::task_type>(t, milliseconds, record_time_device);
        }

        auto& statistics = reserved::task_statistics::instance();
        if (statistics.is_calibrating())
        {
          statistics.log_task_time(t, milliseconds);
        }
      }
    }

    t.clear();

    // Do release to the task structure as we don't need to reference it when
    // we have called end()
    support_task.reset();

    return *this;
  }

  /**
   * @brief Takes a lambda function and executes it on the host in a graph callback node.
   *
   * @tparam Fun type of lambda function
   * @param f Lambda function to execute
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    start();

    SCOPE(exit)
    {
      end();
    };

    auto& t = *support_task;

    // Get the vector of kernel(s) to perform
    // When chained is enable, we expect a vector of kernel description which
    // should be executed one after the other.
    if constexpr (chained)
    {
      kernel_descs = ::std::apply(f, deps.non_void_instance(t));
      assert(!kernel_descs.empty());
    }
    else
    {
      // We have an unchained cuda_kernel, which means there is a single
      // CUDA kernel described, and the function should return a single
      // descriptor, not a vector
      static_assert(!chained);

      cuda_kernel_desc res = ::std::apply(f, deps.non_void_instance(t));
      kernel_descs.push_back(res);
    }
  }

  // Manually add one kernel
  auto& add_kernel_desc(cuda_kernel_desc d)
  {
    kernel_descs.push_back(mv(d));
    return *this;
  }

  // Manually add a vector of kernels
  auto& add_kernel_desc(const ::std::vector<cuda_kernel_desc>& descs)
  {
    for (const auto& d : descs)
    {
      add_kernel_desc(d);
    }
    return *this;
  }

private:
  // This does submit all kernels and print statistics if needed
  void launch_kernels()
  {
    // If CUDASTF_CUDA_KERNEL_DEBUG is set, we display the number of registers
    // used by the kernel(s)
    static const bool display_register_cnt = [] {
      const char* env = ::std::getenv("CUDASTF_CUDA_KERNEL_DEBUG");
      return env && (atoi(env) != 0);
    }();

    // Print some statistics if needed
    if (display_register_cnt)
    {
      if (kernel_descs.size() > 1)
      {
        fprintf(stderr, "cuda_kernel_chain (%s):\n", symbol.c_str());
        for (size_t i = 0; i < kernel_descs.size(); i++)
        {
          fprintf(stderr, "- kernel %ld uses %d register(s)\n", i, kernel_descs[i].get_num_registers());
        }
      }
      else
      {
        fprintf(stderr, "cuda_kernel (%s): uses %d register(s)\n", symbol.c_str(), kernel_descs[0].get_num_registers());
      }
    }

    auto& t = *support_task;

    if constexpr (::std::is_same_v<Ctx, graph_ctx>)
    {
      auto lock = t.lock_ctx_graph();
      auto& g   = t.get_ctx_graph();

      // We have two situations : either there is a single kernel and we put the kernel in the context's
      // graph, or we rely on a child graph
      if (kernel_descs.size() == 1)
      {
        kernel_descs[0].launch_in_graph(t.get_node(), g);
      }
      else
      {
        ::std::vector<cudaGraphNode_t>& chain = t.get_node_chain();
        chain.resize(kernel_descs.size());

        // Create a chain of kernels
        for (size_t i = 0; i < kernel_descs.size(); i++)
        {
          kernel_descs[i].launch_in_graph(chain[i], g);
          if (i > 0)
          {
#if _CCCL_CTK_AT_LEAST(13, 0)
            cuda_safe_call(cudaGraphAddDependencies(g, &chain[i - 1], &chain[i], nullptr, 1));
#else // _CCCL_CTK_AT_LEAST(13, 0)
            cuda_safe_call(cudaGraphAddDependencies(g, &chain[i - 1], &chain[i], 1));
#endif // _CCCL_CTK_AT_LEAST(13, 0)
          }
        }
      }
    }
    else
    {
      // Rely on stream semantic to have a dependency between the kernels
      for (auto& k : kernel_descs)
      {
        k.launch(t.get_stream());
      }
    }
  }

  ::std::string symbol;
  Ctx& ctx;
  // Statically defined deps
  task_dep_vector<Deps...> deps;

  // To store a task that implements cuda_kernel(_chain). Note that we do not
  // store the task with Deps... but a "dynamic" task where all dependencies
  // are added using add_deps.
  using underlying_task_type = decltype(::std::declval<Ctx>().task());
  ::std::optional<underlying_task_type> support_task;

  // Dependencies added with add_deps
  ::std::vector<task_dep_untyped> dynamic_deps;

  ::std::optional<exec_place> e_place;

  // What kernel(s) must be done ? We also store this in a vector if there is a
  // single kernel (with the cuda_kernel construct)
  ::std::vector<cuda_kernel_desc> kernel_descs;

  // Are we making some measurements ?
  bool record_time;
  int record_time_device;
  cudaEvent_t start_event, end_event;
};

} // end namespace reserved
} // end namespace cuda::experimental::stf
