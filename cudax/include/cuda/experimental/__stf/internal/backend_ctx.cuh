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
 * @brief Implements backend_ctx which is the base of all backends, such as `stream_ctx` or `graph_ctx`
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

#include <cuda/experimental/__stf/allocators/block_allocator.cuh>
#include <cuda/experimental/__stf/internal/async_resources_handle.cuh>
#include <cuda/experimental/__stf/internal/execution_policy.cuh> // backend_ctx<T>::launch() uses execution_policy
#include <cuda/experimental/__stf/internal/hooks.cuh>
#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/internal/machine.cuh> // backend_ctx_untyped::impl usese machine
#include <cuda/experimental/__stf/internal/reorderer.cuh> // backend_ctx_untyped::impl uses reorderer
#include <cuda/experimental/__stf/internal/repeat.cuh>
#include <cuda/experimental/__stf/internal/scheduler.cuh> // backend_ctx_untyped::impl uses scheduler
#include <cuda/experimental/__stf/internal/slice.cuh> // backend_ctx<T> uses shape_of
#include <cuda/experimental/__stf/internal/task_state.cuh> // backend_ctx_untyped::impl has-a ctx_stack
#include <cuda/experimental/__stf/internal/thread_hierarchy.cuh>
#include <cuda/experimental/__stf/localization/composite_slice.cuh>

// XXX there is currently a dependency on this header for places.h
// Until we find a solution we need to include this
#include <cuda/experimental/__stf/places/exec/green_context.cuh>

#include <atomic>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace cuda::experimental::stf
{

template <typename T>
class logical_data;

template <typename T>
class frozen_logical_data;

class graph_ctx;

class null_partition;

namespace reserved
{

template <typename Ctx, typename shape_t, typename partitioner_t, typename... Deps>
class parallel_for_scope;

template <typename Ctx, typename thread_hierarchy_spec_t, typename... Deps>
class launch_scope;

/**
 * @brief Result of `host_launch` (below)
 *
 * @tparam Deps Types of dependencies
 *
 * @see `host_launch`
 */
template <typename Ctx, bool called_from_launch, typename... Deps>
class host_launch_scope
{
public:
  host_launch_scope(Ctx& ctx, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
  {}

  host_launch_scope(const host_launch_scope&)            = delete;
  host_launch_scope& operator=(const host_launch_scope&) = delete;
  // move-constructible
  host_launch_scope(host_launch_scope&&) = default;

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

  /**
   * @brief Takes a lambda function and executes it on the host in a graph callback node.
   *
   * @tparam Fun type of lambda function
   * @param f Lambda function to execute
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto t = ctx.task(exec_place::host);
    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    t.start();
    SCOPE(exit)
    {
      t.end();
    };

    auto& dot = *ctx.get_dot();
    if (dot.is_tracing())
    {
      dot.template add_vertex<typename Ctx::task_type, logical_data_untyped>(t);
    }

    auto payload = [&]() {
      if constexpr (called_from_launch)
      {
        return tuple_prepend(thread_hierarchy<>(), deps.instance(t));
      }
      else
      {
        return deps.instance(t);
      }
    }();
    auto* wrapper = new ::std::pair<Fun, decltype(payload)>{::std::forward<Fun>(f), mv(payload)};

    auto callback = [](void* untyped_wrapper) {
      auto w = static_cast<decltype(wrapper)>(untyped_wrapper);
      SCOPE(exit)
      {
        delete w;
      };
      ::std::apply(::std::forward<Fun>(w->first), mv(w->second));
    };

    if constexpr (::std::is_same_v<Ctx, graph_ctx>)
    {
      cudaHostNodeParams params = {.fn = callback, .userData = wrapper};
      // Put this host node into the child graph that implements the graph_task<>
      cuda_safe_call(cudaGraphAddHostNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &params));
    }
    else
    {
      cuda_safe_call(cudaLaunchHostFunc(t.get_stream(), callback, wrapper));
    }
  }

private:
  ::std::string symbol;
  Ctx& ctx;
  task_dep_vector<Deps...> deps;
};

} // end namespace reserved

/**
 * @brief Description of a CUDA kernel
 *
 * This is used to describe kernels passed to the `ctx.cuda_kernel` and
 * `ctx.cuda_kernel_chain` API calls.
 */
struct cuda_kernel_desc
{
  template <typename Fun, typename... Args>
  cuda_kernel_desc(Fun func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, Args... args)
      : func((const void*) func)
      , gridDim(gridDim_)
      , blockDim(blockDim_)
      , sharedMem(sharedMem_)
  {
    using TupleType = ::std::tuple<::std::decay_t<Args>...>;

    // We first copy all arguments into a tuple because the kernel
    // implementation needs pointers to the argument, so we cannot use
    // directly those passed in the pack of arguments
    auto arg_tuple = ::std::make_shared<TupleType>(std::forward<Args>(args)...);

    // Ensure we are packing arguments of the proper types to call func
    static_assert(::std::is_invocable_v<Fun, Args...>);

    // Get the address of every tuple entry
    ::std::apply(
      [this](auto&... elems) {
        // Push back the addresses of each tuple element into the args vector
        ((args_ptr.push_back(static_cast<void*>(&elems))), ...);
      },
      *arg_tuple);

    // Save the tuple in a typed erased value
    arg_tuple_type_erased = mv(arg_tuple);
  }

  /* __global__ function */
  const void* func;
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem;

  // Vector of pointers to the arg_tuple which saves arguments in a typed-erased way
  ::std::vector<void*> args_ptr;

private:
  ::std::shared_ptr<void> arg_tuple_type_erased;
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

  /**
   * @brief Takes a lambda function and executes it on the host in a graph callback node.
   *
   * @tparam Fun type of lambda function
   * @param f Lambda function to execute
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    // If a place is specified, use it
    auto t = e_place ? ctx.task(e_place.value()) : ctx.task();

    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    t.start();
    SCOPE(exit)
    {
      t.end();
    };

    auto& dot = *ctx.get_dot();
    if (dot.is_tracing())
    {
      dot.template add_vertex<typename Ctx::task_type, logical_data_untyped>(t);
    }

    // When chained is enable, we expect a vector of kernel description which should be executed one after the other
    if constexpr (chained)
    {
      ::std::vector<cuda_kernel_desc> res = ::std::apply(f, deps.instance(t));
      assert(!res.empty());

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        // We have two situations : either there is a single kernel and we put the kernel in the context's
        // graph, or we rely on a child graph
        if (res.size() == 1)
        {
          insert_one_kernel(res[0], t.get_node(), t.get_ctx_graph());
        }
        else
        {
          // Get the (child) graph associated to the task
          auto g = t.get_graph();

          cudaGraphNode_t n      = nullptr;
          cudaGraphNode_t prev_n = nullptr;

          // Create a chain of kernels
          for (size_t i = 0; i < res.size(); i++)
          {
            if (i > 0)
            {
              prev_n = n;
            }

            insert_one_kernel(res[i], n, g);
            if (i > 0)
            {
              cuda_safe_call(cudaGraphAddDependencies(g, &prev_n, &n, 1));
            }
          }
        }
      }
      else
      {
        // Rely on stream semantic to have a dependency between the kernels
        for (auto& k : res)
        {
          cuda_safe_call(
            cudaLaunchKernel(k.func, k.gridDim, k.blockDim, k.args_ptr.data(), k.sharedMem, t.get_stream()));
        }
      }
    }
    else
    {
      // We have an unchained cuda_kernel, which means there is a single
      // CUDA kernel described, and the function should return a single
      // descriptor, not a vector
      static_assert(!chained);

      cuda_kernel_desc res = ::std::apply(f, deps.instance(t));

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        insert_one_kernel(res, t.get_node(), t.get_ctx_graph());
      }
      else
      {
        cuda_safe_call(
          cudaLaunchKernel(res.func, res.gridDim, res.blockDim, res.args_ptr.data(), res.sharedMem, t.get_stream()));
      }
    }
  }

private:
  /* Add a kernel to a CUDA graph given its description */
  auto insert_one_kernel(cuda_kernel_desc& k, cudaGraphNode_t& n, cudaGraph_t& g) const
  {
    cudaKernelNodeParams kconfig;
    kconfig.blockDim       = k.blockDim;
    kconfig.extra          = nullptr;
    kconfig.func           = const_cast<void*>(k.func);
    kconfig.gridDim        = k.gridDim;
    kconfig.kernelParams   = k.args_ptr.data();
    kconfig.sharedMemBytes = k.sharedMem;
    cuda_safe_call(cudaGraphAddKernelNode(&n, g, nullptr, 0, &kconfig));
  }

  ::std::string symbol;
  Ctx& ctx;
  task_dep_vector<Deps...> deps;
  ::std::optional<exec_place> e_place;
};

} // end namespace reserved

/**
 * @brief Unified context!!!
 *
 */
class backend_ctx_untyped
{
public:
  /**
   * @brief current context status
   *
   * We keep track of the status of context so that we do not make API calls at an
   * inappropriate time, such as synchronizing twice.
   */
  enum class phase
  {
    setup, // the context is getting initialized
    submitted, // between acquire and release
    finalized, // we have called finalize
  };

#if 0
    static ::std::string phase_to_string(phase p) {
        switch (p) {
        case phase::setup: return ::std::string("setup");
        case phase::submitted: return ::std::string("submitted");
        case phase::finalized: return ::std::string("finalized");
        }
        return ::std::string("error");
   }
#endif

protected:
  /**
   * @brief This stores states attached to any context, which are not specific to a
   * given backend. For example the stack of tasks.
   */
  class impl
  {
  public:
    friend class backend_ctx_untyped;

    impl(async_resources_handle async_resources = async_resources_handle())
        : auto_scheduler(reserved::scheduler::make(getenv("CUDASTF_SCHEDULE")))
        , auto_reorderer(reserved::reorderer::make(getenv("CUDASTF_TASK_ORDER")))
        , async_resources(async_resources ? mv(async_resources) : async_resources_handle())
    {
      // Enable peer memory accesses (if not done already)
      reserved::machine::instance().enable_peer_accesses();

      // If CUDASTF_DISPLAY_STATS is set to a non 0 value, record stats
      const char* record_stats_env = getenv("CUDASTF_DISPLAY_STATS");
      if (record_stats_env && atoi(record_stats_env) != 0)
      {
        is_recording_stats = true;
      }

      // Initialize a structure to generate a visualization of the activity in this context
      dot = ::std::make_shared<reserved::per_ctx_dot>(
        reserved::dot::instance().is_tracing(),
        reserved::dot::instance().is_tracing_prereqs(),
        reserved::dot::instance().is_timing());

      // We generate symbols if we may use them
#ifdef CUDASTF_DEBUG
      generate_event_symbols = true;
#else
      generate_event_symbols = dot->is_tracing_prereqs();
#endif
      // Record it in the list of all traced contexts
      reserved::dot::instance().per_ctx.push_back(dot);
    }

    virtual ~impl()
    {
      // We can't assert here because there may be tasks inside tasks
      //_CCCL_ASSERT(total_task_cnt == 0, "You created some tasks but forgot to call finalize().");

      if (!is_recording_stats)
      {
        return;
      }

      display_transfers();

      fprintf(stderr, "TOTAL SYNC COUNT: %lu\n", reserved::counter<reserved::join_tag>::load());
    }

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Due to circular dependencies, every context defines the same, but we
    // cannot implement it here.
    virtual void update_uncached_allocator(block_allocator_untyped custom) = 0;

    virtual cudaGraph_t graph() const
    {
      return nullptr;
    }

#if defined(_CCCL_COMPILER_MSVC)
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code
#endif // _CCCL_COMPILER_MSVC
    virtual event_list stream_to_event_list(cudaStream_t, ::std::string) const
    {
      fprintf(stderr, "Internal error.\n");
      abort();
      return event_list();
    }
#if defined(_CCCL_COMPILER_MSVC)
    _CCCL_DIAG_POP
#endif // _CCCL_COMPILER_MSVC

    virtual size_t epoch() const
    {
      return size_t(-1);
    }

    auto& get_default_allocator()
    {
      return default_allocator;
    }

    auto& get_uncached_allocator()
    {
      return uncached_allocator;
    }

    /**
     * @brief Write-back data and erase automatically created data instances
     * The implementation requires logical_data_untyped_impl to be complete
     */
    void erase_all_logical_data();

    /**
     * @brief Add an allocator to the vector of allocators which will be
     * deinitialized when the context is finalized
     */
    void attach_allocator(block_allocator_untyped a)
    {
      attached_allocators.push_back(mv(a));
    }

    /**
     * @brief Detach all allocators previously attached in this context to
     * release ressources that might have been cached
     */
    void detach_allocators(backend_ctx_untyped& bctx)
    {
      // Deinitialize all attached allocators in reversed order
      for (auto it : each(attached_allocators.rbegin(), attached_allocators.rend()))
      {
        stack.add_dangling_events(it->deinit(bctx));
      }

      // Erase the vector of allocators now that they were deinitialized
      attached_allocators.clear();

      stack.add_dangling_events(composite_cache.deinit());
    }

    void display_transfers() const
    {
      //            fprintf(stderr, "display_transfers() => transfers.size() %ld\n", transfers.size());
      //            if (transfers.size() == 0)
      //                return;

      size_t total_cnt   = 0;
      size_t total_bytes = 0;

      fprintf(stderr, "CTX STATS\n");
      for (auto& e : transfers)
      {
        ::std::pair<int, int> nodes     = e.first;
        ::std::pair<size_t, size_t> res = e.second;
        fprintf(stderr, "\t%d->%d : cnt %zu (%zu bytes)\n", nodes.first, nodes.second, res.first, res.second);
        total_cnt += res.first;
        total_bytes += res.second;
      }

      fprintf(stderr, "TOTAL: %zu transfers (%zu bytes)\n", total_cnt, total_bytes);
    }

    void cleanup()
    {
      // assert(!stack.hasCurrentTask());
      attached_allocators.clear();
      total_task_cnt.store(0);
      // Leave custom_allocator, auto_scheduler, and auto_reordered as they were.
    }

    /* Current context-wide allocator (same as default_allocator unless it is changed) */
    block_allocator_untyped custom_allocator;
    block_allocator_untyped default_allocator;
    block_allocator_untyped uncached_allocator;
    reserved::ctx_stack stack;

    // A vector of all allocators used in this ctx, so that they are
    // destroyed when calling finalize()
    ::std::vector<block_allocator_untyped> attached_allocators;
    reserved::composite_slice_cache composite_cache;

    ::std::unique_ptr<reserved::scheduler> auto_scheduler;
    ::std::unique_ptr<reserved::reorderer> auto_reorderer;
    // Stats-related stuff
    ::std::unordered_map<::std::pair<int, int>,
                         ::std::pair<size_t, size_t>,
                         cuda::experimental::stf::hash<::std::pair<int, int>>>
      transfers;
    bool is_recording_stats = false;
    // Keep track of the number of tasks generated in the context
    ::std::atomic<size_t> total_task_cnt;

    // This data structure contains all resources useful for an efficient
    // asynchronous execution. This will for example contain pools of CUDA
    // streams which are costly to create.
    //
    // We use an optional to avoid instantiating it until we have initialized it
    async_resources_handle async_resources;

    // Do we need to generate symbols for events ? This is true when we are
    // in debug mode (as we may inspect structures with a debugger, or when
    // generating a dot output)
    bool generate_event_symbols = false;

    ::std::shared_ptr<reserved::per_ctx_dot>& get_dot()
    {
      return dot;
    }

    const ::std::shared_ptr<reserved::per_ctx_dot>& get_dot() const
    {
      return dot;
    }

    auto get_phase() const
    {
      return ctx_phase;
    }

    void set_phase(backend_ctx_untyped::phase p)
    {
      ctx_phase = p;
    }

    bool has_start_events() const
    {
      return stack.has_start_events();
    }

    const event_list& get_start_events() const
    {
      return stack.get_start_events();
    }

  private:
    // Used if we print the task graph using DOT
    ::std::shared_ptr<reserved::per_ctx_dot> dot;

    backend_ctx_untyped::phase ctx_phase = backend_ctx_untyped::phase::setup;
  };

public:
  backend_ctx_untyped() = delete;

  backend_ctx_untyped(::std::shared_ptr<impl> impl)
      : pimpl(mv(impl))
  {
    assert(pimpl);
  }

  explicit operator bool() const
  {
    return pimpl != nullptr;
  }

  bool operator==(const backend_ctx_untyped& rhs) const
  {
    return pimpl == rhs.pimpl;
  }

  bool operator!=(const backend_ctx_untyped& rhs) const
  {
    return !(*this == rhs);
  }

  async_resources_handle& async_resources() const
  {
    assert(pimpl);
    assert(pimpl->async_resources);
    return pimpl->async_resources;
  }

  auto& get_stack()
  {
    assert(pimpl);
    return pimpl->stack;
  }

  bool reordering_tasks() const
  {
    assert(pimpl);
    return pimpl->auto_reorderer != nullptr;
  }

  auto& get_composite_cache()
  {
    return pimpl->composite_cache;
  }

  ::std::pair<exec_place, bool> schedule_task(const task& t) const
  {
    assert(pimpl);
    assert(pimpl->auto_scheduler);
    return pimpl->auto_scheduler->schedule_task(t);
  }

  void reorder_tasks(::std::vector<int>& tasks, ::std::unordered_map<int, reserved::reorderer_payload>& task_map)
  {
    assert(pimpl);
    assert(pimpl->auto_reorderer);
    pimpl->auto_reorderer->reorder_tasks(tasks, task_map);
  }

  void increment_task_count()
  {
    ++pimpl->total_task_cnt;
  }

  size_t task_count() const
  {
    return pimpl->total_task_cnt;
  }

  /* Customize the allocator used by all logical data */
  void set_allocator(block_allocator_untyped custom)
  {
    pimpl->custom_allocator = mv(custom);
  }

  /* Customize the uncached allocator used by other allocators */
  void set_uncached_allocator(block_allocator_untyped custom)
  {
    pimpl->uncached_allocator = mv(custom);
  }

  auto& get_allocator()
  {
    return pimpl->custom_allocator;
  }
  const auto& get_allocator() const
  {
    return pimpl->custom_allocator;
  }

  auto& get_default_allocator()
  {
    return pimpl->get_default_allocator();
  }

  auto& get_uncached_allocator()
  {
    return pimpl->get_uncached_allocator();
  }

  void update_uncached_allocator(block_allocator_untyped uncached_allocator)
  {
    pimpl->update_uncached_allocator(mv(uncached_allocator));
  }

  void attach_allocator(block_allocator_untyped a)
  {
    pimpl->attach_allocator(mv(a));
  }

  void add_transfer(const data_place& src_node, const data_place& dst_node, size_t s)
  {
    if (!pimpl->is_recording_stats)
    {
      return;
    }
    ::std::pair<int, int> nodes(device_ordinal(src_node), device_ordinal(dst_node));
    // Increment the count for the pair
    pimpl->transfers[nodes].first++;
    // Add the value of s to the sum for the pair
    pimpl->transfers[nodes].second += s;
  }

  bool generate_event_symbols() const
  {
    return pimpl->generate_event_symbols;
  }

  cudaGraph_t graph() const
  {
    return pimpl->graph();
  }

  event_list stream_to_event_list(cudaStream_t stream, ::std::string event_symbol) const
  {
    assert(pimpl);
    return pimpl->stream_to_event_list(stream, mv(event_symbol));
  }

  size_t epoch() const
  {
    return pimpl->epoch();
  }

  // protected:
  impl& get_state()
  {
    assert(pimpl);
    return *pimpl;
  }

  const impl& get_state() const
  {
    assert(pimpl);
    return *pimpl;
  }

  const auto& get_dot() const
  {
    assert(pimpl);
    return pimpl->get_dot();
  }

  auto& get_dot()
  {
    assert(pimpl);
    return pimpl->get_dot();
  }

  template <typename parent_ctx_t>
  void set_parent_ctx(parent_ctx_t& parent_ctx)
  {
    reserved::per_ctx_dot::set_parent_ctx(parent_ctx.get_dot(), get_dot());
  }

  auto get_phase() const
  {
    return pimpl->get_phase();
  }

  void set_phase(backend_ctx_untyped::phase p)
  {
    pimpl->set_phase(p);
  }

  bool has_start_events() const
  {
    return pimpl->has_start_events();
  }

  const event_list& get_start_events() const
  {
    return pimpl->get_start_events();
  }

  // Shortcuts to manipulate the current affinity stored in the async_resources_handle of the ctx
  void push_affinity(::std::vector<::std::shared_ptr<exec_place>> p) const
  {
    async_resources().push_affinity(mv(p));
  }
  void push_affinity(::std::shared_ptr<exec_place> p) const
  {
    async_resources().push_affinity(mv(p));
  }
  void pop_affinity() const
  {
    async_resources().pop_affinity();
  }

  const ::std::vector<::std::shared_ptr<exec_place>>& current_affinity() const
  {
    return async_resources().current_affinity();
  }

  const exec_place& current_exec_place() const
  {
    _CCCL_ASSERT(has_affinity(), "current_exec_place() cannot be called without setting the affinity first.");
    assert(current_affinity().size() > 0);
    return *(current_affinity()[0]);
  }

  bool has_affinity() const
  {
    return async_resources().has_affinity();
  }

  // Determines the default execution place for a given context, which
  // corresponds to the execution place when no place is provided.
  //
  // By default, we select the current device, unless an affinity was set in the
  // context, in which case we take the first execution place in the current
  // places.
  exec_place default_exec_place() const
  {
    return has_affinity() ? current_exec_place() : exec_place::current_device();
  }

  // Automatically pick a CUDA stream from the pool attached to the current
  // execution place
  auto pick_dstream()
  {
    return default_exec_place().get_stream_pool(async_resources(), true).next();
  }
  cudaStream_t pick_stream()
  {
    return pick_dstream().stream;
  }

private:
  ::std::shared_ptr<impl> pimpl;
};

/**
 * @brief This is a placeholder class so that we can put common utilities to design a
 * backend ctx. The state of the backend itself is put elsewhere.
 */
template <typename Engine>
class backend_ctx : public backend_ctx_untyped
{
public:
  backend_ctx(::std::shared_ptr<impl> impl)
      : backend_ctx_untyped(mv(impl))
  {
    static_assert(sizeof(*this) == sizeof(backend_ctx_untyped), "Derived value type cannot add state.");
  }

  ~backend_ctx() = default;

  /**
   * @brief Returns a `logical_data` object with the given shape, tied to this graph. Initial data place is invalid.
   *
   * @tparam T Underlying type for the logical data object
   * @param shape shape of the created object
   * @return `logical_data<T>` usable with this graph
   */
  template <typename T>
  cuda::experimental::stf::logical_data<T> logical_data(shape_of<T> shape)
  {
    return cuda::experimental::stf::logical_data<T>(*this, make_data_interface<T>(shape), data_place::invalid);
  }

  template <typename T>
  auto logical_data(T prototype, data_place dplace = data_place::host)
  {
    EXPECT(dplace != data_place::invalid);
    assert(self());
    return cuda::experimental::stf::logical_data<T>(*this, make_data_interface<T>(prototype), mv(dplace));
  }

  template <typename T, size_t n>
  auto logical_data(T (&array)[n], data_place dplace = data_place::host)
  {
    EXPECT(dplace != data_place::invalid);
    return logical_data(make_slice(&array[0], n), mv(dplace));
  }

  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... more_sizes)
  {
    constexpr size_t num_sizes = sizeof...(Sizes) + 1;
    return logical_data(shape_of<slice<T, num_sizes>>(elements, more_sizes...));
  }

  template <typename T>
  auto logical_data(T* p, size_t n, data_place dplace = data_place::host)
  {
    EXPECT(dplace != data_place::invalid);
    return logical_data(make_slice(p, n), mv(dplace));
  }

  template <typename T>
  frozen_logical_data<T> freeze(cuda::experimental::stf::logical_data<T> d,
                                access_mode m    = access_mode::read,
                                data_place where = data_place::invalid)
  {
    return frozen_logical_data<T>(*this, mv(d), m, mv(where));
  }

  /**
   * @brief Creates a typed task on the current CUDA device
   * @return An instantiation of `task` with the appropriate arguments, suitable for use with `operator->*`.
   */
  template <typename... Deps>
  auto task(task_dep<Deps>... deps)
  {
    return self().task(self().default_exec_place(), mv(deps)...);
  }

  /**
   * @brief Creates an object able to launch a lambda function on the host.
   *
   * @tparam Deps Dependency types
   * @param deps dependencies
   * @return `host_launch_scope<Deps...>` ready for the `->*` operator
   */
  template <typename... Deps>
  auto host_launch(task_dep<Deps>... deps)
  {
    return reserved::host_launch_scope<Engine, false, Deps...>(self(), mv(deps)...);
  }

  template <typename... Deps>
  auto cuda_kernel(task_dep<Deps>... deps)
  {
    return reserved::cuda_kernel_scope<Engine, false, Deps...>(self(), mv(deps)...);
  }

  template <typename... Deps>
  auto cuda_kernel_chain(task_dep<Deps>... deps)
  {
    return reserved::cuda_kernel_scope<Engine, true, Deps...>(self(), mv(deps)...);
  }

  template <typename thread_hierarchy_spec_t, typename... Deps>
  auto launch(thread_hierarchy_spec_t spec, exec_place e_place, task_dep<Deps>... deps)
  {
    return reserved::launch_scope<Engine, thread_hierarchy_spec_t, Deps...>(self(), mv(spec), mv(e_place), mv(deps)...);
  }

  /* Using ctx.launch with a host place */
  template <typename... Deps>
  auto launch(exec_place_host, task_dep<Deps>... deps)
  {
    return reserved::host_launch_scope<Engine, true, Deps...>(self(), mv(deps)...);
  }

  /* Default execution policy, explicit place */
  // default depth to avoid breaking all codes (XXX temporary)
  template <typename... Deps>
  auto launch(exec_place e_place, task_dep<Deps>... deps)
  {
    return launch(par(par()), mv(e_place), mv(deps)...);
  }

  /* Default execution policy, on automatically selected device */
  template <typename... Deps>
  auto launch(task_dep<Deps>... deps)
  {
    return launch(self().default_exec_place(), mv(deps)...);
  }

  auto repeat(size_t count)
  {
    return reserved::repeat_scope(self(), count);
  }

  auto repeat(::std::function<bool()> condition)
  {
    return reserved::repeat_scope(self(), mv(condition));
  }

  /*
   * parallel_for : apply an operation over a shaped index space
   */
  template <typename S, typename... Deps>
  auto parallel_for(exec_place e_place, S shape, task_dep<Deps>... deps)
  {
    return reserved::parallel_for_scope<Engine, S, null_partition, Deps...>(self(), mv(e_place), mv(shape), mv(deps)...);
  }

  template <typename partitioner_t, typename S, typename... Deps>
  auto parallel_for(partitioner_t, exec_place e_place, S shape, task_dep<Deps>... deps)
  {
    return reserved::parallel_for_scope<Engine, S, partitioner_t, Deps...>(self(), mv(e_place), mv(shape), mv(deps)...);
  }

  template <typename S, typename... Deps>
  auto parallel_for(exec_place_grid e_place, S shape, task_dep<Deps>... deps) = delete;

  template <typename partitioner_t, typename S, typename... Deps>
  auto parallel_for(partitioner_t p, exec_place_grid e_place, S shape, task_dep<Deps>... deps)
  {
    return parallel_for(mv(p), exec_place(mv(e_place)), mv(shape), mv(deps)...);
  }

  template <typename S, typename... Deps>
  auto parallel_for(S shape, task_dep<Deps>... deps)
  {
    return parallel_for(self().default_exec_place(), mv(shape), mv(deps)...);
  }

private:
  Engine& self()
  {
    static_assert(::std::is_base_of_v<backend_ctx<Engine>, Engine>);
    return static_cast<Engine&>(*this);
  }

  template <typename T, typename... P>
  auto make_data_interface(P&&... p)
  {
    return ::std::make_shared<typename Engine::template data_interface<T>>(::std::forward<P>(p)...);
  }
};

} // end namespace cuda::experimental::stf
