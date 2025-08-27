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
#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/internal/machine.cuh> // backend_ctx_untyped::impl usese machine
#include <cuda/experimental/__stf/internal/reorderer.cuh> // backend_ctx_untyped::impl uses reorderer
#include <cuda/experimental/__stf/internal/repeat.cuh>
#include <cuda/experimental/__stf/internal/scheduler.cuh> // backend_ctx_untyped::impl uses scheduler
#include <cuda/experimental/__stf/internal/slice.cuh> // backend_ctx<T> uses shape_of
#include <cuda/experimental/__stf/internal/thread_hierarchy.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
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

class stream_ctx;

namespace reserved
{

template <typename Ctx, typename exec_place_t, typename shape_t, typename partitioner_t, typename... DepsAndOps>
class parallel_for_scope;

template <typename Ctx, typename thread_hierarchy_spec_t, typename... Deps>
class launch_scope;

template <typename Ctx, bool called_from_launch, typename... Deps>
class host_launch_scope;

template <typename Ctx, bool chained, typename... Deps>
class cuda_kernel_scope;

// We need to have a map of logical data stored in the ctx.
class logical_data_untyped_impl;

} // end namespace reserved

/**
 * @brief This is the underlying context implementation common to all types.
 *
 * We use this class rather than the front-end ones (stream_ctx, graph_ctx,
 * ...) in the internal methods where we don't always know types for example.
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

protected:
  /**
   * @brief This stores states attached to any context, which are not specific to a
   * given backend.
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
      // Forces init
      cudaError_t ret = cudaFree(0);

      // If we are running the task in the context of a CUDA callback, we are
      // not allowed to issue any CUDA API call.
      EXPECT((ret == cudaSuccess || ret == cudaErrorNotPermitted));

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
      generate_event_symbols = dot->is_tracing_prereqs();

      // Record it in the list of all traced contexts
      reserved::dot::instance().track_ctx(dot);
    }

    virtual ~impl()
    {
      // Make sure everything is clean before leaving that context
      _CCCL_ASSERT(dangling_events.size() == 0, "");

      // Otherwise there are tasks which were not completed
      _CCCL_ASSERT(leaves.size() == 0, "");

#ifndef NDEBUG
      _CCCL_ASSERT(total_task_cnt == total_finished_task_cnt, "Not all tasks were finished.");
#endif

      if (logical_data_stats_enabled)
      {
        print_logical_data_summary();
      }

      if (!is_recording_stats)
      {
        return;
      }

      display_transfers();
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

    void set_graph_cache_policy(::std::function<bool()> fn)
    {
      cache_policy = mv(fn);
    }

    ::std::optional<::std::function<bool()>> get_graph_cache_policy() const
    {
      return cache_policy;
    }

    virtual executable_graph_cache_stat* graph_get_cache_stat()
    {
      return nullptr;
    }

#if _CCCL_COMPILER(MSVC)
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code
#endif // _CCCL_COMPILER(MSVC)
    virtual event_list stream_to_event_list(cudaStream_t, ::std::string) const
    {
      fprintf(stderr, "Internal error.\n");
      abort();
      return event_list();
    }
#if _CCCL_COMPILER(MSVC)
    _CCCL_DIAG_POP
#endif // _CCCL_COMPILER(MSVC)

    virtual size_t stage() const
    {
      return size_t(-1);
    }

    virtual ::std::string to_string() const = 0;

    /**
     * @brief Indicate if the backend needs to keep track of dangling events, or if these will be automatically
     * synchronized
     */
    virtual bool track_dangling_events() const = 0;

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

    bool logical_data_stats_enabled = false;
    ::std::vector<::std::pair<::std::string, size_t>> previous_logical_data_stats;

    // We need logical_data_untyped_impl to be defined to print this
    void print_logical_data_summary() const;

    ::std::unordered_map<int, reserved::logical_data_untyped_impl&> logical_data_ids;
    mutable ::std::mutex logical_data_ids_mutex;

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
     * release resources that might have been cached
     */
    void detach_allocators(backend_ctx_untyped& bctx)
    {
      const bool track_dangling = bctx.track_dangling_events();

      // Deinitialize all attached allocators in reversed order
      for (auto it : each(attached_allocators.rbegin(), attached_allocators.rend()))
      {
        auto deinit_res = it->deinit(bctx);
        if (track_dangling)
        {
          add_dangling_events(bctx, mv(deinit_res));
        }
      }

      // Erase the vector of allocators now that they were deinitialized
      attached_allocators.clear();

      // We "duplicate" the code of the deinit to remove any storage and avoid a move
      auto composite_deinit_res = composite_cache.deinit();
      if (track_dangling)
      {
        add_dangling_events(bctx, mv(composite_deinit_res));
      }
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
      attached_allocators.clear();
      // Leave custom_allocator, auto_scheduler, and auto_reordered as they were.
    }

    /* Current context-wide allocator (same as default_allocator unless it is changed) */
    block_allocator_untyped custom_allocator;
    block_allocator_untyped default_allocator;
    block_allocator_untyped uncached_allocator;

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
    ::std::atomic<size_t> total_task_cnt = 0;

#ifndef NDEBUG
    // Keep track of the number of completed tasks in that context
    ::std::atomic<size_t> total_finished_task_cnt = 0;
#endif

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

    /*
     *
     *  Start events : keep track of what events any work in a context depends on
     *
     */
    bool has_start_events() const
    {
      return (start_events.size() > 0);
    }

    void add_start_events(backend_ctx_untyped& bctx, const event_list& lst)
    {
      start_events.merge(lst);

      // We only add events at the beginning of the context, but use them
      // often, so it's good to optimize anyhow
      start_events.optimize(bctx);
    }

    const event_list& get_start_events() const
    {
      return start_events;
    }

    // Events which denote the beginning of the context : any task with no
    // dependency, or logical data with a reference copy should depend on it.
    event_list start_events;

    /*
     * Dangling events : events that we need to synchronize automatically
     * because they would be leaked otherwise (eg. waiting for the events
     * generated by the destructor of a logical data)
     */

    void add_dangling_events(backend_ctx_untyped& bctx, const event_list& lst)
    {
      auto guard = ::std::lock_guard(dangling_events_mutex);
      dangling_events.merge(lst);
      /* If the number of dangling events gets too high, we try to optimize
       * the list to avoid keeping events alive for no reason. */
      if (dangling_events.size() > 16)
      {
        dangling_events.optimize(bctx);
      }
    }

    // Some asynchronous operations cannot be waited on when they occur.
    // For example, when destroying a logical data, it is possible that
    // asynchronous operations are not completed immediately (write back
    // copies, deallocations, ...). A fence can be used to wait on these
    // "dangling" events.
    event_list dangling_events;
    mutable ::std::mutex dangling_events_mutex;

    class leaf_tasks
    {
    public:
      /* Add one task to the leaf tasks */
      void add(const task& t)
      {
        // this will create a new key in the map
        leaf_tasks_mutex.lock();
        event_list& done_prereqs = leaf_tasks[t.get_unique_id()];
        leaf_tasks_mutex.unlock();

        // XXX we need a copy method for event_list
        done_prereqs.merge(t.get_done_prereqs());
      }

      /* Remove one task (if it is still a leaf task, otherwise do nothing) */
      void remove(int task_id)
      {
        // Erase that leaf task if it is found, or do nothing
        auto guard = ::std::lock_guard(leaf_tasks_mutex);
        leaf_tasks.erase(task_id);
      }

      const auto& get_leaf_tasks() const
      {
        return leaf_tasks;
      }

      auto& get_leaf_tasks()
      {
        return leaf_tasks;
      }

      size_t size() const
      {
        return leaf_tasks.size();
      }

      void clear()
      {
        leaf_tasks.clear();
      }

      ::std::mutex leaf_tasks_mutex;

    private:
      // To synchronize with all work submitted in this context, we need to
      // synchronize will all "leaf tasks". Leaf tasks are task that have no
      // outgoing dependencies. Leaf tasks will eventually depend on tasks which
      // are not leaf, so it is sufficient to wait for leaf tasks.
      //
      // Instead of storing tasks, we store a map of id to event lists
      ::std::unordered_map<int /* task_id */, event_list> leaf_tasks;
    };

    // Insert a fence with all pending asynchronous operations on the current context
    [[nodiscard]] inline event_list insert_fence(reserved::per_ctx_dot& dot)
    {
      auto prereqs = event_list();
      // Create a node in the DOT output (if any)
      int fence_unique_id = -1;
      bool dot_is_tracing = dot.is_tracing();
      if (dot_is_tracing)
      {
        fence_unique_id = reserved::unique_id_t();
        dot.add_fence_vertex(fence_unique_id);
      }

      {
        auto guard = ::std::lock_guard(leaves.leaf_tasks_mutex);

        // Sync with the events of all leaf tasks
        for (auto& [t_id, t_done_prereqs] : leaves.get_leaf_tasks())
        {
          // Add the events associated with the termination of that leaf tasks to the list of events
          prereqs.merge(mv(t_done_prereqs));

          // Add an edge between that leaf task and the fence node in the DOT output
          if (dot_is_tracing)
          {
            dot.add_edge(t_id, fence_unique_id, reserved::edge_type::fence);
          }
        }

        /* Remove all leaf tasks */
        leaves.clear();

        /* Erase start events if any */
        start_events.clear();

        _CCCL_ASSERT(leaves.get_leaf_tasks().size() == 0, "");
      }

      {
        // Wait for all pending get() operations associated to frozen logical data
        auto guard = ::std::lock_guard(pending_freeze_mutex);

        for (auto& [fake_t_id, get_prereqs] : pending_freeze)
        {
          // Depend on the get() operation
          prereqs.merge(mv(get_prereqs));

          // Add an edge between that freeze and the fence node in the DOT output
          if (dot_is_tracing)
          {
            dot.add_edge(fake_t_id, fence_unique_id, reserved::edge_type::fence);
          }
        }

        pending_freeze.clear();
      }

      // Sync with events which have not been synchronized with, and which are
      // not "reachable". For example if some async operations occurred in a data
      // handle destructor there could be some remaining events to sync with to
      // make sure data were properly deallocated.
      auto guard = ::std::lock_guard(dangling_events_mutex);
      if (dangling_events.size() > 0)
      {
        prereqs.merge(mv(dangling_events));

        // We consider that dangling events have been sync'ed with, so there is
        // no need to keep track of them.
        dangling_events.clear();
      }

      _CCCL_ASSERT(dangling_events.size() == 0, "");

      return prereqs;
    }

    void add_pending_freeze(const task& fake_t, const event_list& events)
    {
      auto guard = ::std::lock_guard(pending_freeze_mutex);

      // This creates an entry if necessary (there can be multiple gets)
      event_list& prereqs = pending_freeze[fake_t.get_unique_id()];

      // Add these events to the stored list
      prereqs.merge(events);
    }

    // When we unfreeze a logical data, there is no need to automatically sync
    // with the get events because unfreezing implies the get events where
    // sync'ed with
    void remove_pending_freeze(const task& fake_t)
    {
      auto guard = ::std::lock_guard(pending_freeze_mutex);
      pending_freeze.erase(fake_t.get_unique_id());
    }

    leaf_tasks leaves;

  private:
    // Used if we print the task graph using DOT
    ::std::shared_ptr<reserved::per_ctx_dot> dot;

    backend_ctx_untyped::phase ctx_phase = backend_ctx_untyped::phase::setup;
    ::std::optional<::std::function<bool()>> cache_policy;

    // To automatically synchronize with pending get() operartion for
    // frozen_logical_data, we keep track of the events. The freeze operation
    // is identified by the id of the "fake" task, and this map should be
    // cleaned when unfreezing which means it has been synchronized with.
    ::std::unordered_map<int /* fake_task_id */, event_list> pending_freeze;
    ::std::mutex pending_freeze_mutex;
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

#ifndef NDEBUG
  void increment_finished_task_count()
  {
    ++pimpl->total_finished_task_cnt;
  }
#endif

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

  void enable_logical_data_stats()
  {
    pimpl->logical_data_stats_enabled = true;
  }

  cudaGraph_t graph() const
  {
    return pimpl->graph();
  }

  void set_graph_cache_policy(::std::function<bool()> policy)
  {
    pimpl->set_graph_cache_policy(mv(policy));
  }

  auto get_graph_cache_policy() const
  {
    return pimpl->get_graph_cache_policy();
  }

  executable_graph_cache_stat* graph_get_cache_stat()
  {
    return pimpl->graph_get_cache_stat();
  }

  event_list stream_to_event_list(cudaStream_t stream, ::std::string event_symbol) const
  {
    assert(pimpl);
    return pimpl->stream_to_event_list(stream, mv(event_symbol));
  }

  size_t stage() const
  {
    return pimpl->stage();
  }

  ::std::string to_string() const
  {
    return pimpl->to_string();
  }

  bool track_dangling_events() const
  {
    return pimpl->track_dangling_events();
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

  auto dot_section(::std::string symbol) const
  {
    return reserved::dot::section::guard(mv(symbol));
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
    return cuda::experimental::stf::logical_data<T>(*this, make_data_interface<T>(shape), data_place::invalid());
  }

  template <typename T>
  auto logical_data(T prototype, data_place dplace = data_place::host())
  {
    EXPECT(!dplace.is_invalid());
    assert(self());
    return cuda::experimental::stf::logical_data<T>(*this, make_data_interface<T>(prototype), mv(dplace));
  }

  template <typename T, size_t n>
  auto logical_data(T (&array)[n], data_place dplace = data_place::host())
  {
    EXPECT(!dplace.is_invalid());
    return logical_data(make_slice(&array[0], n), mv(dplace));
  }

  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... more_sizes)
  {
    constexpr size_t num_sizes = sizeof...(Sizes) + 1;
    return logical_data(shape_of<slice<T, num_sizes>>(elements, more_sizes...));
  }

  template <typename T>
  auto logical_data(T* p, size_t n, data_place dplace = data_place::host())
  {
    _CCCL_ASSERT(!dplace.is_invalid(), "invalid data place");
    return logical_data(make_slice(p, n), mv(dplace));
  }

  auto token()
  {
    // We do not use a shape because we want the first rw() access to succeed
    // without an initial write()
    //
    // Note that we do not disable write back as the write-back mechanism is
    // handling void_interface specifically to ignore it anyway.
    return logical_data(void_interface{});
  }

  template <typename T>
  frozen_logical_data<T>
  freeze(cuda::experimental::stf::logical_data<T> d,
         access_mode m    = access_mode::read,
         data_place where = data_place::invalid(),
         bool user_freeze = true)
  {
    return frozen_logical_data<T>(*this, mv(d), m, mv(where), user_freeze);
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

  template <typename exec_place_t,
            typename S,
            typename... Deps,
            typename = ::std::enable_if_t<::std::is_base_of_v<exec_place, exec_place_t>>>
  auto parallel_for(exec_place_t e_place, S shape, Deps... deps)
  {
    if constexpr (::std::is_integral_v<S>)
    {
      return parallel_for(mv(e_place), box(shape), mv(deps)...);
    }
    else
    {
      return reserved::parallel_for_scope<Engine, exec_place_t, S, null_partition, Deps...>(
        self(), mv(e_place), mv(shape), mv(deps)...);
    }
  }

  template <typename partitioner_t,
            typename exec_place_t,
            typename S,
            typename... Deps,
            typename = ::std::enable_if_t<std::is_base_of_v<exec_place, exec_place_t>>>
  auto parallel_for([[maybe_unused]] partitioner_t p, exec_place_t e_place, S shape, Deps... deps)
  {
    if constexpr (::std::is_integral_v<S>)
    {
      return parallel_for(mv(p), mv(e_place), box(shape), mv(deps)...);
    }
    else
    {
      return reserved::parallel_for_scope<Engine, exec_place_t, S, partitioner_t, Deps...>(
        self(), mv(e_place), mv(shape), mv(deps)...);
    }
  }

  template <typename S, typename... Deps>
  auto parallel_for(exec_place_grid e_place, S shape, Deps... deps) = delete;

  template <typename S, typename... Deps>
  auto parallel_for(S shape, Deps... deps)
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
