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
 * @brief Stackable context and logical data to nest contexts
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

#include "cuda/experimental/__stf/allocators/adapters.cuh"
#include "cuda/experimental/__stf/utility/hash.cuh"
#include "cuda/experimental/__stf/utility/source_location.cuh"
#include "cuda/experimental/stf.cuh"

/**
 * TODO insert a big comment explaining the design and how to reason about this !
 */

namespace cuda::experimental::stf
{

template <typename T>
class stackable_logical_data;

template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep;

namespace reserved
{

// This helper converts stackable_task_dep to the underlying task_dep. If we
// have a stackable_logical_data A, A.read() is indeed a stackable_task_dep,
// which we can pass to stream_ctx/graph_ctx constructs by extracting the
// underlying task_dep.
//
// By default, return the argument as-is (perfect forwarding)
template <typename U>
decltype(auto) to_task_dep(U&& u)
{
  return ::std::forward<U>(u);
}

// Overload for stackable_task_dep (non-const version)
template <typename T, typename reduce_op, bool initialize>
task_dep<T, reduce_op, initialize>& to_task_dep(stackable_task_dep<T, reduce_op, initialize>& sdep)
{
  return sdep.underlying_dep();
}

// Overload for stackable_task_dep (const version)
template <typename T, typename reduce_op, bool initialize>
const task_dep<T, reduce_op, initialize>& to_task_dep(const stackable_task_dep<T, reduce_op, initialize>& sdep)
{
  return sdep.underlying_dep();
}

template <typename T, typename reduce_op, bool initialize>
task_dep<T, reduce_op, initialize> to_task_dep(stackable_task_dep<T, reduce_op, initialize>&& sdep)
{
  // Return by value or whatever makes sense:
  return ::std::move(sdep.underlying_dep());
}

} // end namespace reserved

/**
 * @brief Base class with a virtual pop method to enable type erasure
 *
 * This is used to implement the automatic call to pop() on logical data when a
 * context node is popped.
 */
class stackable_logical_data_impl_state_base
{
public:
  virtual ~stackable_logical_data_impl_state_base()                         = default;
  virtual void pop_before_finalize() const                                  = 0;
  virtual void pop_after_finalize(const event_list& finalize_prereqs) const = 0;
};

/**
 * @brief This class defines a context that behaves as a context which can have nested subcontexts (implemented as local
 * CUDA graphs)
 */
class stackable_ctx
{
public:
  class impl
  {
  private:
    /*
     * State of each nested context
     */
    struct ctx_node
    {
      ctx_node(context ctx, cudaStream_t support_stream, ::std::shared_ptr<stream_adapter> alloc_adapters)
          : ctx(mv(ctx))
          , support_stream(mv(support_stream))
          , alloc_adapters(mv(alloc_adapters))
      {}

      // To avoid prematurely destroying data created in a nested context, we need to hold a reference to them
      //
      // This happens for example in this case where we want to defer the release
      // of the resources of "a" until we call pop() because this is when we would
      // have submitted the CUDA graph where "a" is used. Destroying it earlier
      // would mean we destroy that memory before the graph is even launched.
      //
      // ctx.push()
      // {
      //    auto a = ctx.logical_data(...);
      //    ... use a ...
      // }
      // ctx.pop()
      void retain_data(::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
      {
        // This keeps a reference to the shared_ptr until retained_data is destroyed
        retained_data.push_back(mv(data_impl));
      }

      void track_pushed_data(::std::shared_ptr<stackable_logical_data_impl_state_base> data_impl)
      {
        _CCCL_ASSERT(data_impl, "invalid value");
        pushed_data.push_back(mv(data_impl));
      }

      context ctx;
      cudaStream_t support_stream;
      // A wrapper to forward allocations from a node to its parent node (none is used at the root level)
      ::std::shared_ptr<stream_adapter> alloc_adapters;

      // This map keeps track of the logical data that were pushed in this ctx node
      // key: logical data's unique id
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> pushed_data;

      // Where was the push() called ?
      _CUDA_VSTD::source_location callsite;

    private:
      // If we want to keep the state of some logical data implementations until this node is popped
      ::std::vector<::std::shared_ptr<stackable_logical_data_impl_state_base>> retained_data;
    };

    class node_hierarchy
    {
    public:
      node_hierarchy()
      {
        // May grow up later if more contexts are needed
        int initialize_size = 16;
        parent.resize(initialize_size);
        children.resize(initialize_size);
        free_list.resize(initialize_size);

        for (int i = 0; i < initialize_size; i++)
        {
          // The order of the nodes does not really matter, but it may be
          // easier to follow if we get nodes in a reasonable order (sequence
          // from 0 ...)
          free_list.push_back(initialize_size - 1 - i);
        }
      }

      int get_avail_entry()
      {
        // XXX implement growth mechanism
        // remember size of parent, push new items in the free list ? (grow())
        _CCCL_ASSERT(!free_list.size(), "no slot available");

        int res = free_list.back();
        free_list.pop_back();

        parent[res] = -1;
        // the node should be unused
        _CCCL_ASSERT(children[res].size() == 0, "invalid state");

        return res;
      }

      // Make the node available again
      void discard_node(int offset)
      {
        // Remove this child from it's parent (if any)
        int p = parent[offset];
        if (p != -1)
        {
          bool found = false;
          ::std::vector<int> new_children;
          new_children.resize(children[p].size() - 1);
          for (auto c : children[p])
          {
            if (c == offset)
            {
              found = true;
            }
            else
            {
              new_children.push_back(c);
            }
          }

          // Ensure we did find the node in it's parent's children
          _CCCL_ASSERT(found, "invalid hierarchy state");
          ::std::swap(children[p], new_children);
        }

        children[offset].clear();
        parent[offset] = -1;
      }

      void set_parent(int parent_offset, int child_offset)
      {
        parent[child_offset] = parent_offset;

        children[parent_offset].push_back(child_offset);
      }

    private:
      // Offset of the node's parent : -1 if none. Only valid for entries not in free-list.
      ::std::vector<int> parent;

      // If a node has children, indicate their offset here
      ::std::vector<::std::vector<int>> children;

      // Available offsets to create new nodes
      ::std::vector<int> free_list;
    };

  public:
    impl()
    {
      // Stats are disabled by default, unless a non null value is passed to this env variable
      const char* display_graph_stats_str = getenv("CUDASTF_DISPLAY_GRAPH_STATS");
      display_graph_stats                 = (display_graph_stats_str && atoi(display_graph_stats_str) != 0);

      // Create the root node
      push(_CUDA_VSTD::source_location::current());
    }

    ~impl()
    {
      print_cache_stats_summary();
    }

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Define move constructor and move assignment operator
    impl(impl&&) noexcept            = default;
    impl& operator=(impl&&) noexcept = default;

    /**
     * @brief Create a new nested level
     */
    void push(const _CUDA_VSTD::source_location& loc)
    {
      // fprintf(stderr, "stackable_ctx::push() depth() was %ld\n", depth());

      // These resources are not destroyed when we pop, so we create it only if needed
      if (async_handles.size() < nodes.size())
      {
        async_handles.emplace_back();
      }

      if (nodes.size() == 0)
      {
        nodes.emplace_back(stream_ctx(), nullptr, nullptr);
      }
      else
      {
        // Get a stream from previous context (we haven't pushed the new one yet)
        cudaStream_t stream = nodes[depth()].ctx.pick_stream();

        auto gctx = graph_ctx(stream, async_handles.back());

        // Useful for tools
        gctx.set_parent_ctx(nodes[depth()].ctx);
        gctx.get_dot()->set_ctx_symbol("stacked_ctx_" + ::std::to_string(nodes.size()));

        auto wrapper = ::std::make_shared<stream_adapter>(gctx, stream);

        gctx.update_uncached_allocator(wrapper->allocator());

        nodes.emplace_back(gctx, stream, wrapper);

        if (display_graph_stats)
        {
          nodes.back().callsite = loc;
        }
      }
    }

    /**
     * @brief Terminate the current nested level and get back to the previous one
     */
    void pop()
    {
      // fprintf(stderr, "stackable_ctx::pop() depth() was %ld\n", depth());
      _CCCL_ASSERT(nodes.size() > 0, "Calling pop while no context was pushed");

      auto& current_node = nodes.back();

      // Automatically pop data if needed
      for (auto& d_impl : current_node.pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_before_finalize();
      }

      // Ensure everything is finished in the context
      current_node.ctx.finalize();

      if (display_graph_stats)
      {
        executable_graph_cache_stat* stat = current_node.ctx.graph_get_cache_stat();
        _CCCL_ASSERT(stat, "");

        const auto& loc = current_node.callsite;
        stats_map[loc][::std::make_pair(stat->nnodes, stat->nedges)] += *stat;
      }

      // To create prereqs that depend on this finalize() stage, we get the
      // stream used in this context, and insert events in it.
      cudaStream_t stream         = current_node.support_stream;
      event_list finalize_prereqs = nodes[depth() - 1].ctx.stream_to_event_list(stream, "finalized");

      for (auto& d_impl : current_node.pushed_data)
      {
        _CCCL_ASSERT(d_impl, "invalid value");
        d_impl->pop_after_finalize(finalize_prereqs);
      }

      // Destroy the resources used in the wrapper allocator (if any)
      if (current_node.alloc_adapters)
      {
        current_node.alloc_adapters->clear();
      }

      // Destroy the current node
      nodes.pop_back();
    }

    /**
     * @brief Get the nesting depth
     */
    size_t depth() const
    {
      return nodes.size() - 1;
    }

    ctx_node& get_node(size_t level)
    {
      _CCCL_ASSERT(level < nodes.size(), "invalid value");
      return nodes[level];
    }

    const ctx_node& get_node(size_t level) const
    {
      _CCCL_ASSERT(level < nodes.size(), "invalid value");
      return nodes[level];
    }

    /**
     * @brief Returns a reference to the context for a specific ctx node
     */
    auto& get_ctx(size_t level)
    {
      _CCCL_ASSERT(level < nodes.size(), "invalid value");
      return nodes[level].ctx;
    }

    /**
     * @brief Returns a const reference to the context for a specific ctx node
     */
    const auto& get_ctx(size_t level) const
    {
      _CCCL_ASSERT(level < nodes.size(), "invalid value");
      return nodes[level].ctx;
    }

  private:
    void print_cache_stats_summary() const
    {
      if (!display_graph_stats || stats_map.size() == 0)
      {
        return;
      }

      fprintf(stderr, "Executable Graph Cache Statistics Summary\n");
      fprintf(stderr, "=========================================\n");

      for (const auto& [location, stat_map] : stats_map)
      {
        fprintf(stderr, "Call-Site: %s:%d (%s)\n", location.file_name(), location.line(), location.function_name());

        fprintf(stderr, "  Nodes  Edges  InstantiateCnt  UpdateCnt\n");
        fprintf(stderr, "  --------------------------------------\n");

        for (const auto& [key, stat] : stat_map)
        {
          fprintf(stderr, "  %5zu  %5zu  %13zu  %9zu\n", key.first, key.second, stat.instantiate_cnt, stat.update_cnt);
        }
        fprintf(stderr, "\n");
      }
    }

    // State for each node
    ::std::vector<ctx_node> nodes;

    // Handles to retain some asynchronous states, we maintain it separately
    // from nodes because we keep its entries even when we pop a level
    ::std::vector<async_resources_handle> async_handles;

    bool display_graph_stats;

    // Create a map indexed by source locations, the value stored are a map of stats indexed per (nnodes,nedges) pairs
    using stored_type_t =
      ::std::unordered_map<::std::pair<size_t, size_t>, executable_graph_cache_stat, hash<::std::pair<size_t, size_t>>>;
    ::std::unordered_map<_CUDA_VSTD::source_location,
                         stored_type_t,
                         reserved::source_location_hash,
                         reserved::source_location_equal>
      stats_map;
  };

  stackable_ctx()
      : pimpl(::std::make_shared<impl>())
  {}

  const auto& get_node(size_t level) const
  {
    return pimpl->get_node(level);
  }

  auto& get_node(size_t level)
  {
    return pimpl->get_node(level);
  }

  const auto& get_root_ctx() const
  {
    return pimpl->get_ctx(0);
  }

  auto& get_root_ctx()
  {
    return pimpl->get_ctx(0);
  }

  const auto& get_head_ctx() const
  {
    return pimpl->get_ctx(depth());
  }

  auto& get_head_ctx()
  {
    return pimpl->get_ctx(depth());
  }

  void push(const _CUDA_VSTD::source_location loc = _CUDA_VSTD::source_location::current())
  {
    pimpl->push(loc);
  }

  void pop()
  {
    pimpl->pop();
  }

  size_t depth() const
  {
    return pimpl->depth();
  }

  template <typename T>
  auto logical_data(shape_of<T> s)
  {
    // fprintf(stderr, "initialize from shape.\n");
    return stackable_logical_data(*this, depth(), true, get_root_ctx().logical_data(mv(s)), true);
  }

  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... more_sizes)
  {
    return stackable_logical_data(
      *this, depth(), true, get_root_ctx().template logical_data<T>(elements, more_sizes...), true);
  }

  template <typename T>
  auto logical_data_no_export(shape_of<T> s)
  {
    // fprintf(stderr, "initialize from shape.\n");
    return stackable_logical_data(*this, depth(), true, get_head_ctx().logical_data(mv(s)), false);
  }

  template <typename T, typename... Sizes>
  auto logical_data_no_export(size_t elements, Sizes... more_sizes)
  {
    return stackable_logical_data(
      *this, depth(), true, get_head_ctx().template logical_data<T>(elements, more_sizes...), false);
  }

  stackable_logical_data<void_interface> logical_token();

  template <typename... Pack>
  auto logical_data(Pack&&... pack)
  {
    // fprintf(stderr, "initialize from value.\n");
    return stackable_logical_data(
      *this, depth(), false, get_head_ctx().logical_data(::std::forward<Pack>(pack)...), true);
  }

  // Helper function to process a single argument
  template <typename T1>
  void process_argument(const T1&, ::std::vector<::std::pair<int, access_mode>>&) const
  {
    // Do nothing for non-stackable_task_dep
  }

  template <typename T1, typename reduce_op, bool initialize>
  void process_argument(const stackable_task_dep<T1, reduce_op, initialize>& dep,
                        ::std::vector<::std::pair<int, access_mode>>& combined_accesses) const
  {
    // If the stackable logical data appears in multiple deps of the same
    // task, we need to combine access modes to push the data automatically
    // with an appropriate mode.
    int id  = dep.get_d().get_unique_id();
    auto it = ::std::find_if(
      combined_accesses.begin(), combined_accesses.end(), [id](const ::std::pair<int, access_mode>& entry) {
        return entry.first == id;
      });
    _CCCL_ASSERT(it != combined_accesses.end(), "internal error");
    access_mode combined_m = it->second;

    // If the logical data was not at the appropriate level, we may
    // automatically push it. In this case, we need to update the logical data
    // referred stored in the task_dep object.
    bool need_update = dep.get_d().validate_access(*this, combined_m);
    //    if (need_update)
    {
      // The underlying dep eg. obtained when calling l.read() was resulting in
      // an task_dep_untyped where the logical data was the one at the "top of
      // the stack". Since a push method was done automatically, the logical
      // data that needs to be used was incorrect, and we update it.
      dep.underlying_dep().update_data(dep.get_d().get_ld());
    }
  }

  template <typename T1>
  void combine_argument_access_modes(const T1&, ::std::vector<::std::pair<int, access_mode>>&) const
  {
    // nothing
  }

  template <typename T1, typename reduce_op, bool initialize>
  void combine_argument_access_modes(const stackable_task_dep<T1, reduce_op, initialize>& dep,
                                     ::std::vector<::std::pair<int, access_mode>>& combined_accesses) const
  {
    int id        = dep.get_d().get_unique_id();
    access_mode m = dep.get_access_mode();

    // Now we go through the vector and update it if necessary, or add to the vector
    auto it = ::std::find_if(
      combined_accesses.begin(), combined_accesses.end(), [id](const ::std::pair<int, access_mode>& entry) {
        return entry.first == id;
      });

    if (it != combined_accesses.end())
    {
      it->second = it->second | m; // Merge access modes
    }
    else
    {
      combined_accesses.emplace_back(id, m); // Insert if not found
    }
  }

  // Process each argument passed to this construct to ensure we are accessing
  // data at the proper depth, or automatically push them at the appropriate
  // depth if necessary. If this happens, we may update the task_dep objects to
  // reflect the actual logical data that needs to be used.
  template <typename... Pack>
  void process_pack(const Pack&... pack) const
  {
    // This is a map of logical data, and the combined access modes
    ::std::vector<::std::pair<int, access_mode>> combined_accesses;
    (combine_argument_access_modes(pack, combined_accesses), ...);

    // fprintf(stderr, "process_pack begin.\n");
    (process_argument(pack, combined_accesses), ...);
    //  fprintf(stderr, "process_pack end.\n");
  }

  template <typename... Pack>
  auto task(Pack&&... pack)
  {
    process_pack(pack...);
    return get_head_ctx().task(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  template <typename... Pack>
  auto parallel_for(Pack&&... pack)
  {
    process_pack(pack...);
    return get_head_ctx().parallel_for(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel(Pack&&... pack)
  {
    process_pack(pack...);
    return get_head_ctx().cuda_kernel(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  template <typename... Pack>
  auto cuda_kernel_chain(Pack&&... pack)
  {
    process_pack(pack...);
    return get_head_ctx().cuda_kernel_chain(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }
#endif

  template <typename... Pack>
  auto host_launch(Pack&&... pack)
  {
    process_pack(pack...);
    return get_head_ctx().host_launch(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  auto task_fence()
  {
    return get_head_ctx().task_fence();
  }

  template <typename... Pack>
  void push_affinity(Pack&&... pack) const
  {
    process_pack(pack...);
    get_head_ctx().push_affinity(reserved::to_task_dep(::std::forward<Pack>(pack))...);
  }

  void pop_affinity() const
  {
    get_head_ctx().pop_affinity();
  }

  auto& async_resources() const
  {
    return get_head_ctx().async_resources();
  }

  auto dot_section(::std::string symbol) const
  {
    return get_head_ctx().dot_section(mv(symbol));
  }

  size_t task_count() const
  {
    return get_head_ctx().task_count();
  }

  void finalize()
  {
    // There must be only one level left
    _CCCL_ASSERT(depth() == 0, "All nested contexts must have been popped");

    get_head_ctx().finalize();
  }

public:
  ::std::shared_ptr<impl> pimpl;
};

template <typename T>
class stackable_logical_data
{
  class impl
  {
    // We separate the impl and the state so that if the stackable logical data
    // gets destroyed before the stackable context gets destroyed, we can save
    // this state in the context, in a vector of type-erased retained states
    class state : public stackable_logical_data_impl_state_base
    {
    public:
      state(stackable_ctx _sctx)
          : sctx(mv(_sctx))
      {}

      // This method is called when we pop the stackable_logical_data before we
      // have called finalize() on the nested context. This destroys the
      // logical data that was created in the nested context.
      virtual void pop_before_finalize() const override
      {
        size_t sctx_depth = sctx.depth();
        size_t data_depth = depth();
        _CCCL_ASSERT(sctx_depth > 0, "internal error");

        auto& dnode = data_nodes.back();

        _CCCL_ASSERT(data_depth == sctx_depth - 1 || data_depth == sctx_depth, "internal error");

        // Maybe the logical data was already destroyed if the stackable
        // logical data was destroyed before ctx pop, and that the data state
        // was retained. In this case, the data_node object was already
        // destroyed and there is no need to do it here.
        if (data_depth == sctx_depth)
        {
          _CCCL_ASSERT(!dnode.frozen_ld.has_value(), "internal error");
          data_nodes.pop_back();
        }

        // Unfreezing data will create a dependency which we need to track to
        // display a dependency between the context and its parent in DOT.
        // We do this operation before finalizing the context.
        auto& parent_dnode = get_data_node(sctx_depth - 1);
        _CCCL_ASSERT(parent_dnode.frozen_ld.has_value(), "internal error");
        parent_dnode.get_cnt--;
        sctx.get_node(sctx_depth)
          .ctx.get_dot()
          ->ctx_add_output_id(parent_dnode.frozen_ld.value().unfreeze_fake_task_id());
      }

      // Unfreeze the logical data after the context has been finalized.
      virtual void pop_after_finalize(const event_list& finalize_prereqs) const override
      {
        nvtx_range r("stackable_logical_data::pop_after_finalize");

        auto& dnode = data_nodes.back();
        _CCCL_ASSERT(dnode.frozen_ld.has_value(), "internal error");

        // Only unfreeze if there are no other subcontext still using it
        if (dnode.get_cnt == 0)
        {
          dnode.frozen_ld.value().unfreeze(finalize_prereqs);
          dnode.frozen_ld.reset();
        }
      }

      size_t depth() const
      {
        return data_nodes.size() - 1 + offset_depth;
      }

      int get_unique_id() const
      {
        _CCCL_ASSERT(data_nodes.size() > 0, "cannot get the id of an uninitialized stackable_logical_data");
        // Get the ID of the base logical data
        return data_nodes[0].ld.get_unique_id();
      }

      bool is_read_only() const
      {
        return read_only;
      }

      class data_node
      {
      public:
        data_node(logical_data<T> ld)
            : ld(mv(ld))
        {}

        // Get the access mode used to freeze data
        access_mode get_frozen_mode() const
        {
          _CCCL_ASSERT(frozen_ld.has_value(), "cannot query frozen mode : not frozen");
          return frozen_ld.value().get_access_mode();
        }

        void set_symbol(const ::std::string& symbol)
        {
          ld.set_symbol(symbol);
        }

        logical_data<T> ld;

        // Frozen counterpart of ld (if any)
        ::std::optional<frozen_logical_data<T>> frozen_ld;

        // Once frozen, count number of calls to get
        mutable int get_cnt;
      };

      mutable ::std::vector<data_node> data_nodes;

      auto& get_data_node(size_t level)
      {
        return data_nodes[level - offset_depth];
      }

      const auto& get_data_node(size_t level) const
      {
        return data_nodes[level - offset_depth];
      }

      void set_symbol(::std::string symbol_)
      {
        symbol = mv(symbol_);
        data_nodes.back().set_symbol(symbol);
      }

      mutable stackable_ctx sctx;

      // If the logical data was created at a level that is not directly the
      // root of the context, we remember this offset
      size_t base_depth   = 0;
      size_t offset_depth = 0;

      ::std::string symbol;

      // Indicate whether it is allowed to access this logical data with
      // write() or rw() access
      bool read_only = false;
    };

  public:
    impl() = default;
    impl(stackable_ctx sctx_,
         size_t target_depth,
         bool ld_from_shape,
         logical_data<T> ld,
         bool can_export,
         data_place where = data_place::invalid)
        : sctx(mv(sctx_))
    {
      impl_state = ::std::make_shared<state>(sctx);

      impl_state->base_depth = target_depth;

      // TODO pass this offset directly rather than a boolean for more flexibility ? (e.g. creating a ctx of depth 2,
      // export at depth 1, not 0 ...)
      impl_state->offset_depth = can_export ? 0 : target_depth;

      // Save the logical data at the base level
      impl_state->data_nodes.emplace_back(ld);

      // If necessary, import data recursively until we reach the target depth
      for (size_t current_depth = impl_state->offset_depth + 1; current_depth <= target_depth; current_depth++)
      {
        push(ld_from_shape ? access_mode::write : access_mode::rw, where);
      }
    }

    // stackable_logical_data::impl::~impl
    ~impl()
    {
      if (!impl_state)
      {
        return;
      }

      size_t s_size = impl_state->data_nodes.size();

      // There is at least the logical data passed when we created the stackable_logical_data
      _CCCL_ASSERT(s_size > 0, "internal error");

      // This state will be retained until we pop the ctx enough times
      size_t offset_depth = impl_state->offset_depth;

      if (depth() != offset_depth)
      {
        // if there is a parent with a frozen data
        impl_state->data_nodes[depth() - 1].get_cnt--;
      }

      impl_state->data_nodes.pop_back();

      // If there were at least 2 logical data before pop_back, there remains
      // at least one logical data, so we need to retain the impl.
      if (s_size > 1)
      {
        if (offset_depth < sctx.depth())
        {
          // If that was an exportable data, offset_depth = 0, it can be deleted
          // when the first stacked level is popped (ie. when the CUDA graph has
          // been executed)
          sctx.get_node(offset_depth + 1).retain_data(mv(impl_state));
        }
      }

      impl_state = nullptr;
    }

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Define move constructor and move assignment operator
    impl(impl&&) noexcept            = default;
    impl& operator=(impl&&) noexcept = default;

    const auto& get_ld() const
    {
      return impl_state->data_nodes.back().ld;
    }
    auto& get_ld()
    {
      return impl_state->data_nodes.back().ld;
    }

    int get_unique_id() const
    {
      return impl_state->get_unique_id();
    }

    /* Push one level up (from the current data depth) */
    void push(access_mode m, data_place where = data_place::invalid) const
    {
      // fprintf(stderr, "stackable_logical_data::push() %s mode = %s\n", impl_state->symbol.c_str(),
      // access_mode_string(m));

      const size_t ctx_depth          = sctx.depth();
      const size_t current_data_depth = depth();

      // (current_data_depth + 1) is the data depth after pushing
      _CCCL_ASSERT(ctx_depth >= current_data_depth + 1, "Invalid depth");

      auto& from_node = sctx.get_node(current_data_depth);
      auto& to_node   = sctx.get_node(current_data_depth + 1);

      context& from_ctx = from_node.ctx;
      context& to_ctx   = to_node.ctx;

      auto& from_data_node = impl_state->data_nodes[current_data_depth];

      if (where == data_place::invalid)
      {
        // use the default place
        where = from_ctx.default_exec_place().affine_data_place();
      }

      _CCCL_ASSERT(where != data_place::invalid, "Invalid data place");

      // Freeze the logical data of the node
      // TODO refcnt here, check compatibility in freeze mode
      _CCCL_ASSERT(!from_data_node.frozen_ld.has_value(), "data node already frozen");
      auto frozen_ld         = from_ctx.freeze(from_data_node.ld, m, mv(where), false /* not a user freeze */);
      from_data_node.get_cnt = 0;

      // Save in the optional value (we keep using frozen_ld variable to avoid
      // using frozen_ld.value() in the rest of this function)
      from_data_node.frozen_ld = frozen_ld;

      // FAKE IMPORT : use the stream needed to support the (graph) ctx
      cudaStream_t stream = to_node.support_stream;

      T inst  = frozen_ld.get(where, stream);
      auto ld = to_ctx.logical_data(inst, where);
      from_data_node.get_cnt++;

      if (!impl_state->symbol.empty())
      {
        // TODO reflect base/offset depths
        ld.set_symbol(impl_state->symbol + "." + ::std::to_string(current_data_depth + 1 - impl_state->base_depth));
      }

      // The inner context depends on the freeze operation, so we ensure DOT
      // displays these dependencies from the freeze in the parent context to
      // the child context itself.
      to_ctx.get_dot()->ctx_add_input_id(frozen_ld.freeze_fake_task_id());

      // Keep track of data that were pushed in this context.  This will be
      // used to pop data automatically when nested contexts are popped.
      to_node.track_pushed_data(impl_state);

      // Save the logical data created to keep track of the data instance
      // obtained with the get method of the frozen_logical_data object.
      impl_state->data_nodes.emplace_back(mv(ld));
    }

    /* Pop one level down */
    void pop_before_finalize() const
    {
      impl_state->pop_before_finalize();
    }

    void pop_after_finalize(const event_list& finalize_prereqs) const
    {
      impl_state->pop_after_finalize(finalize_prereqs);
    }

    size_t depth() const
    {
      return impl_state->depth();
    }

    void set_symbol(::std::string symbol)
    {
      impl_state->set_symbol(mv(symbol));
    }

    auto get_symbol() const
    {
      return impl_state->symbol;
    }

    // The write-back mechanism here refers to the write-back of the data at the bottom of the stack (user visible)
    void set_write_back(bool flag)
    {
      _CCCL_ASSERT(impl_state->s.size() > 0, "invalid value");
      impl_state->s[0].set_write_back(flag);
    }

    void set_read_only(bool flag = true)
    {
      impl_state->read_only = flag;
    }

    bool is_read_only() const
    {
      return impl_state->is_read_only();
    }

    // TODO why making sctx private or why do we need to expose this at all ?
    auto& get_sctx()
    {
      return sctx;
    }

    size_t get_offset_depth() const
    {
      return impl_state->offset_depth;
    }

    // Get the access mode used to freeze at depth d
    // TODO move to state
    access_mode get_frozen_mode(size_t d) const
    {
      return impl_state->get_data_node(d).get_frozen_mode();
    }

  private:
    // TODO replace this mutable by a const ...
    mutable stackable_ctx sctx; // in which stackable context was this created ?

    ::std::shared_ptr<state> impl_state;
  };

public:
  stackable_logical_data() = default;

  /* Create a logical data in the stackable ctx : in order to make it possible
   * to export all the way down to the root context, we create the logical data
   * in the root, and import them. */
  template <typename... Args>
  stackable_logical_data(
    stackable_ctx sctx, bool ld_from_shape, size_t target_depth, logical_data<T> ld, bool can_export)
      : pimpl(::std::make_shared<impl>(sctx, ld_from_shape, target_depth, mv(ld), can_export))
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data>, "");
    static_assert(::std::is_move_assignable_v<stackable_logical_data>, "");
  }

  const auto& get_ld() const
  {
    return pimpl->get_ld();
  }
  auto& get_ld()
  {
    return pimpl->get_ld();
  }

  int get_unique_id() const
  {
    return pimpl->get_unique_id();
  }

  size_t depth() const
  {
    return pimpl->depth();
  }

  void push(access_mode m, data_place where = data_place::invalid) const
  {
    pimpl->push(m, mv(where));
  }

  // Helpers
  template <typename... Pack>
  auto read(Pack&&... pack) const
  {
    using U = rw_type_of<T>;
    return stackable_task_dep<U, ::std::monostate, false>(*this, get_ld().read(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto write(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld().write(::std::forward<Pack>(pack)...));
  }

  template <typename... Pack>
  auto rw(Pack&&... pack)
  {
    return stackable_task_dep(*this, get_ld().rw(::std::forward<Pack>(pack)...));
  }

  auto shape() const
  {
    return get_ld().shape();
  }

  auto& set_symbol(::std::string symbol)
  {
    pimpl->set_symbol(mv(symbol));
    return *this;
  }

  void set_write_back(bool flag)
  {
    pimpl->set_write_back(flag);
  }

  void set_read_only(bool flag = true)
  {
    pimpl->set_read_only(flag);
  }

  bool is_read_only() const
  {
    return pimpl->is_read_only();
  }

  auto get_symbol() const
  {
    return pimpl->get_symbol();
  }

  auto get_impl()
  {
    return pimpl;
  }

  // Test whether it is valid to access this stackable_logical_data with a
  // given access mode, and automatically push data at the proper context depth
  // if necessary.
  //
  // Returns true if the task_dep needs an update
  bool validate_access(const stackable_ctx& sctx, access_mode m) const
  {
    // fprintf(stderr,
    //         "Calling validate_access() on stackable_logical_data %p - symbol=%s - requested mode %s\n",
    //         pimpl.get(),
    //         get_symbol().c_str(),
    //         access_mode_string(m));

    size_t offset_depth = pimpl->get_offset_depth();

    // Are we trying to access a logical data that cannot be "exported" at this level ?
    _CCCL_ASSERT(sctx.depth() >= offset_depth, "Invalid value");

    // Fast path : do nothing if we are not in a stacked ctx
    if (sctx.depth() == offset_depth)
    {
      // fprintf(stderr, "validate : FAST PATH %s : sctx.depth() == offset_depth == %ld\n", get_symbol().c_str(),
      // offset_depth);
      return false;
    }

    // TODO assert sctx == this->sctx

    _CCCL_ASSERT(m == access_mode::read || m == access_mode::rw || m == access_mode::write,
                 "Unsupported access mode in nested context");

    _CCCL_ASSERT(!is_read_only() || m == access_mode::read, "read only data cannot be modified");

    // If the stackable logical data is already at the appropriate depth, we
    // simply need to ensure we don't make an illegal access (eg. writing a
    // read only variable)
    size_t d = depth();

    // Data depth cannot be higher than context depth
    _CCCL_ASSERT(sctx.depth() >= d, "Invalid value");

    if (sctx.depth() == d)
    {
      _CCCL_ASSERT(d > 0, "data must have already been pushed");
      _CCCL_ASSERT(access_mode_is_compatible(pimpl->get_frozen_mode(d - 1), m), "Invalid access mode");
      // fprintf(stderr, "VALIDATED ACCESS ON stackable_logical_data %p - symbol=%s\n", pimpl.get(),
      //  get_symbol().c_str());
      return false;
    }

    // If we reach this point, this means we need to automatically push data

    // The access mode will be very conservative for these implicit accesses
    access_mode push_mode =
      is_read_only() ? access_mode::read : ((m == access_mode::write) ? access_mode::write : access_mode::rw);

    while (sctx.depth() > d)
    {
      // fprintf(stderr,
      //         "AUTOMATIC PUSH of data %p (symbol=%s) with mode %s at depth %ld\n",
      //         pimpl.get(),
      //         get_symbol().c_str(),
      //         access_mode_string(push_mode),
      //         d);

      pimpl->push(push_mode, data_place::current_device());
      d++;
    }

    return true;
  }

private:
  ::std::shared_ptr<impl> pimpl;
};

inline stackable_logical_data<void_interface> stackable_ctx::logical_token()
{
  return stackable_logical_data<void_interface>(*this, depth(), true, get_root_ctx().logical_token(), true);
}

/**
 * @brief Task dependency for a stackable logical data
 */
template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep
{
public:
  stackable_task_dep(stackable_logical_data<T> _d, task_dep<T, reduce_op, initialize> _dep)
      : d(mv(_d))
      , dep(mv(_dep))
  {}

  const stackable_logical_data<T>& get_d() const
  {
    return d;
  }

  // Provide non-const and const accessors.
  auto& underlying_dep()
  {
    // `*this` is a stackable_task_dep. Upcast to the base subobject.
    return dep;
  }

  const auto& underlying_dep() const
  {
    return dep;
  }

  access_mode get_access_mode() const
  {
    return dep.get_access_mode();
  }

private:
  stackable_logical_data<T> d;
  data_place dplace;
  mutable task_dep<T, reduce_op, initialize> dep;
};

#ifdef UNITTESTED_FILE
#  ifdef __CUDACC__
namespace reserved
{

template <typename T>
static __global__ void kernel_set(T* addr, T val)
{
  printf("SETTING ADDR %p at %d\n", addr, val);
  *addr = val;
}

template <typename T>
static __global__ void kernel_add(T* addr, T val)
{
  *addr += val;
}

template <typename T>
static __global__ void kernel_check_value(T* addr, T val)
{
  printf("CHECK %d EXPECTED %d\n", *addr, val);
  if (*addr != val)
  {
    ::cuda::std::terminate();
  }
}

} // namespace reserved

UNITTEST("stackable task_fence")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.push();
  lA.push(access_mode::write, data_place::current_device());
  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  ctx.task_fence();
  ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 44);
  };
  ctx.pop();
  ctx.finalize();
};

UNITTEST("stackable host_launch")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.push();
  lA.push(access_mode::write, data_place::current_device());
  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  // ctx.host_launch(lA.read())->*[](auto a){ _CCCL_ASSERT(a(0) == 42, "invalid value"); };
  ctx.pop();
  ctx.finalize();
};

#  endif // __CUDACC__
#endif // UNITTESTED_FILE

} // end namespace cuda::experimental::stf
