//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Stackable context and logical data to nest contexts

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <algorithm>
#include <atomic>
#include <iostream>
#include <shared_mutex>
#include <stack>
#include <thread>

#include "cuda/experimental/__stf/allocators/adapters.cuh"
#include "cuda/experimental/__stf/internal/task.cuh"
#include "cuda/experimental/__stf/stackable/conditional_nodes.cuh"
#include "cuda/experimental/__stf/stackable/stackable_ctx_impl.cuh"
#include "cuda/experimental/__stf/stackable/stackable_task_dep.cuh"
#include "cuda/experimental/__stf/utility/hash.cuh"
#include "cuda/experimental/__stf/utility/source_location.cuh"
#include "cuda/experimental/stf.cuh"

//! \brief Stackable Context Design Overview
//!
//! The stackable context allows nesting CUDA STF contexts to create hierarchical task graphs.
//! This enables complex workflows where tasks can be organized in a tree-like structure.
//!
//! Key concepts:
//! - **Context Stack**: Nested contexts form a stack where each level can have its own task graph
//! - **Data Movement**: Logical data can be imported ("pushed") between context levels automatically
//!
//! Usage pattern:
//! ```
//! stackable_ctx sctx;
//! auto data = sctx.logical_data(...);
//!
//! sctx.push();  // Enter nested context
//! data.push(access_mode::rw);  // Import data into nested context
//! // ... work with data in nested context ...
//! sctx.pop();   // Exit nested context, execute graph
//! ```
//!
//! By default, a task using a logical data in a nested context will
//! automatically issue a `push` in a `rw` mode. Advanced users can still push in
//! read-only mode to ensure data can be used concurrently from different nested
//! contexts.

namespace cuda::experimental::stf
{
/**
 * @brief Check whether a granted access mode permits a requested access mode.
 *
 * When data is imported into a nested context with a given mode (granted),
 * this checks if a subsequent operation requesting a different mode is valid.
 * For example, data imported as read-only cannot be written.
 */
inline bool access_mode_is_mutating(access_mode m)
{
  return m == access_mode::write || m == access_mode::rw || m == access_mode::reduce
      || m == access_mode::reduce_no_init;
}

inline bool access_mode_permits(access_mode granted, access_mode requested)
{
  return !access_mode_is_mutating(requested) || access_mode_is_mutating(granted);
}

//! Logical data type used in a stackable_ctx context type.
//!
//! It should behaves exactly like a logical_data with additional API to import
//! it across nested contexts.
template <typename T>
class stackable_logical_data
{
public:
  /// @brief Alias for `T` - matches logical_data<T> convention
  using element_type = T;

private:
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
      void pop_before_finalize(int ctx_offset) override
      {
        // either data node is valid at the same offset (if the logical data
        // wasn't destroyed), or its parent must be valid.
        // int parent_offset = node_tree.parent[ctx_offset];
        int parent_offset = sctx.get_parent_offset(ctx_offset);
        _CCCL_ASSERT(parent_offset != -1, "");

        // check the parent data is valid
        _CCCL_ASSERT(data_nodes[parent_offset].has_value(), "");

        auto& parent_dnode = data_nodes[parent_offset].value();

        // Maybe the logical data was already destroyed if the stackable
        // logical data was destroyed before ctx pop, and that the data state
        // was retained. In this case, the data_node object was already
        // cleared and there is no need to do it here.
        if (data_nodes[ctx_offset].has_value())
        {
          access_mode frozen_mode = get_frozen_mode(parent_offset);
          if ((frozen_mode == access_mode::rw) && (data_nodes[ctx_offset].value().effective_mode == access_mode::read))
          {
            static ::std::atomic<int> warning_count{0};
            if (warning_count.fetch_add(1, ::std::memory_order_relaxed) < 100)
            {
              fprintf(stderr,
                      "Warning : no write access on data pushed with a write mode (may be suboptimal) (symbol %s)\n",
                      symbol.empty() ? "(no symbol)" : symbol.c_str());
              if (warning_count.load(::std::memory_order_relaxed) == 100)
              {
                fprintf(stderr, "Warning: Suppressing further write mode warnings (reached limit of 100)\n");
              }
            }
          }

          _CCCL_ASSERT(!data_nodes[ctx_offset].value().frozen_ld.has_value(), "internal error");
          data_nodes[ctx_offset].reset();
        }

        // Unfreezing data will create a dependency which we need to track to
        // display a dependency between the context and its parent in DOT.
        // We do this operation before finalizing the context.
        _CCCL_ASSERT(parent_dnode.frozen_ld.has_value(), "internal error");
        parent_dnode.get_cnt--;

        sctx.get_node(ctx_offset)
          ->ctx.get_dot()
          ->ctx_add_output_id(parent_dnode.frozen_ld.value().unfreeze_fake_task_id());
      }

      // Unfreeze the logical data after the context has been finalized.
      void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) override
      {
        nvtx_range r("stackable_logical_data::pop_after_finalize");

        _CCCL_ASSERT(data_nodes[parent_offset].has_value(), "");
        auto& dnode = data_nodes[parent_offset].value();

        _CCCL_ASSERT(dnode.frozen_ld.has_value(), "internal error");

        dnode.unfreeze_prereqs.merge(finalize_prereqs);

        // Only unfreeze if there are no other subcontext still using it
        _CCCL_ASSERT(dnode.get_cnt >= 0, "get_cnt should never be negative");
        if (dnode.get_cnt == 0)
        {
          dnode.frozen_ld.value().unfreeze(dnode.unfreeze_prereqs);
          dnode.frozen_ld.reset();
        }
      }

      int get_unique_id() const
      {
        _CCCL_ASSERT(data_root_offset != -1, "");
        _CCCL_ASSERT(data_nodes[data_root_offset].has_value(), "");

        // Get the ID of the base logical data
        return get_data_node(data_root_offset).ld.get_unique_id();
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

        event_list unfreeze_prereqs;

        // Once frozen, count number of calls to get
        int get_cnt = 0;

        // Keep track of actual data accesses, so that we can detect if we
        // eventually did not need to freeze a data in write mode, for example.
        access_mode effective_mode = access_mode::none;
      };

      auto& get_data_node(int offset)
      {
        _CCCL_ASSERT(offset != -1, "invalid value");
        _CCCL_ASSERT(data_nodes[offset].has_value(), "invalid value");
        return data_nodes[offset].value();
      }

      const auto& get_data_node(int offset) const
      {
        _CCCL_ASSERT(offset != -1, "invalid value");
        _CCCL_ASSERT(data_nodes[offset].has_value(), "invalid value");
        return data_nodes[offset].value();
      }

      template <typename Func>
      void traverse_data_nodes(Func&& func) const
      {
        ::std::stack<int> node_stack;
        node_stack.push(data_root_offset);

        while (!node_stack.empty())
        {
          int offset = node_stack.top();
          node_stack.pop();

          // Call the provided function on the current node
          func(offset);

          // Push children to stack (reverse order to maintain left-to-right order)
          const auto& children = sctx.get_children_offsets(offset);
          for (auto it = children.rbegin(); it != children.rend(); ++it)
          {
            if (was_imported(*it))
            {
              node_stack.push(*it);
            }
          }
        }
      }

      void set_symbol(::std::string symbol_)
      {
        auto ctx_lock = sctx.acquire_exclusive_lock();
        symbol        = mv(symbol_);
        traverse_data_nodes([this](int offset) {
          get_data_node(offset).ld.set_symbol(this->symbol);
        });
      }

      int get_data_root_offset() const
      {
        return data_root_offset;
      }

      bool was_imported(int offset) const
      {
        _CCCL_ASSERT(offset != -1, "");

        if (offset >= int(data_nodes.size()))
        {
          return false;
        }

        return data_nodes[offset].has_value();
      }

      // Mark how a construct accessed this data node, so that we may detect if
      // we were overly cautious when freezing data in RW mode. This would
      // prevent concurrent accesses from different contexts, and may require
      // to push data in read only, if appropriate.
      void mark_access(int offset, access_mode m)
      {
        _CCCL_ASSERT(offset != -1 && data_nodes[offset].has_value(), "Failed to find data node for mark_access");
        data_nodes[offset].value().effective_mode |= m;
      }

      bool is_frozen(int offset) const
      {
        _CCCL_ASSERT(data_nodes[offset].has_value(), "");
        return data_nodes[offset].value().frozen_ld.has_value();
      }

      access_mode get_frozen_mode(int offset) const
      {
        _CCCL_ASSERT(is_frozen(offset), "");
        return data_nodes[offset].value().frozen_ld.value().get_access_mode();
      }

      ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
      {
        return ::std::shared_lock<::std::shared_mutex>(mutex);
      }

      ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
      {
        return ::std::unique_lock<::std::shared_mutex>(mutex);
      }

      friend impl;

    private:
      // Centralized method to grow data nodes to a certain size
      void grow_data_nodes(int target_size,
                           size_t factor_numerator   = node_hierarchy::default_growth_numerator,
                           size_t factor_denominator = node_hierarchy::default_growth_denominator)
      {
        if (target_size < int(data_nodes.size()))
        {
          return; // Already large enough
        }

        size_t new_size =
          ::std::max(static_cast<size_t>(target_size), data_nodes.size() * factor_numerator / factor_denominator);
        data_nodes.resize(new_size);
      }

      stackable_ctx sctx;

      ::std::vector<::std::optional<data_node>> data_nodes;

      int data_root_offset;

      ::std::string symbol;

      // Indicate whether it is allowed to access this logical data with
      // write() or rw() access
      bool read_only = false;

      mutable ::std::shared_mutex mutex;
    };

  public:
    impl() = default;
    impl(stackable_ctx sctx_,
         int target_offset,
         bool ld_from_shape,
         logical_data<T> ld,
         bool can_export,
         data_place where = data_place::invalid())
        : sctx(mv(sctx_))
    {
      impl_state = ::std::make_shared<state>(sctx);

      // TODO pass this offset directly rather than a boolean for more flexibility ? (e.g. creating a ctx of depth 2,
      // export at depth 1, not 0 ...)
      int data_root_offset         = can_export ? sctx.get_root_offset() : target_offset;
      impl_state->data_root_offset = data_root_offset;

      // Save the logical data at the base level
      if (data_root_offset >= int(impl_state->data_nodes.size()))
      {
        impl_state->grow_data_nodes(data_root_offset + 1);
      }
      _CCCL_ASSERT(!impl_state->data_nodes[data_root_offset].has_value(), "");

      impl_state->data_nodes[data_root_offset].emplace(ld);

      // If necessary, import data recursively until we reach the target depth.
      // We first find the path from the target to the root and we push along this path
      if (target_offset != data_root_offset)
      {
        // Recurse from the target offset to the root offset
        ::std::stack<int> path;
        int current = target_offset;
        while (current != data_root_offset)
        {
          path.push(current);

          current = sctx.get_parent_offset(current);
          _CCCL_ASSERT(current != -1, "");
        }

        // push along the path
        while (!path.empty())
        {
          int offset = path.top();
          push(offset, ld_from_shape ? access_mode::write : access_mode::rw, where);

          path.pop();
        }
      }
    }

    // stackable_logical_data::impl::~impl
    ~impl()
    {
      // Nothing to clean up if moved or default-constructed
      if (!impl_state)
      {
        return;
      }

      auto ctx_lock = sctx.acquire_exclusive_lock(); // to protect retained_data

      int data_root_offset = impl_state->get_data_root_offset();

      _CCCL_ASSERT(impl_state->data_nodes[data_root_offset].has_value(), "");

      // Do NOT destroy data_nodes here: the root data_node may still hold a
      // frozen_ld that children depend on for pop_after_finalize to unfreeze.
      // Ownership is transferred to children via retain_data; the shared_ptr
      // ref-count ensures all data_nodes (including the root) are destroyed
      // only after every child context has finished its pop sequence.
      const auto& root_children = sctx.get_children_offsets(data_root_offset);
      for (auto c : root_children)
      {
        if (impl_state->was_imported(c))
        {
          // Transfer shared_ptr ownership to child context's retained_data vector.
          // The child context will keep impl_state alive until it's popped.
          sctx.get_node(c)->retain_data(impl_state);
        }
      }

      impl_state = nullptr;
    }

    // Delete copy constructor and copy assignment operator
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    // Non movable
    impl(impl&&) noexcept            = delete;
    impl& operator=(impl&&) noexcept = delete;

    ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
    {
      return impl_state->acquire_shared_lock();
    }

    ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
    {
      return impl_state->acquire_exclusive_lock();
    }

    const auto& get_ld(int offset) const
    {
      _CCCL_ASSERT(offset != -1 && impl_state->was_imported(offset), "Failed to find imported data");
      return impl_state->get_data_node(offset).ld;
    }

    auto& get_ld(int offset)
    {
      _CCCL_ASSERT(offset != -1 && impl_state->was_imported(offset), "Failed to find imported data");
      return impl_state->get_data_node(offset).ld;
    }

    int get_data_root_offset() const
    {
      return impl_state->get_data_root_offset();
    }

    int get_unique_id() const
    {
      return impl_state->get_unique_id();
    }

    /* Import data into the ctx at this offset */
    void push(int ctx_offset, access_mode m, data_place where = data_place::invalid())
    {
      int parent_offset = sctx.get_parent_offset(ctx_offset);

      // Base case: if this is root context (no parent), data should already exist
      if (parent_offset == -1)
      {
        _CCCL_ASSERT(impl_state->was_imported(ctx_offset), "Root context must already have data");
        return;
      }

      if (ctx_offset >= int(impl_state->data_nodes.size()))
      {
        impl_state->grow_data_nodes(ctx_offset + 1);
      }

      if (impl_state->data_nodes[ctx_offset].has_value())
      {
        // Data already exists - ensure existing mode is compatible, no upgrades possible
        auto& existing_node = impl_state->data_nodes[ctx_offset].value();
        _CCCL_ASSERT(access_mode_permits(existing_node.effective_mode, m), "Cannot change existing access mode");
        return;
      }

      // Ancestor compatibility is now handled by recursive push calls

      // Check if parent has data, if not push with max required mode
      access_mode max_required_parent_mode =
        (m == access_mode::write || m == access_mode::reduce) ? access_mode::write : access_mode::rw;

      if (!impl_state->data_nodes[parent_offset].has_value())
      {
        // RECURSIVE CALL: Ensure parent has data first
        push(parent_offset, max_required_parent_mode, where);
      }

      _CCCL_ASSERT(impl_state->data_nodes[parent_offset].has_value(), "parent data should be available here");

      auto& to_node   = sctx.get_node(ctx_offset);
      auto& from_node = sctx.get_node(parent_offset);

      context& to_ctx   = to_node->ctx;
      context& from_ctx = from_node->ctx;

      auto& from_data_node = impl_state->data_nodes[parent_offset].value();

      if (where.is_invalid())
      {
        // use the default place
        where = from_ctx.default_exec_place().affine_data_place();
      }

      _CCCL_ASSERT(!where.is_invalid(), "Invalid data place");

      // Freeze the logical data of the parent node if it wasn't yet
      if (!from_data_node.frozen_ld.has_value())
      {
        from_data_node.frozen_ld = from_ctx.freeze(from_data_node.ld, m, where, false /* not a user freeze */);
        from_data_node.get_cnt   = 0;
      }
      else
      {
        // Data is already frozen - this is an IMPLICIT push
        // For implicit pushes, use conservative mode: write/rw unless specifically read-only
        access_mode existing_frozen_mode = from_data_node.frozen_ld.value().get_access_mode();

        // Check if we need to upgrade the frozen mode for implicit push
        if (!access_mode_permits(existing_frozen_mode, m))
        {
          fprintf(stderr,
                  "Error: Incompatible access mode - existing frozen mode %s conflicts with requested mode %s\n",
                  access_mode_string(existing_frozen_mode),
                  access_mode_string(m));
          abort();
        }
      }

      _CCCL_ASSERT(from_data_node.frozen_ld.has_value(), "");
      auto& frozen_ld = from_data_node.frozen_ld.value();

      // FAKE IMPORT : use the stream needed to support the (graph) ctx
      cudaStream_t stream = to_node->support_stream;

      // Ensure there is a copy of the data in the data place, we keep a
      // reference count of each context using this frozen data so that we only
      // unfreeze once possible.
      ::std::pair<T, event_list> get_res = frozen_ld.get(where);
      auto ld                            = to_ctx.logical_data(get_res.first, where);
      from_data_node.get_cnt++;

      to_node->ctx_prereqs.merge(mv(get_res.second));

      if (!impl_state->symbol.empty())
      {
        ld.set_symbol(impl_state->symbol);
      }

      // The inner context depends on the freeze operation, so we ensure DOT
      // displays these dependencies from the freeze in the parent context to
      // the child context itself.
      to_ctx.get_dot()->ctx_add_input_id(frozen_ld.freeze_fake_task_id());

      // Keep track of data that were pushed in this context.  This will be
      // used to pop data automatically when nested contexts are popped.
      to_node->track_pushed_data(impl_state);

      // Create the node at the requested offset based on the logical data we
      // have just created from the data frozen in its parent.
      impl_state->data_nodes[ctx_offset].emplace(mv(ld));
    }

    /* Pop one level down */
    void pop_before_finalize(int ctx_offset)
    {
      impl_state->pop_before_finalize(ctx_offset);
    }

    void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs)
    {
      impl_state->pop_after_finalize(parent_offset, finalize_prereqs);
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
      _CCCL_ASSERT(!impl_state->data_nodes.empty(), "invalid value");
      impl_state->data_nodes[impl_state->data_root_offset].value().ld.set_write_back(flag);
    }

    // Indicate that this logical data will only be used in a read-only mode
    // now. Implicit data push will therefore be done in a read-only mode,
    // which allows concurrent read accesses from  different contexts (ie. from
    // multiple CUDA graphs)
    void set_read_only(bool flag = true)
    {
      impl_state->read_only = flag;
    }

    bool is_read_only() const
    {
      return impl_state->is_read_only();
    }

    void mark_access(int offset, access_mode m)
    {
      return impl_state->mark_access(offset, m);
    }

    bool was_imported(int offset) const
    {
      return impl_state->was_imported(offset);
    }

    bool is_frozen(int offset) const
    {
      return impl_state->is_frozen(offset);
    }

    // Get the access mode used to freeze at a given offset
    access_mode get_frozen_mode(int offset) const
    {
      return impl_state->get_frozen_mode(offset);
    }

    int get_ctx_head_offset() const
    {
      return sctx.get_head_offset();
    }

  private:
    template <typename, typename, bool>
    friend class stackable_task_dep;

    stackable_ctx sctx;

    ::std::shared_ptr<state> impl_state;
  };

public:
  stackable_logical_data() = default;

  /* Create a logical data in the stackable ctx : in order to make it possible
   * to export all the way down to the root context, we create the logical data
   * in the root, and import them. */
  stackable_logical_data(stackable_ctx sctx, int ctx_offset, bool ld_from_shape, logical_data<T> ld, bool can_export)
      : pimpl(::std::make_shared<impl>(sctx, ctx_offset, ld_from_shape, mv(ld), can_export))
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data>);
    static_assert(::std::is_move_assignable_v<stackable_logical_data>);
  }

  int get_data_root_offset() const
  {
    return pimpl->get_data_root_offset();
  }

  const auto& get_ld(int offset) const
  {
    return pimpl->get_ld(offset);
  }

  auto& get_ld(int offset)
  {
    return pimpl->get_ld(offset);
  }

  int get_unique_id() const
  {
    return pimpl->get_unique_id();
  }

  void push(int ctx_offset, access_mode m, data_place where = data_place::invalid()) const
  {
    pimpl->push(ctx_offset, m, mv(where));
  }

  void push(access_mode m, data_place where = data_place::invalid()) const
  {
    int ctx_offset = pimpl->get_ctx_head_offset();
    pimpl->push(ctx_offset, m, mv(where));
  }

  // Helpers — return lazy stackable_task_dep descriptors.
  auto read() const
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, access_mode::read);
  }

  auto read(data_place dp) const
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, access_mode::read, mv(dp));
  }

  auto write()
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, access_mode::write);
  }

  auto write(data_place dp)
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, access_mode::write, mv(dp));
  }

  auto rw()
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, access_mode::rw);
  }

  auto rw(data_place dp)
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, access_mode::rw, mv(dp));
  }

  template <typename Op>
  auto reduce(Op)
  {
    return stackable_task_dep<T, Op, true>(*this, access_mode::reduce);
  }

  template <typename Op>
  auto reduce(Op, data_place dp)
  {
    return stackable_task_dep<T, Op, true>(*this, access_mode::reduce, mv(dp));
  }

  template <typename Op>
  auto reduce(Op, no_init)
  {
    return stackable_task_dep<T, Op, false>(*this, access_mode::reduce_no_init);
  }

  template <typename Op>
  auto reduce(Op, no_init, data_place dp)
  {
    return stackable_task_dep<T, Op, false>(*this, access_mode::reduce_no_init, mv(dp));
  }

  auto dep_with_mode(access_mode m)
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, m);
  }

  auto dep_with_mode(access_mode m, data_place dp)
  {
    return stackable_task_dep<T, ::std::monostate, false>(*this, m, mv(dp));
  }

  auto shape() const
  {
    return get_ld(get_data_root_offset()).shape();
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

  auto get_impl() const
  {
    return pimpl;
  }

  // Test whether it is valid to access this stackable_logical_data with a
  // given access mode, and automatically push data at the proper context depth
  // if necessary.
  //
  // Returns true if the task_dep needs an update
  bool validate_access(int ctx_offset, const stackable_ctx& sctx, access_mode m) const
  {
    // Grab the lock of the data, note that we are already holding the context lock in read mode
    auto lock = pimpl->acquire_exclusive_lock();

    _CCCL_ASSERT(m != access_mode::none && m != access_mode::relaxed, "Unsupported access mode in nested context");

    _CCCL_ASSERT(!is_read_only() || m == access_mode::read, "read only data cannot be modified");

    if (get_data_root_offset() == ctx_offset)
    {
      return false;
    }

    // If the stackable logical data is already available at the appropriate depth, we
    // simply need to ensure we don't make an illegal access (eg. writing a
    // read only variable)
    if (pimpl->was_imported(ctx_offset))
    {
      // Always validate access modes - find the actual parent with frozen data
      int parent_offset = sctx.get_parent_offset(ctx_offset);
      if (parent_offset != -1 && pimpl->is_frozen(parent_offset))
      {
        access_mode parent_frozen_mode = pimpl->get_frozen_mode(parent_offset);
        // Check access mode compatibility with parent's frozen mode
        if (!access_mode_permits(parent_frozen_mode, m))
        {
          fprintf(stderr,
                  "Error: Invalid access mode transition - parent frozen with %s, requesting %s\n",
                  access_mode_string(parent_frozen_mode),
                  access_mode_string(m));
          abort();
        }
      }

      // To potentially detect if we were overly cautious when importing data
      // in rw mode, we record how we access it in this construct
      pimpl->mark_access(ctx_offset, m);

      // We need to update because the current ctx offset was not the base offset
      return true;
    }

    // If we reach this point, this means we need to automatically push data

    // The access mode will be very conservative for these implicit accesses
    access_mode push_mode =
      is_read_only() ? access_mode::read
                     : ((m == access_mode::write || m == access_mode::reduce) ? access_mode::write : access_mode::rw);

    // Recurse from the target offset to its first imported(pushed) parent
    ::std::stack<int> path;
    int current = ctx_offset;
    while (!pimpl->was_imported(current))
    {
      path.push(current);

      current = sctx.get_parent_offset(current);
      _CCCL_ASSERT(current != -1, "");
    }

    // use the affine data place for the current default place
    auto where = sctx.get_ctx(ctx_offset).default_exec_place().affine_data_place();

    // push along the path
    while (!path.empty())
    {
      int offset = path.top();
      pimpl->push(offset, push_mode, where);
      path.pop();
    }

    // To potentially detect if we were overly cautious when importing data
    // in rw mode, we record how we access it in this construct
    pimpl->mark_access(ctx_offset, m);

    return true;
  }

private:
  ::std::shared_ptr<impl> pimpl;
};

inline stackable_logical_data<void_interface> stackable_ctx::token()
{
  int head = pimpl->get_head_offset();
  return stackable_logical_data<void_interface>(*this, head, true, get_root_ctx().token(), true);
}

//! Task dependency specification for a stackable logical data.
//!
//! This is a lazy descriptor that stores the data, access mode, and data place.
//! The actual task_dep is only created when resolve() is called with the correct
//! context offset, ensuring it always references the right logical data level.
template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep
{
public:
  using data_t      = T;
  using dep_type    = T;
  using op_and_init = ::std::pair<reduce_op, ::std::bool_constant<initialize>>;
  using op_type     = reduce_op;
  enum : bool
  {
    does_work = !::std::is_same_v<reduce_op, ::std::monostate>
  };

  stackable_task_dep(stackable_logical_data<T> _d, access_mode _mode, data_place _dplace = data_place::affine())
      : d(mv(_d))
      , mode(_mode)
      , dplace(mv(_dplace))
  {}

  // Implicit conversion to task_dep for interop with non-stackable contexts.
  // Resolves at the current thread's head offset.
  operator task_dep<T, reduce_op, initialize>() const
  {
    auto& sctx = d.get_impl()->sctx;
    int offset = sctx.get_head_offset();
    d.validate_access(offset, sctx, mode);
    return resolve(offset);
  }

  const stackable_logical_data<T>& get_d() const
  {
    return d;
  }

  access_mode get_access_mode() const
  {
    return mode;
  }

  const data_place& get_dplace() const
  {
    return dplace;
  }

  // Create a concrete task_dep from the logical data at the given context offset.
  // Callers must ensure validate_access has already been called for this offset.
  task_dep<T, reduce_op, initialize> resolve(int context_offset) const
  {
    auto& context_ld = d.get_ld(context_offset);
    return task_dep<T, reduce_op, initialize>(context_ld, mode, dplace);
  }

private:
  stackable_logical_data<T> d;
  access_mode mode;
  data_place dplace;
};

//! \brief Handle to a cudaGraphExec_t produced by stackable_ctx::pop_prologue().
//!
//! Returned by stackable_ctx::pop_prologue(), this handle exposes the
//! executable graph instantiated from the nested context and lets the user
//! launch it repeatedly before committing the pop via pop_epilogue().
//!
//! All public methods assert that the handle is still valid: copying is
//! allowed but a copy does not prolong validity. Every outstanding handle
//! (original or copy) becomes invalid the moment pop_epilogue() runs.
//!
//! Usage:
//! \code
//! stackable_ctx ctx;
//! ctx.push();
//! // ... build graph ...
//! auto h = ctx.pop_prologue();
//! for (int i = 0; i < N; ++i) h.launch();
//! ctx.pop_epilogue();
//! \endcode
class launchable_graph_handle
{
public:
  launchable_graph_handle() = default;

  //! \brief Underlying executable graph. Aborts if the handle is invalid.
  //!
  //! On the first call, lazily instantiates the graph (cache lookup +
  //! cudaGraphInstantiate if not cached) and orders `stream()` behind the
  //! nested context's freeze/get events (dep A) so that callers can legally
  //! drive the graph manually via `cudaGraphLaunch(exec(), stream())`.
  //! Subsequent calls (and any calls after a prior `launch()`) skip both
  //! steps.
  //!
  //! Note: unlike `graph()`, calling `exec()` *does* force instantiation.
  //! Callers that only want to embed the graph as a child node in another
  //! graph should use `graph()` and avoid `exec()` entirely.
  cudaGraphExec_t exec() const
  {
    validate_("exec");
    auto shared_exec = ctx_.pimpl->prepare_handle_for_exec(node_offset_);
    _CCCL_ASSERT(shared_exec, "invalid executable graph");
    return *shared_exec;
  }

  //! \brief Support stream the graph was prepared against. Aborts if invalid.
  //!
  //! Does NOT perform any stream synchronization - it is purely observational.
  //! If you plan to drive the graph manually, call `exec()` first (it will
  //! lazily sync), or use `launch()` which does both in one step.
  cudaStream_t stream() const
  {
    validate_("stream");
    return support_stream_;
  }

  //! \brief Launch the graph once on the support stream.
  //!
  //! On the first call, waits for the nested context's freeze/get events
  //! (dep A). Subsequent calls skip the sync and issue the launch directly.
  void launch()
  {
    validate_("launch");
    ctx_.pimpl->launch_prepared_graph(node_offset_, support_stream_);
  }

  //! \brief Underlying (non-executable) CUDA graph topology.
  //!
  //! Returns the finalized `cudaGraph_t` built from the nested context, for
  //! callers who want to embed it as a child node into another graph via
  //! `cudaGraphAddChildGraphNode` instead of launching the pre-instantiated
  //! executable graph returned by `exec()`.
  //!
  //! The graph is owned by the stackable_ctx and stays valid only until
  //! `pop_epilogue()` is called. If you need it to outlive the epilogue,
  //! clone it with `cudaGraphClone`.
  //!
  //! Unlike `exec()`, this accessor does NOT trigger `cudaGraphInstantiate`.
  //! Callers that only need the graph topology (embedding as a child graph,
  //! inspecting nodes, etc.) pay no instantiation cost as long as neither
  //! `exec()` nor `launch()` is ever called on this handle.
  //!
  //! On the first call, `graph()` does perform the same lazy dep-A sync as
  //! `exec()` / `launch()` (it orders `stream()` behind the nested
  //! context's freeze events). This makes `handle.stream()` a valid event
  //! source for ordering an outer launch stream: record an event on
  //! `stream()` and wait on it from your launch stream before launching
  //! the outer graph that embeds this child.
  cudaGraph_t graph() const
  {
    validate_("graph");
    ctx_.pimpl->prepare_handle_for_graph(node_offset_);
    return graph_;
  }

  //! \brief True iff the owning stackable_ctx has not yet run pop_epilogue().
  bool valid() const
  {
    return !token_.expired();
  }

private:
  friend class stackable_ctx;

  void validate_(const char* op) const
  {
    if (token_.expired())
    {
      fprintf(stderr, "Error: launchable_graph_handle::%s() called after pop_epilogue()\n", op);
      abort();
    }
  }

  // Kept alive by the stackable_ctx::impl while the pop is pending. We hold
  // a weak_ptr so that pop_epilogue() can invalidate every outstanding
  // handle simply by dropping its shared_ptr.
  ::std::weak_ptr<int> token_;
  // Copy of the ctx - shares its impl via shared_ptr. Keeps impl alive even
  // if the original stackable_ctx goes out of scope between prologue and
  // epilogue.
  stackable_ctx ctx_;
  int node_offset_             = -1;
  cudaGraph_t graph_           = nullptr;
  cudaStream_t support_stream_ = nullptr;
};

inline launchable_graph_handle stackable_ctx::pop_prologue()
{
  auto r = pimpl->pop_prologue_impl();

  launchable_graph_handle h;
  h.token_          = r.token;
  h.ctx_            = *this;
  h.node_offset_    = r.node_offset;
  h.graph_          = r.graph;
  h.support_stream_ = r.support_stream;
  return h;
}

//! \brief RAII wrapper for automatic push/pop management (lock_guard style)
//!
//! This class provides automatic scope management for nested contexts,
//! following the same semantics as std::lock_guard.
//! The constructor calls push() and the destructor calls pop().
//!
//! Usage (direct constructor style):
//! \code
//! {
//!   stackable_ctx::graph_scope_guard scope{ctx};
//!   // nested context operations...
//! }
//! \endcode
//!
//! Usage (factory method style):
//! \code
//! {
//!   auto scope = ctx.graph_scope();
//!   // nested context operations...
//! }
//! \endcode
class stackable_ctx::graph_scope_guard
{
public:
  using context_type = stackable_ctx;

  explicit graph_scope_guard(stackable_ctx& ctx,
                             const ::cuda::std::source_location& loc = ::cuda::std::source_location::current())
      : ctx_(ctx)
  {
    ctx_.push(loc);
  }

  ~graph_scope_guard()
  {
    ctx_.pop();
  }

  graph_scope_guard(const graph_scope_guard&)            = delete;
  graph_scope_guard& operator=(const graph_scope_guard&) = delete;
  graph_scope_guard(graph_scope_guard&&)                 = delete;
  graph_scope_guard& operator=(graph_scope_guard&&)      = delete;

private:
  stackable_ctx& ctx_;
};

inline stackable_ctx::graph_scope_guard stackable_ctx::graph_scope(const ::cuda::std::source_location& loc)
{
  return graph_scope_guard(*this, loc);
}

//! \brief RAII wrapper for a re-launchable pop scope.
//!
//! On construction, calls ctx.push() and then ctx.pop_prologue(). The caller
//! uses launch() as many times as desired; the destructor (or an explicit
//! release()) runs ctx.pop_epilogue() and makes the handle invalid.
//!
//! Usage:
//! \code
//! stackable_ctx ctx;
//! {
//!   stackable_ctx::launchable_graph_scope scope{ctx};
//!   // ... build graph contents as if inside ctx.push()/ctx.pop() ...
//!   for (int i = 0; i < N; ++i) scope.launch();
//! } // pop_epilogue() runs automatically here
//! \endcode
class stackable_ctx::launchable_graph_scope
{
public:
  using context_type = stackable_ctx;

  explicit launchable_graph_scope(stackable_ctx& ctx,
                                  const ::cuda::std::source_location& loc = ::cuda::std::source_location::current())
      : ctx_(ctx)
  {
    ctx_.push(loc);
    // The graph body is built by the caller after construction. The handle
    // is populated on release() / destructor by calling pop_prologue() only
    // once we're ready - but the plan names this class a "scope" in the
    // same spirit as graph_scope_guard, so we run pop_prologue() lazily on
    // the first call to launch(). This matches the natural call order
    // (push -> build -> launch -> pop_epilogue).
  }

  ~launchable_graph_scope() noexcept
  {
    release();
  }

  launchable_graph_scope(const launchable_graph_scope&)            = delete;
  launchable_graph_scope& operator=(const launchable_graph_scope&) = delete;
  launchable_graph_scope(launchable_graph_scope&&)                 = delete;
  launchable_graph_scope& operator=(launchable_graph_scope&&)      = delete;

  //! \brief Launch the graph once. The first call triggers pop_prologue().
  void launch()
  {
    ensure_prepared_();
    handle_.launch();
  }

  //! \brief Expose the executable graph. Triggers pop_prologue() on demand.
  cudaGraphExec_t exec()
  {
    ensure_prepared_();
    return handle_.exec();
  }

  //! \brief Expose the support stream. Triggers pop_prologue() on demand.
  cudaStream_t stream()
  {
    ensure_prepared_();
    return handle_.stream();
  }

  //! \brief Expose the underlying CUDA graph (for embedding into another
  //! graph via cudaGraphAddChildGraphNode). Triggers pop_prologue() on
  //! demand.
  cudaGraph_t graph()
  {
    ensure_prepared_();
    return handle_.graph();
  }

  //! \brief Explicitly commit the pop (idempotent).
  //!
  //! Runs pop_prologue() (if not already done) and pop_epilogue(). After
  //! release(), further calls to launch()/exec()/stream()/graph() are invalid.
  void release() noexcept
  {
    if (released_)
    {
      return;
    }
    released_ = true;

    if (!prepared_)
    {
      // No one ever called launch()/exec()/stream()/graph(): we still ran push()
      // in the constructor, so we must match it with a prologue+epilogue
      // pair to tear the node down cleanly. finalize_after_launch handles
      // the no-launch case correctly.
      handle_   = ctx_.pop_prologue();
      prepared_ = true;
    }
    ctx_.pop_epilogue();
  }

private:
  void ensure_prepared_()
  {
    if (!prepared_)
    {
      handle_   = ctx_.pop_prologue();
      prepared_ = true;
    }
  }

  stackable_ctx& ctx_;
  launchable_graph_handle handle_;
  bool prepared_ = false;
  bool released_ = false;
};

//! \brief Shared-ownership, storable handle for a re-launchable popped graph.
//!
//! Returned by `stackable_ctx::pop_prologue_shared()`. Copies share a single
//! underlying state; when the last copy is destroyed (or the last copy is
//! explicitly `reset()`), `pop_epilogue()` runs on the originating context.
//!
//! Unlike `launchable_graph_scope`, this type is copyable and movable, so it
//! can be stored as a data member, placed in containers, or returned from a
//! factory -- making it a natural fit for a classic "build once, launch many
//! times, release later" cache.
//!
//! Example -- build once, store as a data member, launch repeatedly:
//! \code
//! class SimEngine {
//! public:
//!   void build(size_t N, double alpha) {
//!     ctx_.push();
//!     auto lx = ctx_.logical_data(shape_of<slice<double>>(N));
//!     ctx_.parallel_for(lx.shape(), lx.write())->*[] __device__(size_t i, auto x) {
//!       x(i) = 1.0;
//!     };
//!     ctx_.parallel_for(lx.shape(), lx.rw())->*[=] __device__(size_t i, auto x) {
//!       x(i) += alpha;
//!     };
//!     step_graph_ = ctx_.pop_prologue_shared();
//!   }
//!
//!   void step() { step_graph_.launch(); }
//!
//! private:
//!   stackable_ctx ctx_;
//!   stackable_ctx::launchable_graph step_graph_;
//! };
//! \endcode
//!
//! Example -- cache keyed by shape:
//! \code
//! std::unordered_map<size_t, stackable_ctx::launchable_graph> cache;
//! if (auto it = cache.find(N); it == cache.end()) {
//!   ctx.push();
//!   // ... build graph ...
//!   cache.emplace(N, ctx.pop_prologue_shared());
//! }
//! cache[N].launch();
//! \endcode
//!
//! Example -- embed into a larger graph instead of launching:
//! \code
//! auto sub = ctx.pop_prologue_shared();
//! cudaGraph_t outer = nullptr;
//! cudaGraphCreate(&outer, 0);
//! cudaGraphNode_t child{};
//! // graph() does NOT instantiate; sub.stream() is a valid event source.
//! cudaGraphAddChildGraphNode(&child, outer, nullptr, 0, sub.graph());
//! \endcode
class stackable_ctx::launchable_graph
{
public:
  launchable_graph()                                       = default;
  launchable_graph(const launchable_graph&)                = default;
  launchable_graph(launchable_graph&&) noexcept            = default;
  launchable_graph& operator=(const launchable_graph&)     = default;
  launchable_graph& operator=(launchable_graph&&) noexcept = default;
  ~launchable_graph()                                      = default;

  //! \brief Launch the graph once on its support stream.
  void launch()
  {
    check_("launch");
    state_->handle.launch();
  }

  //! \brief Underlying executable graph. Triggers lazy instantiation + dep-A
  //! sync on the first call (same contract as `launchable_graph_handle::exec()`).
  cudaGraphExec_t exec() const
  {
    check_("exec");
    return state_->handle.exec();
  }

  //! \brief Support stream the graph was prepared against. Purely observational.
  cudaStream_t stream() const
  {
    check_("stream");
    return state_->handle.stream();
  }

  //! \brief Underlying cudaGraph_t topology (for embedding as a child graph).
  //! Triggers lazy dep-A sync but does NOT call `cudaGraphInstantiate`.
  cudaGraph_t graph() const
  {
    check_("graph");
    return state_->handle.graph();
  }

  //! \brief True iff this copy still holds a shared reference and the
  //! underlying pop has not been epiloged (e.g. manually via
  //! `ctx.pop_epilogue()`).
  bool valid() const noexcept
  {
    return state_ && state_->handle.valid();
  }

  explicit operator bool() const noexcept
  {
    return valid();
  }

  //! \brief Number of live shared copies referring to the same graph. Debug
  //! introspection only. Returns 0 for a default-constructed / moved-from
  //! instance.
  long use_count() const noexcept
  {
    return state_ ? state_.use_count() : 0;
  }

  //! \brief Drop this shared reference eagerly. When this was the last copy,
  //! `pop_epilogue()` runs now instead of at destruction time. Idempotent.
  void reset() noexcept
  {
    state_.reset();
  }

private:
  friend class stackable_ctx;

  struct state
  {
    // Keep the ctx impl alive regardless of the originating stackable_ctx's
    // lifetime - a shared launchable_graph copy can outlive the variable that
    // created it.
    stackable_ctx ctx;
    launchable_graph_handle handle;

    state(stackable_ctx c, launchable_graph_handle h)
        : ctx(mv(c))
        , handle(mv(h))
    {}

    ~state()
    {
      // Guard against users who manually called pop_epilogue() on the ctx
      // while shared copies were still alive: the handle's token is expired
      // and pop_epilogue has already run, so we must not call it again.
      if (handle.valid())
      {
        ctx.pop_epilogue();
      }
    }

    state(const state&)            = delete;
    state& operator=(const state&) = delete;
  };

  void check_(const char* op) const
  {
    if (!state_)
    {
      fprintf(stderr, "Error: launchable_graph::%s() called on an empty handle\n", op);
      abort();
    }
  }

  ::std::shared_ptr<state> state_;
};

inline stackable_ctx::launchable_graph stackable_ctx::pop_prologue_shared()
{
  // pop_prologue() already performs the validity / state checks.
  auto handle = pop_prologue();

  launchable_graph g;
  g.state_ = ::std::make_shared<launchable_graph::state>(*this, mv(handle));
  return g;
}

#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
//! \brief RAII guard for while loop contexts with conditional graphs
//!
//! This guard automatically creates a while loop context using push_while() on construction
//! and calls pop() on destruction. It provides access to the conditional handle created.
class stackable_ctx::while_graph_scope_guard
{
public:
  using context_type = stackable_ctx;

  explicit while_graph_scope_guard(
    stackable_ctx& ctx,
    unsigned int default_launch_value       = 0,
    unsigned int flags                      = cudaGraphCondAssignDefault,
    const ::cuda::std::source_location& loc = ::cuda::std::source_location::current())
      : ctx_(ctx)
  {
    ctx_.push_while(&conditional_handle_, default_launch_value, flags, loc);
  }

  ~while_graph_scope_guard()
  {
    ctx_.pop();
  }

  cudaGraphConditionalHandle cond_handle() const
  {
    return conditional_handle_;
  }

  template <typename... Deps>
  class condition_update_scope
  {
  public:
    condition_update_scope(stackable_ctx& ctx, cudaGraphConditionalHandle handle, Deps... deps)
        : ctx_(ctx)
        , handle_(handle)
        , tdeps(mv(deps)...)
    {}

    template <typename T>
    using data_t_of = typename T::data_t;

    template <typename CondFunc>
    void operator->*(CondFunc&& cond_func)
    {
      ::std::apply(
        [this](auto&&... deps) {
          return this->ctx_.cuda_kernel(deps...).set_symbol("condition_update");
        },
        tdeps)
          ->*[cond_func = mv(cond_func), h = handle_](data_t_of<Deps>... args) {
                return cuda_kernel_desc{
                  reserved::condition_update_kernel<CondFunc, data_t_of<Deps>...>, 1, 1, 0, h, cond_func, args...};
              };
    }

  private:
    stackable_ctx& ctx_;
    cudaGraphConditionalHandle handle_;
    ::std::tuple<::std::decay_t<Deps>...> tdeps;
  };

  //! \brief Helper for updating while loop condition using a device lambda
  //!
  //! The lambda should return true to continue the loop, false to exit.
  template <typename... Args>
  auto update_cond(Args&&... args)
  {
    return condition_update_scope(ctx_, cond_handle(), args...);
  }

  while_graph_scope_guard(const while_graph_scope_guard&)            = delete;
  while_graph_scope_guard& operator=(const while_graph_scope_guard&) = delete;
  while_graph_scope_guard(while_graph_scope_guard&&)                 = delete;
  while_graph_scope_guard& operator=(while_graph_scope_guard&&)      = delete;

private:
  stackable_ctx& ctx_;
  cudaGraphConditionalHandle conditional_handle_{};
};

inline stackable_ctx::while_graph_scope_guard stackable_ctx::while_graph_scope(
  unsigned int default_launch_value, unsigned int flags, const ::cuda::std::source_location& loc)
{
  return while_graph_scope_guard(*this, default_launch_value, flags, loc);
}

//! \brief RAII guard for repeat loops with automatic counter management
//!
//! This class provides RAII semantics for repeat loops, automatically managing
//! the counter and conditional logic. The loop body is executed in the scope
//! between construction and destruction.
//!
//! It encapsulates the common pattern of creating a counter-based while loop
//! in CUDA STF and automatically handles:
//! - Creating and initializing a loop counter
//! - Setting up the while graph scope
//! - Decrementing the counter and controlling the loop continuation
//!
//! Example usage:
//! ```cpp
//! stackable_ctx ctx;
//! auto data = ctx.logical_data(...);
//!
//! {
//!   auto guard = ctx.repeat_graph_scope(10);
//!   // Tasks added here will run 10 times
//!   ctx.parallel_for(data.shape(), data.rw())->*[] __device__(size_t i, auto d) {
//!     d(i) += 1.0;
//!   };
//! } // Automatic scope cleanup
//! ```
class repeat_graph_scope_guard
{
public:
  template <typename CounterType>
  static void init_counter_value(stackable_ctx& ctx, CounterType counter, size_t count)
  {
    ctx.parallel_for(box(1), counter.write())->*[count] __device__(size_t, auto counter) {
      *counter = count;
    };
  }

  template <typename CounterType>
  static void setup_condition_update(stackable_ctx::while_graph_scope_guard& while_guard, CounterType counter)
  {
    while_guard.update_cond(counter.read())->*[] __device__(auto counter) {
      (*counter)--;
      return (*counter > 0);
    };
  }

  explicit repeat_graph_scope_guard(
    stackable_ctx& ctx,
    size_t count,
    unsigned int default_launch_value = 1,
    unsigned int flags                = cudaGraphCondAssignDefault)
      : ctx_(ctx)
  {
    auto counter_shape = shape_of<scalar_view<size_t>>();
    counter_           = ctx_.logical_data(counter_shape);

    init_counter_value(ctx_, counter_, count);

    while_guard_.emplace(ctx_, default_launch_value, flags, ::cuda::std::source_location::current());

    setup_condition_update(*while_guard_, counter_);
  }

  repeat_graph_scope_guard(const repeat_graph_scope_guard&)            = delete;
  repeat_graph_scope_guard& operator=(const repeat_graph_scope_guard&) = delete;
  repeat_graph_scope_guard(repeat_graph_scope_guard&&)                 = delete;
  repeat_graph_scope_guard& operator=(repeat_graph_scope_guard&&)      = delete;

private:
  stackable_ctx& ctx_;
  // counter_ must be declared before while_guard_ so that while_guard_ is
  // destroyed first (reverse declaration order).  Its destructor calls
  // ctx_.pop() which may still reference counter data.
  stackable_logical_data<scalar_view<size_t>> counter_;
  ::std::optional<stackable_ctx::while_graph_scope_guard> while_guard_;
};

inline auto stackable_ctx::repeat_graph_scope(
  size_t count, unsigned int default_launch_value, unsigned int flags, const ::cuda::std::source_location& loc)
{
  (void) loc;
  return repeat_graph_scope_guard(*this, count, default_launch_value, flags);
}
#endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

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

UNITTEST("stackable host_launch")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.push();
  lA.push(access_mode::write, data_place::current_device());
  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };
  ctx.host_launch(lA.read())->*[](auto a) {
    _CCCL_ASSERT(a(0) == 42, "invalid value");
  };
  ctx.pop();
  ctx.finalize();
};

UNITTEST("graph_scope basic RAII")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test basic RAII behavior - scope automatically calls push/pop
  {
    auto scope = ctx.graph_scope(); // push() called here
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
    };
    // pop() called automatically when scope goes out of scope
  }

  ctx.finalize();
};

UNITTEST("graph_scope direct constructor style")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test direct constructor style (like std::lock_guard)
  {
    stackable_ctx::graph_scope_guard scope{ctx}; // Direct constructor, push() called here
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 24);
    };
    // pop() called automatically when scope goes out of scope
  }

  ctx.finalize();
};

UNITTEST("graph_scope nested scopes")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  auto lB = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test nested scopes work correctly using direct constructor style
  {
    stackable_ctx::graph_scope_guard outer_scope{ctx}; // outer push()
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 10);
    };

    {
      stackable_ctx::graph_scope_guard inner_scope{ctx}; // inner push() (nested)
      lB.push(access_mode::write, data_place::current_device());
      ctx.task(lB.write())->*[](cudaStream_t stream, auto b) {
        reserved::kernel_set<<<1, 1, 0, stream>>>(b.data_handle(), 20);
      };
      // inner pop() called automatically here
    }

    // Verify outer scope still works after inner scope closed
    ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 10);
    };
    // outer pop() called automatically here
  }

  ctx.finalize();
};

UNITTEST("graph_scope multiple sequential scopes")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  // Test multiple sequential scopes
  {
    auto scope1 = ctx.graph_scope();
    lA.push(access_mode::write, data_place::current_device());
    ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 100);
    };
  } // pop() for scope1

  {
    auto scope2 = ctx.graph_scope();
    ctx.task(lA.rw())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_add<<<1, 1, 0, stream>>>(a.data_handle(), 23);
    };
  } // pop() for scope2

  {
    auto scope3 = ctx.graph_scope();
    ctx.task(lA.read())->*[](cudaStream_t stream, auto a) {
      reserved::kernel_check_value<<<1, 1, 0, stream>>>(a.data_handle(), 123);
    };
  } // pop() for scope3

  ctx.finalize();
};

inline void test_graph_scope_with_tmp()
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    reserved::kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };

  {
    auto scope = ctx.graph_scope();

    // Create temporary data in nested context
    auto temp = ctx.logical_data(shape_of<slice<int>>(1024));

    ctx.parallel_for(lA.shape(), temp.write(), lA.read())->*[] __device__(size_t i, auto temp, auto a) {
      // Copy data and modify
      temp(i) = a(i) * 2;
    };

    ctx.parallel_for(lA.shape(), lA.write(), temp.read())->*[] __device__(size_t i, auto a, auto temp) {
      // Copy back
      a(i) = temp(i) + 1;
    };

    // temp automatically cleaned up when scope ends
  }

  ctx.finalize();
}

UNITTEST("graph_scope with temporary data")
{
  test_graph_scope_with_tmp();
};

inline void test_graph_scope()
{
  stackable_ctx ctx;

  // Initialize array similar to stackable2.cu
  int array[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    array[i] = 1 + i * i;
  }

  auto lA = ctx.logical_data(array).set_symbol("A");

  // Test iterative pattern: {tmp = a, a++; tmp*=2; a+=tmp} using graph_scope RAII
  for (size_t iter = 0; iter < 3; iter++) // Use fewer iterations for faster testing
  {
    auto graph = ctx.graph_scope(); // RAII: automatic push/pop

    auto tmp = ctx.logical_data(lA.shape()).set_symbol("tmp");

    ctx.parallel_for(tmp.shape(), tmp.write(), lA.read())->*[] __device__(size_t i, auto tmp, auto a) {
      tmp(i) = a(i);
    };

    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 1;
    };

    ctx.parallel_for(tmp.shape(), tmp.rw())->*[] __device__(size_t i, auto tmp) {
      tmp(i) *= 2;
    };

    ctx.parallel_for(lA.shape(), tmp.read(), lA.rw())->*[] __device__(size_t i, auto tmp, auto a) {
      a(i) += tmp(i);
    };

    // ctx.pop() is called automatically when 'graph' goes out of scope
  }

  ctx.finalize();
}

UNITTEST("graph_scope iterative pattern")
{
  test_graph_scope();
};

UNITTEST("stackable task on exec_place::host()")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.task(exec_place::host(), lA.write())->*[](cudaStream_t stream, auto) {
    cuda_safe_call(cudaStreamSynchronize(stream));
  };
  ctx.finalize();
};

UNITTEST("stackable task with set_symbol and set_exec_place")
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.task(lA.write()).set_symbol("task").set_exec_place(exec_place::host())->*[](cudaStream_t stream, auto) {
    cuda_safe_call(cudaStreamSynchronize(stream));
  };
  ctx.finalize();
};

inline void test_pop_prologue_repeated_launch()
{
  constexpr int N = 16;

  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  auto handle = ctx.pop_prologue();

  // prepare_launch() instantiates the graph but does NOT launch it, so the
  // graph actually runs exactly N times (once per handle.launch()).
  for (int k = 0; k < N; ++k)
  {
    handle.launch();
  }

  ctx.pop_epilogue();

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == N, "pop_prologue: relaunched graph did not accumulate correctly");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue + repeated launch accumulates N times")
{
  test_pop_prologue_repeated_launch();
};

inline void test_pop_prologue_manual_exec_launch()
{
  // Same setup as test_pop_prologue_repeated_launch, but drive the graph
  // manually via cudaGraphLaunch(handle.exec(), handle.stream()) as the
  // *first* launch. exec() is responsible for lazily performing the
  // prereq sync; stream() is purely observational.
  constexpr int N = 8;

  stackable_ctx ctx;

  int array[512];
  for (size_t i = 0; i < 512; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  auto handle = ctx.pop_prologue();

  // Manual launches only: never call handle.launch(). exec() must be the
  // one to lazily sync the support stream behind the freeze events.
  cudaGraphExec_t ex = handle.exec();
  cudaStream_t s     = handle.stream();
  for (int k = 0; k < N; ++k)
  {
    cuda_safe_call(cudaGraphLaunch(ex, s));
  }

  ctx.pop_epilogue();

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == N, "pop_prologue: manual cudaGraphLaunch did not accumulate correctly");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue + manual cudaGraphLaunch via exec()/stream()")
{
  test_pop_prologue_manual_exec_launch();
};

inline void test_pop_prologue_zero_launches()
{
  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 7;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  // Prologue then epilogue without a single launch() call: the graph is
  // instantiated, prereqs are synced, resources are released, and data is
  // unfrozen. Device memory is unchanged since no launch ran.
  auto handle = ctx.pop_prologue();
  (void) handle;
  ctx.pop_epilogue();

  // After pop_epilogue the context is usable again for the next pop.
  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 100;
  };
  ctx.pop();

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == 107, "post-epilogue ctx is not reusable");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue + pop_epilogue with zero launches")
{
  test_pop_prologue_zero_launches();
};

inline void test_pop_prologue_handle_invalidation()
{
  stackable_ctx ctx;

  int array[4];
  for (size_t i = 0; i < 4; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  auto handle = ctx.pop_prologue();
  _CCCL_ASSERT(handle.valid(), "handle must be valid between prologue and epilogue");
  handle.launch();
  _CCCL_ASSERT(handle.valid(), "handle must still be valid after launch");
  ctx.pop_epilogue();
  _CCCL_ASSERT(!handle.valid(), "handle must be invalidated by pop_epilogue");

  // Copy of the handle must share the same weak_ptr and therefore the same
  // invalidation.
  auto copy = handle;
  _CCCL_ASSERT(!copy.valid(), "copied handle must also be invalid after pop_epilogue");

  ctx.finalize();
}

UNITTEST("pop_prologue invalidates handle after pop_epilogue")
{
  test_pop_prologue_handle_invalidation();
};

inline void test_launchable_graph_scope_raii()
{
  constexpr int N = 5;

  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  {
    stackable_ctx::launchable_graph_scope scope{ctx};
    lA.push(access_mode::rw, data_place::current_device());
    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 2;
    };

    for (int k = 0; k < N; ++k)
    {
      scope.launch();
    }
    // pop_epilogue() runs here via the destructor
  }

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == 2 * N, "launchable_graph_scope: relaunched graph did not accumulate correctly");
    }
  };

  ctx.finalize();
}

UNITTEST("launchable_graph_scope RAII")
{
  test_launchable_graph_scope_raii();
};

inline void test_pop_prologue_shared_basic()
{
  constexpr int N = 5;

  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  {
    auto g = ctx.pop_prologue_shared();
    _CCCL_ASSERT(g.valid(), "fresh launchable_graph must be valid");
    _CCCL_ASSERT(g.use_count() == 1, "single owner at creation");
    for (int k = 0; k < N; ++k)
    {
      g.launch();
    }
    // g goes out of scope here -> pop_epilogue runs
  }

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == N, "pop_prologue_shared: relaunched graph did not accumulate correctly");
    }
  };

  // The ctx must be usable again after the shared owner released it.
  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 7;
  };
  ctx.pop();

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == N + 7, "stackable_ctx must be reusable after shared launchable_graph released");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue_shared last copy triggers pop_epilogue")
{
  test_pop_prologue_shared_basic();
};

inline void test_pop_prologue_shared_copies()
{
  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  auto g1 = ctx.pop_prologue_shared();
  auto g2 = g1; // shared copy
  _CCCL_ASSERT(g1.use_count() == 2, "two shared owners");
  _CCCL_ASSERT(g2.valid(), "copy must be valid");

  g1.launch();
  g2.launch();

  // Drop one copy; the other must still drive the graph.
  g1.reset();
  _CCCL_ASSERT(!g1.valid(), "reset copy becomes invalid");
  _CCCL_ASSERT(g2.valid(), "surviving copy remains valid");
  _CCCL_ASSERT(g2.use_count() == 1, "use_count drops to one after reset");
  g2.launch();

  // Final reset fires pop_epilogue exactly once.
  g2.reset();
  _CCCL_ASSERT(!g2.valid(), "last copy is invalid after reset");

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == 3, "pop_prologue_shared: expected 3 accumulations (2 before reset + 1 after)");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue_shared multiple copies share a single graph")
{
  test_pop_prologue_shared_copies();
};

inline void test_pop_prologue_shared_stored_in_container()
{
  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  // Build a graph inside a helper lambda and stash the resulting shared
  // handle in a std::vector that outlives the lambda scope - simulates the
  // "factory returns a shared graph to caller" pattern.
  ::std::vector<stackable_ctx::launchable_graph> cache;
  auto build_one = [&] {
    ctx.push();
    lA.push(access_mode::rw, data_place::current_device());
    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 1;
    };
    cache.push_back(ctx.pop_prologue_shared());
  };
  build_one();
  _CCCL_ASSERT(!cache.empty() && cache.front().valid(), "stored handle must remain valid after helper returns");

  for (int k = 0; k < 4; ++k)
  {
    cache.front().launch();
  }

  // Tear down via container clear; this drops the last shared copy and runs
  // pop_epilogue.
  cache.clear();

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == 4, "pop_prologue_shared: container-stored graph did not accumulate 4 launches");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue_shared storable across scopes / in containers")
{
  test_pop_prologue_shared_stored_in_container();
};

inline void test_pop_prologue_shared_manual_epilogue()
{
  // If the user manually calls ctx.pop_epilogue() after creating shared
  // copies, outstanding copies must become invalid and the shared state
  // destructor must skip the (already done) epilogue.
  stackable_ctx ctx;

  int array[4];
  for (size_t i = 0; i < 4; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  lA.push(access_mode::rw, data_place::current_device());
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 1;
  };

  auto g1 = ctx.pop_prologue_shared();
  auto g2 = g1;
  g1.launch();

  ctx.pop_epilogue();
  _CCCL_ASSERT(!g1.valid(), "shared copy must observe manual pop_epilogue");
  _CCCL_ASSERT(!g2.valid(), "second shared copy must observe manual pop_epilogue");
  // Letting g1, g2 fall out of scope must not double-epilogue.

  ctx.finalize();
}

UNITTEST("pop_prologue_shared tolerates manual pop_epilogue")
{
  test_pop_prologue_shared_manual_epilogue();
};

#    if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION)
inline void test_pop_prologue_with_while_graph_scope()
{
  constexpr int N              = 3; // re-launch the whole while-graph 3 times
  constexpr size_t inner_iters = 4; // each launch runs the body 4 times

  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; ++i)
  {
    array[i] = 0;
  }
  auto lA = ctx.logical_data(array).set_symbol("A");

  ctx.push();
  {
    auto rg = ctx.repeat_graph_scope(inner_iters);
    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 1;
    };
  }

  auto handle = ctx.pop_prologue();
  for (int k = 0; k < N; ++k)
  {
    handle.launch();
  }
  ctx.pop_epilogue();

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == int(N * inner_iters),
                   "pop_prologue + while-loop: re-launched graph body did not run the expected number of times");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue with while_graph_scope re-launched multiple times")
{
  test_pop_prologue_with_while_graph_scope();
};
#    endif // _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION)

#  endif // __CUDACC__
#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf
