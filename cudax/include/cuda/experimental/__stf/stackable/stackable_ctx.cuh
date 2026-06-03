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
#include <memory>
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
//! It behaves like a logical_data with additional API to import it across
//! nested contexts (push/pop between context levels).
template <typename T>
class stackable_logical_data
{
public:
  /// @brief Alias for `T` - matches logical_data<T> convention
  using element_type = T;

  stackable_logical_data() = default;

  stackable_logical_data(const stackable_logical_data&)                = default;
  stackable_logical_data& operator=(const stackable_logical_data&)     = default;
  stackable_logical_data(stackable_logical_data&&) noexcept            = default;
  stackable_logical_data& operator=(stackable_logical_data&&) noexcept = default;

  /* Create a logical data in the stackable ctx : in order to make it possible
   * to export all the way down to the root context, we create the logical data
   * in the root, and import them. */
  stackable_logical_data(stackable_ctx sctx, int ctx_offset, bool ld_from_shape, logical_data<T> ld, bool can_export)
      : owner_(::std::make_shared<owner>(::std::make_shared<state>(mv(sctx))))
  {
    static_assert(::std::is_move_constructible_v<stackable_logical_data>);
    static_assert(::std::is_move_assignable_v<stackable_logical_data>);

    // TODO pass this offset directly rather than a boolean for more flexibility ? (e.g. creating a ctx of depth 2,
    // export at depth 1, not 0 ...)
    const int data_root_offset        = can_export ? owner_->payload->sctx.get_root_offset() : ctx_offset;
    owner_->payload->data_root_offset = data_root_offset;

    if (static_cast<size_t>(data_root_offset) >= owner_->payload->data_nodes.size())
    {
      owner_->payload->grow_data_nodes(static_cast<size_t>(data_root_offset) + 1);
    }
    _CCCL_ASSERT(!owner_->payload->data_nodes[static_cast<size_t>(data_root_offset)].has_value(), "");

    owner_->payload->data_nodes[static_cast<size_t>(data_root_offset)].emplace(mv(ld));

    if (ctx_offset != data_root_offset)
    {
      ::std::stack<int> path;
      for (int current = ctx_offset; current != data_root_offset;
           current     = owner_->payload->sctx.get_parent_offset(current))
      {
        _CCCL_ASSERT(current >= 0, "");
        path.push(current);
      }

      while (!path.empty())
      {
        push_at(path.top(), ld_from_shape ? access_mode::write : access_mode::rw, data_place::invalid());
        path.pop();
      }
    }
  }

  int get_data_root_offset() const
  {
    return data().data_root_offset;
  }

  const auto& get_ld(int offset) const
  {
    _CCCL_ASSERT(offset >= 0 && data().was_imported(offset), "Failed to find imported data");
    return data().get_data_node(offset).ld;
  }

  auto& get_ld(int offset)
  {
    _CCCL_ASSERT(offset >= 0 && mut_data().was_imported(offset), "Failed to find imported data");
    return mut_data().get_data_node(offset).ld;
  }

  int get_unique_id() const
  {
    const int root = get_data_root_offset();
    _CCCL_ASSERT(root >= 0, "");
    _CCCL_ASSERT(data().data_nodes[static_cast<size_t>(root)].has_value(), "");
    return data().get_data_node(root).ld.get_unique_id();
  }

  void push(int ctx_offset, access_mode m, data_place where = data_place::invalid()) const
  {
    const_cast<stackable_logical_data*>(this)->push_at(ctx_offset, m, mv(where));
  }

  void push(access_mode m, data_place where = data_place::invalid()) const
  {
    push(data().sctx.get_head_offset(), m, mv(where));
  }

  stackable_ctx& sctx()
  {
    return mut_data().sctx;
  }

  const stackable_ctx& sctx() const
  {
    return data().sctx;
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
    auto& st      = mut_data();
    auto ctx_lock = st.sctx.acquire_exclusive_lock();
    st.symbol     = symbol;
    traverse_data_nodes(st, [symbol](state& s, int offset) {
      s.get_data_node(offset).ld.set_symbol(symbol);
    });
    return *this;
  }

  void set_write_back(bool flag)
  {
    const auto& st = data();
    _CCCL_ASSERT(st.data_root_offset >= 0, "invalid value");
    _CCCL_ASSERT(!st.data_nodes.empty(), "invalid value");
    mut_data().get_data_node(st.data_root_offset).ld.set_write_back(flag);
  }

  void set_read_only(bool flag = true)
  {
    mut_data().read_only = flag;
  }

  bool is_read_only() const
  {
    return data().read_only;
  }

  auto get_symbol() const
  {
    return data().symbol;
  }

  // Test whether it is valid to access this stackable_logical_data with a
  // given access mode, and automatically push data at the proper context depth
  // if necessary.
  //
  // Returns true if the task_dep needs an update
  bool validate_access(int ctx_offset, const stackable_ctx& sctx_ref, access_mode m) const
  {
    auto& self = *const_cast<stackable_logical_data*>(this);
    auto lock  = self.mut_data().acquire_exclusive_lock();

    _CCCL_ASSERT(m != access_mode::none && m != access_mode::relaxed, "Unsupported access mode in nested context");

    _CCCL_ASSERT(!is_read_only() || m == access_mode::read, "read only data cannot be modified");

    if (get_data_root_offset() == ctx_offset)
    {
      return false;
    }

    if (data().was_imported(ctx_offset))
    {
      const int parent_offset = sctx_ref.get_parent_offset(ctx_offset);
      if (parent_offset >= 0 && data().is_frozen(parent_offset))
      {
        const access_mode parent_frozen_mode = data().get_frozen_mode(parent_offset);
        if (!access_mode_permits(parent_frozen_mode, m))
        {
          fprintf(stderr,
                  "Error: Invalid access mode transition - parent frozen with %s, requesting %s\n",
                  access_mode_string(parent_frozen_mode),
                  access_mode_string(m));
          abort();
        }
      }

      self.mark_access_at(ctx_offset, m);
      return true;
    }

    const access_mode push_mode =
      is_read_only() ? access_mode::read
                     : ((m == access_mode::write || m == access_mode::reduce) ? access_mode::write : access_mode::rw);

    ::std::stack<int> path;
    for (int current = ctx_offset; !data().was_imported(current); current = sctx_ref.get_parent_offset(current))
    {
      _CCCL_ASSERT(current >= 0, "");
      path.push(current);
    }

    const auto where = sctx_ref.get_ctx(ctx_offset).default_exec_place().affine_data_place();

    while (!path.empty())
    {
      self.push_at(path.top(), push_mode, where);
      path.pop();
    }

    self.mark_access_at(ctx_offset, m);
    return true;
  }

private:
  template <typename, typename, bool>
  friend class stackable_task_dep;

  class state : public stackable_logical_data_impl_state_base
  {
  public:
    explicit state(stackable_ctx _sctx)
        : sctx(mv(_sctx))
    {}

    void pop_before_finalize(int ctx_offset) override
    {
      const int parent_offset = sctx.get_parent_offset(ctx_offset);
      _CCCL_ASSERT(parent_offset >= 0, "");

      _CCCL_ASSERT(data_nodes[static_cast<size_t>(parent_offset)].has_value(), "");
      auto& parent_dnode = data_nodes[static_cast<size_t>(parent_offset)].value();

      if (data_nodes[static_cast<size_t>(ctx_offset)].has_value())
      {
        const access_mode frozen_mode = get_frozen_mode(parent_offset);
        if ((frozen_mode == access_mode::rw)
            && (data_nodes[static_cast<size_t>(ctx_offset)].value().effective_mode == access_mode::read))
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

        _CCCL_ASSERT(!data_nodes[static_cast<size_t>(ctx_offset)].value().frozen_ld.has_value(), "internal error");
        data_nodes[static_cast<size_t>(ctx_offset)].reset();
      }

      _CCCL_ASSERT(parent_dnode.frozen_ld.has_value(), "internal error");
      parent_dnode.get_cnt--;

      sctx.get_node(ctx_offset)->ctx.get_dot()->ctx_add_output_id(parent_dnode.frozen_ld.value().unfreeze_fake_task_id());
    }

    void pop_after_finalize(int parent_offset, const event_list& finalize_prereqs) override
    {
      nvtx_range r("stackable_logical_data::pop_after_finalize");

      _CCCL_ASSERT(parent_offset >= 0, "");
      _CCCL_ASSERT(data_nodes[static_cast<size_t>(parent_offset)].has_value(), "");
      auto& dnode = data_nodes[static_cast<size_t>(parent_offset)].value();

      _CCCL_ASSERT(dnode.frozen_ld.has_value(), "internal error");

      dnode.unfreeze_prereqs.merge(finalize_prereqs);

      _CCCL_ASSERT(dnode.get_cnt >= 0, "get_cnt should never be negative");
      if (dnode.get_cnt == 0)
      {
        dnode.frozen_ld.value().unfreeze(dnode.unfreeze_prereqs);
        dnode.frozen_ld.reset();
      }
    }

    struct data_node
    {
      explicit data_node(logical_data<T> ld)
          : ld(mv(ld))
      {}

      logical_data<T> ld;
      ::std::optional<frozen_logical_data<T>> frozen_ld;
      event_list unfreeze_prereqs;
      int get_cnt                = 0;
      access_mode effective_mode = access_mode::none;
    };

    auto& get_data_node(int offset)
    {
      _CCCL_ASSERT(offset >= 0, "invalid value");
      const auto idx = static_cast<size_t>(offset);
      _CCCL_ASSERT(data_nodes[idx].has_value(), "invalid value");
      return data_nodes[idx].value();
    }

    const auto& get_data_node(int offset) const
    {
      _CCCL_ASSERT(offset >= 0, "invalid value");
      const auto idx = static_cast<size_t>(offset);
      _CCCL_ASSERT(data_nodes[idx].has_value(), "invalid value");
      return data_nodes[idx].value();
    }

    void grow_data_nodes(size_t target_size,
                         size_t factor_numerator   = node_hierarchy::default_growth_numerator,
                         size_t factor_denominator = node_hierarchy::default_growth_denominator)
    {
      if (target_size <= data_nodes.size())
      {
        return;
      }

      const size_t new_size = ::std::max(target_size, data_nodes.size() * factor_numerator / factor_denominator);
      data_nodes.resize(new_size);
    }

    bool was_imported(int offset) const
    {
      if (offset < 0)
      {
        return false;
      }

      const auto idx = static_cast<size_t>(offset);
      return idx < data_nodes.size() && data_nodes[idx].has_value();
    }

    void mark_access(int offset, access_mode m)
    {
      _CCCL_ASSERT(offset >= 0, "Failed to find data node for mark_access");
      _CCCL_ASSERT(data_nodes[static_cast<size_t>(offset)].has_value(), "Failed to find data node for mark_access");
      data_nodes[static_cast<size_t>(offset)].value().effective_mode |= m;
    }

    bool is_frozen(int offset) const
    {
      _CCCL_ASSERT(offset >= 0, "");
      _CCCL_ASSERT(data_nodes[static_cast<size_t>(offset)].has_value(), "");
      return data_nodes[static_cast<size_t>(offset)].value().frozen_ld.has_value();
    }

    access_mode get_frozen_mode(int offset) const
    {
      _CCCL_ASSERT(is_frozen(offset), "");
      return data_nodes[static_cast<size_t>(offset)].value().frozen_ld.value().get_access_mode();
    }

    ::std::shared_lock<::std::shared_mutex> acquire_shared_lock() const
    {
      return ::std::shared_lock<::std::shared_mutex>(mutex);
    }

    ::std::unique_lock<::std::shared_mutex> acquire_exclusive_lock()
    {
      return ::std::unique_lock<::std::shared_mutex>(mutex);
    }

    stackable_ctx sctx;
    ::std::vector<::std::optional<data_node>> data_nodes;
    int data_root_offset = -1;
    ::std::string symbol;
    bool read_only = false;

  private:
    mutable ::std::shared_mutex mutex;
  };

  //! Runs when the last shell copy dies; pins `state` in child graph nodes.
  struct owner
  {
    ::std::shared_ptr<state> payload;

    explicit owner(::std::shared_ptr<state> st)
        : payload(mv(st))
    {}

    ~owner()
    {
      if (payload)
      {
        adopt_into_context(payload);
      }
    }

    owner(const owner&)            = delete;
    owner& operator=(const owner&) = delete;
    owner(owner&&)                 = delete;
    owner& operator=(owner&&)      = delete;
  };

  const state& data() const
  {
    _CCCL_ASSERT(owner_ && owner_->payload, "stackable_logical_data used after move or default construction");
    return *owner_->payload;
  }

  state& mut_data()
  {
    _CCCL_ASSERT(owner_ && owner_->payload, "stackable_logical_data used after move or default construction");
    return *owner_->payload;
  }

  static void adopt_into_context(const ::std::shared_ptr<state>& st)
  {
    if (!st)
    {
      return;
    }

    auto ctx_lock = st->sctx.acquire_exclusive_lock();

    const int data_root_offset = st->data_root_offset;
    _CCCL_ASSERT(data_root_offset >= 0, "");
    _CCCL_ASSERT(st->data_nodes[static_cast<size_t>(data_root_offset)].has_value(), "");

    // Do NOT destroy data_nodes here: the root data_node may still hold a
    // frozen_ld that children depend on for pop_after_finalize to unfreeze.
    // Pin the shared state in imported child nodes so it outlives this handle
    // until every child context has finished its pop sequence.
    const auto& root_children = st->sctx.get_children_offsets(data_root_offset);
    for (auto c : root_children)
    {
      if (st->was_imported(c))
      {
        st->sctx.get_node(c)->retain_data(st);
      }
    }
  }

  template <typename Func>
  static void traverse_data_nodes(state& st, Func&& func)
  {
    ::std::stack<int> node_stack;
    node_stack.push(st.data_root_offset);

    while (!node_stack.empty())
    {
      const int offset = node_stack.top();
      node_stack.pop();

      func(st, offset);

      const auto& children = st.sctx.get_children_offsets(offset);
      for (auto it = children.rbegin(); it != children.rend(); ++it)
      {
        if (st.was_imported(*it))
        {
          node_stack.push(*it);
        }
      }
    }
  }

  void mark_access_at(int offset, access_mode m)
  {
    mut_data().mark_access(offset, m);
  }

  void push_at(int ctx_offset, access_mode m, data_place where = data_place::invalid())
  {
    auto& st                = mut_data();
    const int parent_offset = st.sctx.get_parent_offset(ctx_offset);

    if (parent_offset < 0)
    {
      _CCCL_ASSERT(st.was_imported(ctx_offset), "Root context must already have data");
      return;
    }

    _CCCL_ASSERT(ctx_offset >= 0, "");
    if (static_cast<size_t>(ctx_offset) >= st.data_nodes.size())
    {
      st.grow_data_nodes(static_cast<size_t>(ctx_offset) + 1);
    }

    if (st.data_nodes[static_cast<size_t>(ctx_offset)].has_value())
    {
      auto& existing_node = st.data_nodes[static_cast<size_t>(ctx_offset)].value();
      _CCCL_ASSERT(access_mode_permits(existing_node.effective_mode, m), "Cannot change existing access mode");
      return;
    }

    const access_mode max_required_parent_mode =
      (m == access_mode::write || m == access_mode::reduce) ? access_mode::write : access_mode::rw;

    if (!st.data_nodes[static_cast<size_t>(parent_offset)].has_value())
    {
      push_at(parent_offset, max_required_parent_mode, where);
    }

    _CCCL_ASSERT(st.data_nodes[static_cast<size_t>(parent_offset)].has_value(), "parent data should be available here");

    auto& to_node  = st.sctx.get_node(ctx_offset);
    auto& to_ctx   = to_node->ctx;
    auto& from_ctx = st.sctx.get_node(parent_offset)->ctx;

    auto& from_data_node = st.data_nodes[static_cast<size_t>(parent_offset)].value();

    if (where.is_invalid())
    {
      where = from_ctx.default_exec_place().affine_data_place();
    }

    _CCCL_ASSERT(!where.is_invalid(), "Invalid data place");

    if (!from_data_node.frozen_ld.has_value())
    {
      from_data_node.frozen_ld = from_ctx.freeze(from_data_node.ld, m, where, false /* not a user freeze */);
      from_data_node.get_cnt   = 0;
    }
    else
    {
      const access_mode existing_frozen_mode = from_data_node.frozen_ld.value().get_access_mode();

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

    ::std::pair<T, event_list> get_res = frozen_ld.get(where);
    auto ld                            = to_ctx.logical_data(get_res.first, where);
    from_data_node.get_cnt++;

    to_node->ctx_prereqs.merge(mv(get_res.second));

    if (!st.symbol.empty())
    {
      ld.set_symbol(st.symbol);
    }

    to_ctx.get_dot()->ctx_add_input_id(frozen_ld.freeze_fake_task_id());

    to_node->track_pushed_data(owner_->payload);

    st.data_nodes[static_cast<size_t>(ctx_offset)].emplace(mv(ld));
  }

  // Aliased across shell copies; `owner` runs orphan adoption in its destructor.
  ::std::shared_ptr<owner> owner_;
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
    auto& sctx = d.sctx();
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
//! Returned by stackable_ctx::pop_prologue(), this handle exposes the graph
//! built from the nested context and lets the user launch it repeatedly before
//! committing the pop via pop_epilogue(). The underlying cudaGraphExec_t is
//! instantiated lazily on the first exec()/launch() call.
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
    _CCCL_VERIFY(!token_.expired(), "launchable_graph_handle::exec() called after pop_epilogue()");
    auto shared_exec = ctx_.prepare_handle_for_exec(node_offset_);
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
    _CCCL_VERIFY(!token_.expired(), "launchable_graph_handle::stream() called after pop_epilogue()");
    return support_stream_;
  }

  //! \brief Launch the graph once on the support stream.
  //!
  //! On the first call, waits for the nested context's freeze/get events
  //! (dep A). Subsequent calls skip the sync and issue the launch directly.
  void launch()
  {
    _CCCL_VERIFY(!token_.expired(), "launchable_graph_handle::launch() called after pop_epilogue()");
    ctx_.launch_prepared_graph(node_offset_, support_stream_);
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
  //!
  //! Ordering caveat: `pop_epilogue()` only synchronizes the support stream
  //! (dep A); it does NOT wait on whatever stream you launch the embedding
  //! outer graph on. You are therefore responsible for ensuring the embedded
  //! work has completed (e.g. via `cudaStreamSynchronize` on your launch
  //! stream) before calling `pop_epilogue()`, otherwise the unfreeze of the
  //! pushed data will race the child graph.
  cudaGraph_t graph() const
  {
    _CCCL_VERIFY(!token_.expired(), "launchable_graph_handle::graph() called after pop_epilogue()");
    ctx_.prepare_handle_for_graph(node_offset_);
    return graph_;
  }

  //! \brief True iff the owning stackable_ctx has not yet run pop_epilogue().
  bool valid() const
  {
    return !token_.expired();
  }

  explicit operator bool() const
  {
    return valid();
  }

private:
  friend class stackable_ctx;

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
                             ::cuda::std::source_location loc = ::cuda::std::source_location::current())
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

inline stackable_ctx::graph_scope_guard stackable_ctx::graph_scope(::cuda::std::source_location loc)
{
  return graph_scope_guard(*this, loc);
}

//! \brief RAII wrapper for a re-launchable pop scope.
//!
//! On construction, calls `ctx.push()`. The caller builds the nested graph
//! body, then uses `launch()` (or `exec()` / `stream()` / `graph()`) as many
//! times as desired; the first such call triggers `ctx.pop_prologue()`. The
//! destructor (or an explicit `release()`) runs `ctx.pop_epilogue()` and
//! invalidates the handle.
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
                                  ::cuda::std::source_location loc = ::cuda::std::source_location::current())
      : ctx_(ctx)
  {
    ctx_.push(loc);
    // pop_prologue() runs lazily on the first launch()/exec()/stream()/graph(),
    // or on release() if none of those were called.
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

    // If no one ever called launch()/exec()/stream()/graph(): we still ran push()
    // in the constructor, so we must match it with a prologue+epilogue
    // pair to tear the node down cleanly. finalize_after_launch handles
    // the no-launch case correctly.
    ensure_prepared_();

    ctx_.pop_epilogue();
    released_ = true;
  }

private:
  void ensure_prepared_()
  {
    _CCCL_VERIFY(!released_, "launchable_graph_scope used after release()");
    if (!handle_)
    {
      handle_ = ctx_.pop_prologue();
    }
  }

  stackable_ctx& ctx_;
  launchable_graph_handle handle_;
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
    _CCCL_VERIFY(state_, "launchable_graph::launch() called on an empty/moved-from handle");
    state_->handle.launch();
  }

  //! \brief Underlying executable graph. Triggers lazy instantiation + dep-A
  //! sync on the first call (same contract as `launchable_graph_handle::exec()`).
  cudaGraphExec_t exec() const
  {
    _CCCL_VERIFY(state_, "launchable_graph::exec() called on an empty/moved-from handle");
    return state_->handle.exec();
  }

  //! \brief Support stream the graph was prepared against. Purely observational.
  cudaStream_t stream() const
  {
    _CCCL_VERIFY(state_, "launchable_graph::stream() called on an empty/moved-from handle");
    return state_->handle.stream();
  }

  //! \brief Underlying cudaGraph_t topology (for embedding as a child graph).
  //! Triggers lazy dep-A sync but does NOT call `cudaGraphInstantiate`.
  cudaGraph_t graph() const
  {
    _CCCL_VERIFY(state_, "launchable_graph::graph() called on an empty/moved-from handle");
    return state_->handle.graph();
  }

  //! \brief True iff this copy still holds a shared reference and the
  //! underlying pop has not been epilogued (e.g. manually via
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
    unsigned int default_launch_value = 0,
    unsigned int flags                = cudaGraphCondAssignDefault,
    ::cuda::std::source_location loc  = ::cuda::std::source_location::current())
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
  unsigned int default_launch_value, unsigned int flags, ::cuda::std::source_location loc)
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
  size_t count, unsigned int default_launch_value, unsigned int flags, ::cuda::std::source_location loc)
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

  // pop_prologue() finalizes the graph but does NOT instantiate or launch it;
  // instantiation happens lazily on the first handle.launch(), so the graph
  // actually runs exactly N times (once per handle.launch()).
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

inline void test_pop_prologue_graph_child_embed()
{
  // Exercise launchable_graph_handle::graph(): embed the popped graph as a
  // child node in an outer, user-built graph instead of launching it through
  // the handle. graph() does NOT instantiate an exec graph; it only performs
  // the lazy dep-A sync so that handle.stream() becomes a valid event source
  // for ordering the outer launch.
  //
  // Ordering caveat exercised here: pop_epilogue() only synchronizes the
  // support stream (dep A), NOT the caller's outer launch stream. The caller
  // is therefore responsible for ensuring the embedded work has completed
  // before pop_epilogue() runs, otherwise the unfreeze would race the child
  // graph. We make that explicit with the cudaStreamSynchronize() below.
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

  // graph() returns the finalized cudaGraph_t without instantiating an exec
  // graph, and performs the lazy dep-A sync on handle.stream().
  cudaGraph_t body = handle.graph();

  // Build an outer graph that embeds `body` as a child node.
  cudaGraph_t outer = nullptr;
  cuda_safe_call(cudaGraphCreate(&outer, 0));
  cudaGraphNode_t child{};
  cuda_safe_call(cudaGraphAddChildGraphNode(&child, outer, nullptr, 0, body));

  cudaGraphExec_t outer_exec = nullptr;
  cuda_safe_call(cudaGraphInstantiateWithFlags(&outer_exec, outer, 0));

  // Order the outer launch behind the nested context's freeze/get events:
  // record an event on handle.stream() (where graph() injected dep A) and make
  // our launch stream wait on it before launching the embedded child.
  cudaStream_t launch_stream = nullptr;
  cuda_safe_call(cudaStreamCreate(&launch_stream));
  cudaEvent_t dep_a = nullptr;
  cuda_safe_call(cudaEventCreate(&dep_a));
  cuda_safe_call(cudaEventRecord(dep_a, handle.stream()));
  cuda_safe_call(cudaStreamWaitEvent(launch_stream, dep_a, 0));

  cuda_safe_call(cudaGraphLaunch(outer_exec, launch_stream));

  // The embedded child must finish before pop_epilogue() unfreezes the data.
  cuda_safe_call(cudaStreamSynchronize(launch_stream));

  ctx.pop_epilogue();

  cuda_safe_call(cudaGraphExecDestroy(outer_exec));
  cuda_safe_call(cudaGraphDestroy(outer));
  cuda_safe_call(cudaEventDestroy(dep_a));
  cuda_safe_call(cudaStreamDestroy(launch_stream));

  ctx.host_launch(lA.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); ++i)
    {
      _CCCL_ASSERT(a(i) == 1, "pop_prologue: graph() child-embed did not run the body exactly once");
    }
  };

  ctx.finalize();
}

UNITTEST("pop_prologue graph() embedded as a child graph node")
{
  test_pop_prologue_graph_child_embed();
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
