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
 * @brief Implement the task class and methods to implement the STF programming model
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

/*
 * This is a generic class of "tasks" that are synchronized according to
 * accesses on "data" depending on in/out dependencies
 */

#include <cuda/experimental/__stf/internal/msir.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh> // task has-a task_dep_vector_untyped

#include <optional>

namespace cuda::experimental::stf
{

namespace reserved
{

class mapping_id_tag
{};

using mapping_id_t = reserved::unique_id<reserved::mapping_id_tag>;

} // end namespace reserved

class backend_ctx_untyped;
class logical_data_untyped;
class exec_place;

void reclaim_memory(
  backend_ctx_untyped& ctx, const data_place& place, size_t requested_s, size_t& reclaimed_s, event_list& prereqs);

/**
 * @brief Generic implementation of a task based on asynchronous events and accessing logical data
 */
class task
{
public:
  /**
   * @brief current task status
   *
   * We keep track of the status of task so that we do not make API calls at an
   * inappropriate time, such as setting the symbol once the task has already
   * started, or releasing a task that was not started yet.
   */
  enum class phase
  {
    setup, // the task has not started yet
    running, // between acquire and release
    finished, // we have released
  };

private:
  // pimpl
  class impl
  {
  public:
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    impl(exec_place where = exec_place::current_device())
        : e_place(mv(where))
        , affine_data_place(e_place.affine_data_place())
    {}

    // Vector of user provided deps
    task_dep_vector_untyped deps;

    // This list is only useful when calling the get() method of a task, to
    // reduce overheads, we initialize this vector lazily
    void initialize_reordered_indexes()
    {
      // This list gives converts the original index to the sorted index
      // For example the first entry of the list before being ordered has order t->reordered_index[0]
      reordered_indexes.resize(deps.size());

      int sorted_index = 0;
      for (auto& it : deps)
      {
        reordered_indexes[it.dependency_index] = sorted_index;
        sorted_index++;
      }
    }

    // Get the index of the dependency after reordering, for example
    // deps[reordered_index[0]] is the first piece of data
    ::std::vector<size_t> reordered_indexes;

    // Indices of logical data which were locked (non skipped). Indexes are
    // those obtained after sorting.
    ::std::vector<::std::pair<size_t, access_mode>> unskipped_indexes;

    // Extra events that need to be done before the task starts. These are
    // "extra" as these are in addition to the events that will be required to
    // acquire the logical_data_untypeds accessed by the task
    event_list input_events;

    // A string useful for debugging purpose
    mutable ::std::string symbol;

    // This points to the prerequisites for this task's termination
    event_list done_prereqs;

    // Used to uniquely identify the task
    reserved::unique_id_t unique_id;

    // Used to uniquely identify the task for mapping purposes
    reserved::mapping_id_t mapping_id;

    // This is a pointer to a generic data structure used by "unset_place" to
    // restore previous context
    exec_place saved_place_ctx;

    // Indicate the status of the task
    task::phase phase = task::phase::setup;

    // This is where the task is executed
    exec_place e_place;

    // This is the default data place for the task. In general this is the
    // affine data place of the execution place, but this can be a
    // composite data place when using a grid of places for example.
    data_place affine_data_place;

    // Automatically capture work when this is a graph task (ignored with a
    // CUDA stream backend).
    bool enable_capture = false;
  };

protected:
  // This is the only state
  ::std::shared_ptr<impl> pimpl;

public:
  task()
      : pimpl(::std::make_shared<impl>())
  {}

  task(exec_place ep)
      : pimpl(::std::make_shared<impl>(mv(ep)))
  {}

  task(const task& rhs)
      : pimpl(rhs.pimpl)
  {}
  task(task&&) = default;

  task& operator=(const task& rhs) = default;
  task& operator=(task&& rhs)      = default;

  explicit operator bool() const
  {
    return pimpl != nullptr;
  }

  bool operator==(const task& rhs) const
  {
    return pimpl == rhs.pimpl;
  }

  /// Get the string attached to the task for debugging purposes
  const ::std::string& get_symbol() const
  {
    if (pimpl->symbol.empty())
    {
      pimpl->symbol = "task " + ::std::to_string(pimpl->unique_id);
    }
    return pimpl->symbol;
  }

  /// Attach a string to this task, which can be useful for debugging purposes, or in tracing tools.
  void set_symbol(::std::string new_symbol)
  {
    EXPECT(get_task_phase() == phase::setup);
    pimpl->symbol = mv(new_symbol);
  }

  /// Add one dependency
  void add_dep(task_dep_untyped d)
  {
    EXPECT(get_task_phase() == phase::setup);
    pimpl->deps.push_back(mv(d));
  }

  /// Add a set of dependencies
  void add_deps(task_dep_vector_untyped input_deps)
  {
    EXPECT(get_task_phase() == phase::setup);
    if (pimpl->deps.empty())
    {
      // Frequent case
      pimpl->deps = mv(input_deps);
    }
    else
    {
      pimpl->deps.insert(
        pimpl->deps.end(), ::std::make_move_iterator(input_deps.begin()), ::std::make_move_iterator(input_deps.end()));
    }
  }

  /// Add a set of dependencies
  template <typename... Pack>
  void add_deps(task_dep_untyped first, Pack&&... pack)
  {
    EXPECT(get_task_phase() == phase::setup);
    pimpl->deps.push_back(mv(first));
    if constexpr (sizeof...(Pack) > 0)
    {
      add_deps(::std::forward<Pack>(pack)...);
    }
  }

  /// Add a tuple of dependencies
  template <typename... Args>
  void add_deps(::std::tuple<Args...>& deps_tuple)
  {
    ::std::apply(
      [this](const auto&... deps) {
        // Call add_deps on each dep using a fold expression
        //
        // Note that we use this-> while it seems unnecessary to work-around
        // some compiler issue which otherwise believe the "this" captured
        // value is unused.
        (this->add_deps(deps), ...);
      },
      deps_tuple);
  }

  /// Get the dependencies of the task
  const task_dep_vector_untyped& get_task_deps() const
  {
    return pimpl->deps;
  }

  /// Specify where the task should run
  task& on(exec_place p)
  {
    EXPECT(get_task_phase() == phase::setup);
    // This defines an affine data place too
    set_affine_data_place(p.affine_data_place());
    pimpl->e_place = mv(p);
    return *this;
  }

  /// Get and set the execution place of the task
  const exec_place& get_exec_place() const
  {
    return pimpl->e_place;
  }

  exec_place& get_exec_place()
  {
    return pimpl->e_place;
  }

  void set_exec_place(const exec_place& place)
  {
    // This will both update the execution place and the affine data place
    on(place);
  }

  /// Get and Set the affine data place of the task
  const data_place& get_affine_data_place() const
  {
    return pimpl->affine_data_place;
  }

  void set_affine_data_place(data_place affine_data_place)
  {
    pimpl->affine_data_place = mv(affine_data_place);
  }

  dim4 grid_dims() const
  {
    return get_exec_place().grid_dims();
  }

  /// Get the list of events which mean that the task was executed
  const event_list& get_done_prereqs() const
  {
    return pimpl->done_prereqs;
  }

  /// Add an event list to the list of events which mean that the task was executed
  template <typename T>
  void merge_event_list(T&& tail)
  {
    pimpl->done_prereqs.merge(::std::forward<T>(tail));
  }

  /**
   * @brief Get the identifier of a data instance used by a task
   *
   * We here find the instance id used by a given piece of data in a task.
   * Note that this incurs a certain overhead because it searches through the
   * list of logical data in the task.
   */
  instance_id_t find_data_instance_id(const logical_data_untyped& d) const;

  /**
   * @brief Generic method to retrieve the data instance associated to an
   * index in a task.
   *
   * If `T` is the exact type stored, this returns a reference to a valid data instance in the task. If `T` is
   * `constify<U>`, where `U` is the type stored, this returns an rvalue of type `T`.
   *
   * Calling this outside the start()/end() section will result in undefined behaviour.
   *
   * @remark One should not forget the "template" keyword when using this API with a task `t`
   *    `T &res = t.template get<T>(index);`
   */
  template <typename T, typename logical_data_untyped = logical_data_untyped>
  decltype(auto) get(size_t submitted_index) const;

  // If there are extra input dependencies in addition to STF-induced events
  void set_input_events(event_list _input_events)
  {
    EXPECT(get_task_phase() == phase::setup);
    pimpl->input_events = mv(_input_events);
  }

  const event_list& get_input_events() const
  {
    return pimpl->input_events;
  }

  // Get the unique task identifier
  int get_unique_id() const
  {
    return pimpl->unique_id;
  }

  // Get the unique task mapping identifier
  int get_mapping_id() const
  {
    return pimpl->mapping_id;
  }

  size_t hash() const
  {
    return ::std::hash<impl*>()(pimpl.get());
  }

  void enable_capture()
  {
    fprintf(stderr, "task enable capture (generic task)\n");
    pimpl->enable_capture = true;
  }

  bool is_capture_enabled() const
  {
    return pimpl->enable_capture;
  }

  /**
   * @brief Start a task
   *
   *    SUBMIT = acquire + release at the same time ...
   */
  // Resolve all dependencies at the specified execution place
  // Returns execution prereqs
  event_list acquire(backend_ctx_untyped& ctx);

  void release(backend_ctx_untyped& ctx, event_list& done_prereqs);

  // Returns the current state of the task
  phase get_task_phase() const
  {
    EXPECT(pimpl);
    return pimpl->phase;
  }

  /* When the task has ended, we cannot do anything with it. It is possible
   * that the user-facing task object is not destroyed when the context is
   * synchronized, so we clear it.
   *
   * This for example happens when doing :
   *   auto t = ctx.task(A.rw());
   *   t->*[](auto A){...};
   *   ctx.finalize();
   */
  void clear()
  {
    pimpl.reset((cuda::experimental::stf::task::impl*) nullptr);
  }
};

namespace reserved
{

/* This method lazily allocates data (possibly reclaiming memory) and copies data if needed */
template <typename Data>
void dep_allocate(
  backend_ctx_untyped& ctx,
  Data& d,
  access_mode mode,
  const data_place& dplace,
  const ::std::optional<exec_place> eplace,
  instance_id_t instance_id,
  event_list& prereqs)
{
  auto& inst = d.get_data_instance(instance_id);

  /*
   * DATA LAZY ALLOCATION
   */
  bool already_allocated = inst.is_allocated();
  if (!already_allocated)
  {
    // nvtx_range r("acquire::allocate");
    /* Try to allocate memory : if we fail to do so, we must try to
     * free other instances first, and retry */
    int alloc_attempts = 0;
    while (true)
    {
      ::std::ptrdiff_t s = 0;

      prereqs.merge(inst.get_read_prereq(), inst.get_write_prereq());

      // The allocation routine may decide to store some extra information
      void* extra_args = nullptr;

      d.allocate(dplace, instance_id, s, &extra_args, prereqs);

      // Save extra_args
      inst.set_extra_args(extra_args);

      if (s >= 0)
      {
        // This allocation was successful
        inst.allocated_size = s;
        inst.set_allocated(true);
        inst.reclaimable = true;
        break;
      }

      assert(s < 0);

      // Limit the number of attempts if it's simply not possible
      EXPECT(alloc_attempts++ < 5);

      // We failed to allocate so we try to reclaim
      size_t reclaimed_s = 0;
      size_t needed      = -s;
      reclaim_memory(ctx, dplace, needed, reclaimed_s, prereqs);
    }

    // After allocating a reduction instance, we need to initialize it
    if (mode == access_mode::relaxed)
    {
      assert(eplace.has_value());
      // We have just allocated a new piece of data to perform
      // reductions, so we need to initialize this with an
      // appropriate user-provided operator
      // First get the data instance and then its reduction operator
      ::std::shared_ptr<reduction_operator_base> ops = inst.get_redux_op();
      ops->init_op_untyped(d, dplace, instance_id, eplace.value(), prereqs);
    }
  }
}

} // end namespace reserved

// inline size_t task_state::hash() const {
//     size_t h = 0;
//     for (auto& e: logical_data_ids) {
//         int id = e.first;
//         auto handle = e.second.lock();
//         // ignore expired handles
//         if (handle) {
//             hash_combine(h, ::std::hash<int> {}(id));
//             hash_combine(h, handle->hash());
//         }
//     }

//     return h;
// }

/**
 * @brief Data instance implementation
 *
 * This describes an "instance" of a logical data. This is one copy on a data
 * place. There can be multiple data instances of the same logical data on
 * different places, or at the same places when we have access modes such as
 * reductions.
 *
 * The data instance contains everything required to keep track of the events
 * which need to be fulfilled prior to using that copy of the data. The "state"
 * field makes it possible to implement the MSI protocol.
 *
 * Note that a data instance may correspond to a piece of data that is out of
 * sync, but that is allocated (or not). In this case, future accesses to the
 * logical data associated to this data instance on that data place will
 * transfer a copy from a data instance that is valid. */
class data_instance
{
public:
  data_instance() {}
  data_instance(bool used, data_place dplace)
      : used(used)
      , dplace(mv(dplace))
  {
#if 0
        // Since this will default construct a task, we need to decrement the id
        reserved::mapping_id_t::decrement_id();
#endif
  }

  void set_used(bool flag)
  {
    assert(flag != used);
    used = flag;
  }

  bool get_used() const
  {
    return used;
  }

  void set_dplace(data_place _dplace)
  {
    dplace = mv(_dplace);
  }
  const data_place& get_dplace() const
  {
    return dplace;
  }

  // Returns what is the reduction operator associated to this data instance
  ::std::shared_ptr<reduction_operator_base> get_redux_op() const
  {
    return redux_op;
  }
  // Sets the reduction operator associated to that data instance
  void set_redux_op(::std::shared_ptr<reduction_operator_base> op)
  {
    redux_op = op;
  }

  // Indicates if the data instance is allocated (ie. if it needs to be
  // allocated prior to use). Note that we may have allocated instances that
  // are out of sync too.
  bool is_allocated() const
  {
    return state.is_allocated();
  }

  void set_allocated(bool b)
  {
    state.set_allocated(b);
  }

  reserved::msir_state_id get_msir() const
  {
    return state.get_msir();
  }

  void set_msir(reserved::msir_state_id st)
  {
    state.set_msir(st);
  }

  const event_list& get_read_prereq() const
  {
    return state.get_read_prereq();
  }
  const event_list& get_write_prereq() const
  {
    return state.get_write_prereq();
  }

  void set_read_prereq(event_list prereq)
  {
    state.set_read_prereq(mv(prereq));
  }
  void set_write_prereq(event_list prereq)
  {
    state.set_write_prereq(mv(prereq));
  }

  void add_read_prereq(backend_ctx_untyped& bctx, const event_list& _prereq)
  {
    state.add_read_prereq(bctx, _prereq);
  }
  void add_write_prereq(backend_ctx_untyped& bctx, const event_list& _prereq)
  {
    state.add_write_prereq(bctx, _prereq);
  }

  void clear_read_prereq()
  {
    state.clear_read_prereq();
  }
  void clear_write_prereq()
  {
    state.clear_write_prereq();
  }

  bool has_last_task_relaxed() const
  {
    return last_task_relaxed.has_value();
  }
  void set_last_task_relaxed(task t)
  {
    last_task_relaxed = mv(t);
  }
  const task& get_last_task_relaxed() const
  {
    assert(last_task_relaxed.has_value());
    return last_task_relaxed.value();
  }

  int max_prereq_id() const
  {
    return state.max_prereq_id();
  }

  // Compute a hash of the MSI/Alloc state
  size_t state_hash() const
  {
    return hash<reserved::per_data_instance_msi_state>{}(state);
  }

  void set_extra_args(void* args)
  {
    extra_args = args;
  }

  void* get_extra_args() const
  {
    return extra_args;
  }

  void clear()
  {
    clear_read_prereq();
    clear_write_prereq();
    last_task_relaxed.reset();
  }

private:
  // Is this instance available or not ? If not we can reuse this data
  // instance when looking for an available slot in the vector of data
  // instances attached to the logical data
  bool used = false;

  // If the used flag is set, this tells where this instance is located
  data_place dplace;

  // Reduction operator attached to the data instance
  ::std::shared_ptr<reduction_operator_base> redux_op;

  // @@@@TODO@@@@ There are a lot of unchecked forwarding with this variable,
  // which is public in practice ...
  //
  // This structure contains everything to implement the MSI protocol,
  // including asynchronous prereqs so that we only use a data instance once
  // it's ready to do so
  reserved::per_data_instance_msi_state state;

  // This stores the last task which used this instance with a relaxed coherence mode (redux)
  ::std::optional<task> last_task_relaxed;

  // This generic pointer can be used to store some information in the
  // allocator which is passed to the deallocation routine.
  void* extra_args = nullptr;

public:
  // Size of the memory allocation (bytes). Only valid for allocated instances.
  size_t allocated_size = 0;

  // A false value indicates that this instance cannot be a candidate for
  // memory reclaiming (e.g. because this corresponds to memory allocated by
  // the user)
  bool reclaimable = false;

  bool automatically_pinned = false;
};

template <>
struct hash<task>
{
  ::std::size_t operator()(const task& t) const
  {
    return t.hash();
  }
};

} // namespace cuda::experimental::stf
