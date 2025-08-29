//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Definition of `logical_data` and `task_dep_vector_untyped`
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

#include <cuda/experimental/__stf/internal/backend_ctx.cuh> // logical_data_untyped_impl has a backend_ctx_untyped
#include <cuda/experimental/__stf/internal/constants.cuh>
#include <cuda/experimental/__stf/internal/data_interface.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/pretty_print.cuh>

#include <map>
#include <mutex>
#include <optional>

namespace cuda::experimental::stf
{

logical_data_untyped unpack_state(const ::std::shared_ptr<void>&);

class logical_data_untyped_impl;

/**
 * @brief This represents a set of tasks.
 *
 * The goal of this class is to avoid storing very large containers of task
 * structure which may hold many events for a long time (and needlessly consume
 * resources), so that we can optimize the events. This class is useful when
 * all tasks in the set are used together, for example when we enforce some
 * write-after-read dependency when the writer tasks must depend on all
 * previous readers.
 *
 * We can add an individual task to a set, or remove all of them, but we cannot
 * remove a single task.
 */
class task_set
{
public:
  task_set() = default;

  /* Add a task to the set : this will copy the done prereqs, and save the ID */
  void add(backend_ctx_untyped& bctx, const task& t)
  {
    done_prereqs.merge(t.get_done_prereqs());
    done_prereqs.optimize(bctx);
    // this will maybe depend on the use of dot or not
    task_ids.push_back(t.get_unique_id());
  }

  void clear()
  {
    task_ids.clear();
    done_prereqs.clear();
  }

  bool empty() const
  {
    return task_ids.empty();
  }

  const event_list& get_done_prereqs() const
  {
    return done_prereqs;
  }

  /* Get a reference to the vector of tasks IDs */
  const auto& get_ids() const
  {
    return task_ids;
  }

private:
  /* All task identifiers in the set */
  ::std::vector<int> task_ids;

  // Collection of all events marking the completion of the tasks in the set
  event_list done_prereqs;
};

namespace reserved
{

/**
 * @brief This class describes the status of a logical data (e.g. previous writer,
 * readers...) in a specific task context.
 *
 * This makes it possible to use the same logical data in nested tasks while
 * enforcing the STF coherency model.
 *
 * Implementing the STF model for example requires to introduce implicit
 * dependencies between a task making a write access, and all preceding read
 * accesses. The list of previous readsers, the previous writer etc... are kept on
 * a per context basis.
 */
struct logical_data_state
{
  // Reset structures which hold some state before we can call the destructor
  void clear()
  {
    current_writer.reset();
    previous_writer.reset();
    current_readers.clear();
  }

  access_mode current_mode = access_mode::none;

  // A data structure to store the events to implement dependencies with
  // previous read accesse : future writer will have to wait for them (WaR
  // accesses)
  task_set current_readers;

  // Task currently making a write access : future readers or writer will
  // have to wait for it (RaW or WaW accesses)
  ::std::optional<task> current_writer;

  // Previous writer which all new readers will need to sync with (RaW accesses)
  // We use a vector so that we don't store a null task
  ::std::optional<task> previous_writer;

  /* If we are tracing dependencies to generate a DOT output, we keep track
   * of the identifiers of the tasks which performed a reduction access on
   * that piece of data. */
  ::std::vector<int> pending_redux_id;
};

/*
Implementation for `logical_data_untyped`, which uses the pimpl idiom. This
class cannot be nested inside `logical_data_untyped_impl` because it is used in
`task_state`, which holds a map of `logical_data_untyped_impl &`.

We currently use shared_from_this() when reconstructing a logical data after a
reduction mode.
*/
class logical_data_untyped_impl : public ::std::enable_shared_from_this<logical_data_untyped_impl>
{
public:
  logical_data_untyped_impl(
    backend_ctx_untyped ctx_, ::std::shared_ptr<data_interface> dinterface_, const data_place& memory_node)
      : ctx(mv(ctx_))
      , dinterface(mv(dinterface_))
  {
    assert(this->ctx);
    assert(this->dinterface);

    refcnt.store(0);

    // This will automatically create a weak_ptr from the shared_ptr
    ctx.get_state().logical_data_ids_mutex.lock();
    ctx.get_state().logical_data_ids.emplace(get_unique_id(), *this);
    ctx.get_state().logical_data_ids_mutex.unlock();

    // It is possible that there is no valid copy (e.g. with temporary accesses)
    if (memory_node.is_invalid())
    {
      return; // no data instance present, so nothing to do for the time being
    }

    // Get an id (and reserve it)
    const instance_id_t instance_id = find_instance_id(memory_node);

    // Record this id for the write back mechanism
    reference_instance_id = instance_id;
    enable_write_back     = true;

    // setup MSI status
    auto& inst = get_data_instance(instance_id);

    // If this is host memory, try to pin it
    if (memory_node.is_host())
    {
      // ret may be false if we detected that the instance was already pinned, for example
      inst.automatically_pinned = dinterface->pin_host_memory(instance_id);
    }

#ifndef NDEBUG
    auto mem_type = dinterface->get_memory_type(instance_id);

    if (memory_node.is_managed())
    {
      assert(!mem_type || *mem_type == cudaMemoryTypeManaged);
    }

    if (memory_node.is_device())
    {
      assert(!mem_type || *mem_type == cudaMemoryTypeDevice);
    }
#endif

    inst.set_msir(reserved::msir_state_id::modified);
    inst.set_allocated(true);

    assert(inst.get_read_prereq().size() == 0);
    assert(inst.get_write_prereq().size() == 0);

    // If the beginning of the context depends on a prereq, we assume this logical data depends on it
    if (ctx.has_start_events())
    {
      inst.add_read_prereq(ctx, ctx.get_start_events());
    }

    // This is not an instance allocated by our library, so we will not
    // consider it for reclaiming for example
    inst.reclaimable = false;
  }

  ~logical_data_untyped_impl()
  {
    erase();
  }

  /**
   * @brief `erase` destroys all data instances that were automatically
   * created, and stores a valid value in the reference data instance that
   * was used to create the logical data, if any.
   */
  void erase();

  // Data handle unique id : when this object is moved, it is set automatically to -1
  reserved::unique_id_t unique_id;

  backend_ctx_untyped ctx;

  reserved::logical_data_state state;

  // For temporary or relaxed accesses, we need to be able to find an available entry
  ::std::vector<data_instance> used_instances;

  // A string useful for debugging purpose
  mutable ::std::string symbol;

  // This will be used to store methods to manage data coherency
  // (transfers, allocations etc.) or to have data specific accessors.
  ::std::shared_ptr<data_interface> dinterface;

  void freeze(access_mode freeze_mode, data_place place = data_place::invalid())
  {
    // We cannot freeze some logical data that is already frozen
    assert(!frozen_flag);

    if (freeze_mode != access_mode::read)
    {
      _CCCL_ASSERT(!place.is_invalid(), "invalid data place");
    }

    frozen_flag  = true;
    frozen_mode  = freeze_mode;
    frozen_place = mv(place);
  }

  void unfreeze(task& fake_task, event_list prereqs);

  void set_automatic_unfreeze(task& unfreeze_fake_task_, bool flag)
  {
    automatic_unfreeze = flag;

    // Save for future use when destroying data
    unfreeze_fake_task = unfreeze_fake_task_;
  }

  // This needs the full definition of logical_data_untyped so the implementation is deferred
  template <typename T>
  ::std::pair<T, event_list> get_frozen(task& fake_task, const data_place& dplace, access_mode m);

  /**
   * @brief Indicates whether a logical data is frozen or not, and what is the corresponding mode
   *
   * This is for example used when creating a task to check that it does not
   * try to make illegal accesses on frozen data (eg. concurrent writes)
   */
  ::std::pair<bool, access_mode> is_frozen() const
  {
    return ::std::pair<bool, access_mode>(frozen_flag, frozen_mode);
  }

  // Indicate if the data was frozen, and the corresponding mode (mode is ignored otherwise)
  bool frozen_flag = false;
  access_mode frozen_mode; // meaningful only if frozen
  data_place frozen_place; // meaningful only if frozen and frozen_mode is write or rw

  // When set, frozen data will automatically be unfrozen when getting
  // destroyed. This assumed all dependencies are solved by other means (eg.
  // because it is used within other tasks)
  bool automatic_unfreeze = false;
  ::std::optional<task> unfreeze_fake_task;

  // This defines how to allocate/deallocate raw buffers (ptr+size) within
  // the interface, if undefined (set to nullptr), then the default allocator
  // of the context will be used
  block_allocator_untyped custom_allocator;

  template <typename Q>
  static auto& get_allocator(Q& qthis)
  {
    if (qthis.custom_allocator)
    {
      return qthis.custom_allocator;
    }
    return qthis.ctx.get_allocator();
  }

  /* We get the per-logical data block_allocator, or the ctx one otherwise */
  const auto& get_allocator() const
  {
    return get_allocator(*this);
  }
  auto& get_allocator()
  {
    return get_allocator(*this);
  }

  void set_allocator(block_allocator_untyped a)
  {
    custom_allocator = mv(a);
  }

  // We keep a reference count to see whether there is a task currently
  // accessing that piece of data, so that we cannot evict a piece of data
  // that is used when looking to recycle memory (during the memory
  // reclaiming mechanism).
  ::std::atomic<int> refcnt;

  // identifier of the instance used to create the logical_data_untyped
  // -1 means there is no reference
  // This id will be the instance which will be updated by the write-back
  // mechanism when the data is destroyed, or when the context is
  // synchronized.
  instance_id_t reference_instance_id = instance_id_t::invalid;

  bool enable_write_back = false;

  // Enable or disable write-back. Enabling write-back will cause an error if there is no reference data instance.
  void set_write_back(bool flag)
  {
    if (flag)
    {
      // Do not enable write-back on a logical data that was initialized from a shape, for example
      if (reference_instance_id == instance_id_t::invalid)
      {
        fprintf(stderr, "Error: cannot enable write-back on a logical data with no reference instance.\n");
        abort();
      }
    }

    enable_write_back = flag;
  }

  bool was_erased = false;

  // Get the index of the first available instance_id
  instance_id_t find_unused_instance_id(const data_place& dplace)
  {
    for (auto i : each(used_instances.size()))
    {
      if (!used_instances[i].get_used())
      {
        used_instances[i].set_dplace(dplace);
        used_instances[i].set_used(true);
        return instance_id_t(i);
      }
    }

    used_instances.emplace_back(true, dplace);
    return instance_id_t(used_instances.size() - 1);
  }

  // Get the index of the first used instance_id that matches the data place, or get a new one
  instance_id_t find_instance_id(const data_place& dplace)
  {
    // Try to find a used instance id that has the same data place
    for (auto i : each(used_instances.size()))
    {
      // This will evaluate get_used() first and stop if this is false
      if (used_instances[i].get_used() && (used_instances[i].get_dplace() == dplace))
      {
        return instance_id_t(i);
      }
    }

    // We must use a new entry instead because there was no matching entry
    return find_unused_instance_id(dplace);
  }

  // Make it possible to reuse an id
  void release_instance_id(instance_id_t instance_id)
  {
    const auto i = size_t(instance_id);
    assert(instance_id != instance_id_t::invalid && i < used_instances.size() && used_instances[i].get_used());
    used_instances[i].set_used(false);
    // note that we do not change dplace as it's supposed to be unused when
    // the used flag is set to false.
  }

  // Find a valid copy given a hint and return its instance id. Currently the hint parameter is not used.
  instance_id_t find_source_instance_id(instance_id_t dst_instance_id)
  {
    ::std::ignore = dst_instance_id;
    assert(get_data_instance(dst_instance_id).get_msir() == reserved::msir_state_id::invalid);

// @@@@ TODO @@@@ reimplement !
#if 0
            instance_id_t dst_node = dst_place.memory_node;

            // Force initialization
            auto& machine = reserved::machine::instance();

            // We iterate over nodes, in an order that is could improve locality
            for (int n = 0; n < nnodes(); n++) {
                int n_aux = machine.get_ith_closest_node(dst_node, n);
                if (get_data_instance(n_aux).get_msir() != reserved::msir_state_id::invalid) {
                    return data_place(n_aux);
                }
            }
#else
    auto nnodes = instance_id_t(get_data_instance_count());
    for (auto id : each(nnodes))
    {
      if (get_data_instance(id).get_msir() != reserved::msir_state_id::invalid)
      {
        return id;
      }
    }
#endif

    fprintf(stderr, "FATAL: no valid source found.\n");
    abort();
    return instance_id_t::invalid;
  }

  bool has_interface() const
  {
    return dinterface != nullptr;
  }

  bool is_void_interface() const
  {
    _CCCL_ASSERT(has_interface(), "uninitialized logical data");
    return dinterface->is_void_interface();
  }

  bool has_ref() const
  {
    assert(refcnt.load() >= 0);
    return refcnt.load() != 0;
  }

  void add_ref()
  {
    assert(refcnt.load() >= 0);
    ++refcnt;
  }

  void remove_ref()
  {
    assert(refcnt >= 1);
    --refcnt;
  }

  const data_place& get_instance_dplace(instance_id_t instance_id) const
  {
    return used_instances[size_t(instance_id)].get_dplace();
  }

  const data_instance& get_data_instance(instance_id_t instance_id) const
  {
    return used_instances[size_t(instance_id)];
  }

  data_instance& get_data_instance(instance_id_t instance_id)
  {
    return used_instances[size_t(instance_id)];
  }

  size_t get_data_instance_count() const
  {
    return used_instances.size();
  }

  void allocate(const data_place& memory_node,
                instance_id_t instance_id,
                ::std::ptrdiff_t& s,
                void** extra_args,
                event_list& prereqs)
  {
    _CCCL_ASSERT(!memory_node.is_invalid(), "invalid data place");
    _CCCL_ASSERT(has_interface(), "");
    // nvtx_range r("allocate");
    // Get the allocator for this logical data
    return dinterface->data_allocate(ctx, get_allocator(), memory_node, instance_id, s, extra_args, prereqs);
  }

  void deallocate(const data_place& memory_node, instance_id_t instance_id, void* extra_args, event_list& prereqs)
  {
    _CCCL_ASSERT(!memory_node.is_invalid(), "invalid data place");
    assert(has_interface());
    // nvtx_range r("deallocate");
    return dinterface->data_deallocate(ctx, get_allocator(), memory_node, instance_id, extra_args, prereqs);
  }

  void data_copy(const data_place& dst_node,
                 instance_id_t dst_instance_id,
                 const data_place& src_node,
                 instance_id_t src_instance_id,
                 event_list& prereqs)
  {
    _CCCL_ASSERT(!src_node.is_invalid(), "invalid data place");
    _CCCL_ASSERT(!dst_node.is_invalid(), "invalid data place");
    _CCCL_ASSERT(has_interface(), "");
    // nvtx_range r("data_copy");
    ctx.add_transfer(src_node, dst_node, dinterface->data_footprint());
    return dinterface->data_copy(ctx, dst_node, dst_instance_id, src_node, src_instance_id, prereqs);
  }

  void write_back(const data_place& src_node, instance_id_t instance_id, event_list& prereqs)
  {
    assert(reference_instance_id != instance_id_t::invalid);
    data_copy(get_instance_dplace(reference_instance_id), reference_instance_id, src_node, instance_id, prereqs);
  }

  reserved::logical_data_state& get_state()
  {
    return state;
  }

  int get_unique_id() const
  {
    return unique_id;
  }

  const ::std::string& get_symbol() const
  {
    if (symbol.empty())
    {
      symbol = ::std::to_string(get_unique_id());
    }

    return symbol;
  }

  size_t hash() const
  {
    size_t h = 0;
    for (auto i : each(used_instances.size()))
    {
      // Compute a hash of the MSI/alloc state
      size_t h_i_state = used_instances[i].state_hash();

      // Compute a hash of the actual interface representation
      size_t h_i_data = dinterface->data_hash(instance_id_t(i));

      // Combine it with previously computed hashes
      hash_combine(h, h_i_state);
      hash_combine(h, h_i_data);
    }

    return h;
  }

  void reclaim_update_state(const data_place& memory_node, instance_id_t instance_id, event_list& prereqs)
  {
    auto& current_instance = get_data_instance(instance_id);
    auto current_state     = current_instance.get_msir();

    //    static size_t total_write_back_cnt = 0;

    /* Update MSI status depending on the current states and the required access mode */
    switch (current_state)
    {
      case reserved::msir_state_id::invalid:
        // fprintf(stderr, "RECLAIM %s WITH NO TRANSFER (INVALID)... (wb cnt = %ld)\n", get_symbol().c_str(),
        //         total_write_back_cnt);
        // No-op !
        current_instance.add_read_prereq(ctx, prereqs);
        break;

      case reserved::msir_state_id::modified: {
        // Host becomes the only valid copy
        // XXX @@@@TODO@@@ ! reclaims assumes that 0 == host we need a reference copy
        auto& ref_instance = get_data_instance(instance_id_t(0));
        ref_instance.set_msir(reserved::msir_state_id::modified);
        current_instance.set_msir(reserved::msir_state_id::invalid);

        prereqs.merge(ref_instance.get_read_prereq(), current_instance.get_read_prereq());

        write_back(memory_node, instance_id, prereqs);
        // total_write_back_cnt++;
        // fprintf(stderr, "WRITE BACK... %s (%ld)!!\n", get_symbol().c_str(), total_write_back_cnt);

        ref_instance.add_read_prereq(ctx, prereqs);
        current_instance.add_read_prereq(ctx, prereqs);
        break;
      }

      case reserved::msir_state_id::shared: {
        // fprintf(stderr, "RECLAIM %s WITH NO TRANSFER (SHARED)... (wb cnt = %ld)\n", get_symbol().c_str(),
        //         total_write_back_cnt);

        // Invalidate this copy, others may become either reserved::msir_state_id::shared or
        // reserved::msir_state_id::modified
        int cpy_cnt = 0;

        const auto nnodes = instance_id_t(get_data_instance_count());
        for (auto n : each(nnodes))
        {
          if (get_data_instance(n).get_msir() != reserved::msir_state_id::invalid)
          {
            cpy_cnt++;
          }
        }

        assert(cpy_cnt > 0);

        current_instance.set_msir(reserved::msir_state_id::invalid);
        current_instance.add_read_prereq(ctx, prereqs);

        // Update other copies (if needed)
        for (auto n : each(nnodes))
        {
          auto& inst_n = get_data_instance(n);
          if (inst_n.get_msir() != reserved::msir_state_id::invalid)
          {
            // If there was 2 shared copies, there only remain ones
            // now, which becomes modified. If there were more they
            // remain shared, there cannot be less than 2 copies if
            // this is shared
            auto new_state = (cpy_cnt == 2) ? reserved::msir_state_id::modified : reserved::msir_state_id::shared;
            inst_n.set_msir(new_state);
          }
        }

        break;
      }

      default:
        assert(!"Corrupt MSIR state value found.");
        abort();
    }
  }

  // This does not consumes prereqs, but produces new ones
  inline event_list do_reclaim(const data_place& memory_node, instance_id_t instance_id)
  {
    // Write back data on the host (presumably), but that could be to
    // some other data place such as the memory of a device

    /* previous tasks using this data instance */;
    event_list result = get_pending_done_prereqs(memory_node);

    reclaim_update_state(memory_node, instance_id, result);

    auto& inst = get_data_instance(instance_id);

    deallocate(memory_node, instance_id, inst.get_extra_args(), result);

    inst.set_allocated(false);

    release_instance_id(instance_id);

    return result;
  }

  /* Returns a prereq with all pending tasks accessing this data instance */
  event_list get_pending_done_prereqs(instance_id_t instance_id)
  {
    const auto& i = get_data_instance(instance_id);
    if (!i.has_last_task_relaxed())
    {
      return event_list();
    }

    return i.get_last_task_relaxed().get_done_prereqs();
  }

  /* Returns a prereq with all pending tasks accessing this piece of data on the specified memory node */
  // TODO modify to take a logical_data_untyped::state_t
  event_list get_pending_done_prereqs(const data_place&)
  {
    auto prereqs = event_list();
    auto& state_ = get_state();

    if (state_.current_mode == access_mode::write)
    {
      if (state_.current_writer.has_value())
      {
        prereqs = state_.current_writer->get_done_prereqs();
      }
    }
    else
    {
      prereqs.merge(state_.current_readers.get_done_prereqs());

      if (state_.previous_writer.has_value())
      {
        prereqs.merge(state_.previous_writer->get_done_prereqs());
      }
    }

    return prereqs;
  }

  // prereqs is used to record which prereqs are expected before using a
  // piece of data.
  //
  // Returns prereq_out
  void enforce_msi_protocol(instance_id_t instance_id, access_mode mode, event_list& prereqs)
  {
    // print("BEFORE");

    auto& current_instance = get_data_instance(instance_id);
    auto current_msir      = current_instance.get_msir();

    /* Update msir_statuses depending on the current states and the required access mode */
    switch (mode)
    {
      case access_mode::read:
        switch (current_msir)
        {
          case reserved::msir_state_id::modified:
          case reserved::msir_state_id::shared:
            /* no-op : just reuse previous reqs */
            prereqs.merge(current_instance.get_read_prereq());
            break;

          case reserved::msir_state_id::invalid: {
            /* There must be at least a valid copy at another place, so this becomes shared */

            // There is no local valid copy ... find one !
            instance_id_t dst_instance_id = instance_id;
            auto& dst_instance            = get_data_instance(dst_instance_id);
            auto dst_dplace               = dst_instance.get_dplace();
            auto dst_memory_node          = dst_dplace;

            instance_id_t src_instance_id = find_source_instance_id(dst_instance_id);
            auto& src_instance            = get_data_instance(src_instance_id);
            const auto src_dplace         = src_instance.get_dplace();
            const auto src_memory_node    = src_dplace;

            // Initiate the copy once src and dst are ready
            auto src_avail_prereq = src_instance.get_read_prereq();
            src_avail_prereq.merge(dst_instance.get_read_prereq(), dst_instance.get_write_prereq(), prereqs);

            data_copy(dst_memory_node, dst_instance_id, src_memory_node, src_instance_id, src_avail_prereq);

            // Make sure this is finished before we delete the source, for example
            // We do not remove existing prereqs as there can be concurrent copies along with existing read accesses
            src_instance.add_write_prereq(ctx, src_avail_prereq);
            dst_instance.set_read_prereq(src_avail_prereq);
            // Everything is already in the read_prereq
            dst_instance.clear_write_prereq();

            src_avail_prereq.clear();

            // This is our output
            prereqs.merge(dst_instance.get_read_prereq());

            /*
             * Update MSI statuses
             */
            dst_instance.set_msir(reserved::msir_state_id::shared);

            /* If there was a single copy, they are turned into shared if they were invalid */
            for (auto& inst : used_instances)
            {
              if (inst.get_msir() != reserved::msir_state_id::invalid)
              {
                inst.set_msir(reserved::msir_state_id::shared);
              }
            }
          }
          break;

          case reserved::msir_state_id::reduction:
            // This is where we should reconstruct the data ?

            // Invalidate all other existing copies but that one
            for (auto& inst : used_instances)
            {
              inst.set_msir(reserved::msir_state_id::invalid);
            }

            get_data_instance(instance_id).set_msir(reserved::msir_state_id::modified);

            break;

          default:
            assert(!"Corrupt MSIR state detected.");
            abort();
        }

        break;

      case access_mode::rw:
      case access_mode::write:
      case access_mode::reduce_no_init:
      case access_mode::reduce:
        switch (current_msir)
        {
          case reserved::msir_state_id::modified:
            /* no-op */
            prereqs.merge(current_instance.get_read_prereq(), current_instance.get_write_prereq());
            break;

          case reserved::msir_state_id::shared:
            // There is a local copy, but we need to invalidate others
            prereqs.merge(current_instance.get_read_prereq(), current_instance.get_write_prereq());

            for (size_t i = 0; i < used_instances.size(); i++)
            {
              // All other instances become invalid, and their content
              // need not be preserved
              if (i != size_t(instance_id))
              {
                auto& inst_i = get_data_instance(instance_id_t(i));
                inst_i.set_msir(reserved::msir_state_id::invalid);
                inst_i.clear_read_prereq();
                inst_i.clear_write_prereq();
              }
            }

            current_instance.set_msir(reserved::msir_state_id::modified);
            break;

          case reserved::msir_state_id::invalid: {
            // If we need to perform a copy, this will be the source instance
            instance_id_t src_instance_id = instance_id_t::invalid;
            // Do not find a source if this is write only
            if (mode == access_mode::rw || mode == access_mode::reduce_no_init)
            {
              // There is no local valid copy ... find one !
              instance_id_t dst_instance_id = instance_id;
              auto dst_dplace               = get_instance_dplace(dst_instance_id);
              const auto dst_memory_node    = dst_dplace;

              src_instance_id            = find_source_instance_id(dst_instance_id);
              auto src_dplace            = get_instance_dplace(src_instance_id);
              const auto src_memory_node = src_dplace;

              // Initiate the copy once src and dst are ready
              auto src_avail_prereq = get_data_instance(src_instance_id).get_read_prereq();
              src_avail_prereq.merge(get_data_instance(dst_instance_id).get_read_prereq(),
                                     get_data_instance(dst_instance_id).get_write_prereq(),
                                     prereqs);

              data_copy(dst_memory_node, dst_instance_id, src_memory_node, src_instance_id, src_avail_prereq);

              // Make sure this is finished before we delete the source, for example
              // We remove previous prereqs since this data is normally only used for this copy, and invalidated
              // then
              /* TODO CHECK THIS ... being too conservative ? */
              // THIS INSTEAD  ?get_data_instance(src_instance_id).set_write_prereq(dst_copied_prereq);
              get_data_instance(src_instance_id).set_read_prereq(src_avail_prereq);
              get_data_instance(dst_instance_id).set_read_prereq(src_avail_prereq);

              src_avail_prereq.clear();

              // This is our output
              prereqs.merge(get_data_instance(dst_instance_id).get_read_prereq());
            }
            else
            {
              // Write only
              assert(mode == access_mode::write || mode == access_mode::reduce);
            }

            // Clear and all copies which become invalid, keep prereqs
            // needed to perform the copy
            for (size_t i = 0; i < used_instances.size(); i++)
            {
              auto& inst_i = get_data_instance(instance_id_t(i));
              if (i != size_t(instance_id) && i != size_t(src_instance_id))
              {
                /* We do not clear write prereqs */
                inst_i.clear_read_prereq();
              }

              // including instance_id, but we will set it to modified after
              inst_i.set_msir(reserved::msir_state_id::invalid);
            }

            // This is the only valid instance now
            current_instance.set_msir(reserved::msir_state_id::modified);
            break;
          }

          default:
            assert(!"Corrupt MSIR state detected.");
            abort();
        }

        break;

      case access_mode::relaxed:
        current_instance.set_msir(reserved::msir_state_id::reduction);
        break;

      default:
        assert(!"Corrupt MSIR state detected.");
        abort();
    }
  }

  auto& get_mutex()
  {
    return mutex;
  }

private:
  ::std::mutex mutex;
};

} // namespace reserved

/** @brief Base class of all `logical_data<T>` types. It does not "know" the type of the data, so most of the time it's
 * best to use `logical_data<T>`. Use `logical_data_untyped` only in special circumstances.
 */
class logical_data_untyped
{
public:
  ///@{
  /** @name Constructors */
  logical_data_untyped() = default;

  logical_data_untyped(::std::shared_ptr<reserved::logical_data_untyped_impl> p)
      : pimpl(mv(p))
  {}

  /**
   * @brief Constructs a new `logical_data_untyped object` with the provided context, backend, memory_node, symbol,
   * and data prerequisites.
   *
   * @param ctx A context
   * @param backend A `shared_ptr` to the `data_interface` object underlying the data, must be non-null
   * @param memory_node initial data location
   *
   * The constructor initializes `this` with the provided context, backend, data place, and
   * symbol. If `memory_node` is invalid, the constructor returns early without creating a data instance. Otherwise,
   * it sets up the data instance, pins the host memory if required, and initializes the MSI status and prerequisite
   * events.
   */
  logical_data_untyped(backend_ctx_untyped ctx, ::std::shared_ptr<data_interface> backend, const data_place& memory_node)
      : pimpl(::std::make_shared<reserved::logical_data_untyped_impl>(mv(ctx), mv(backend), memory_node))
  {}
  ///@}

  ///@{ @name Symbol getter/setter
  const ::std::string& get_symbol() const
  {
    return pimpl->get_symbol();
  }
  void set_symbol(::std::string str)
  {
    pimpl->symbol = mv(str);
  }
  ///@}

  ///@{ @name allocator setter
  void set_allocator(block_allocator_untyped a)
  {
    pimpl->set_allocator(mv(a));
  }
  ///@}

  ///@{ @name Data interface getter
  const data_interface& get_data_interface() const
  {
    assert(has_interface());
    return *pimpl->dinterface;
  }
  ///@}

  /** @name Get common data associated with this object. The type must be user-specified and is checked dynamically.
   * @tparam T the type of common data, must be user-specified
   */
  template <typename T>
  const T& common() const
  {
    assert(has_interface());
    return pimpl->dinterface->shape<T>();
  }

  ///@{
  /**
   * @name Retrieves the physical data from this logical data for a given instance id
   *
   * @tparam T Type stored in this object (must be supplied by user)
   * @param instance_id ID of the instance for which the slice is fetched
   * @return T Object represented by this handle
   *
   * The user-provided type is checked dynamically. Program is aborted if there's a mismatch.
   */
  template <typename T>
  decltype(auto) instance(instance_id_t instance_id)
  {
    assert(has_interface());
    return pimpl->dinterface->instance<T>(instance_id);
  }

  template <typename T>
  decltype(auto) instance(instance_id_t instance_id) const
  {
    assert(has_interface());
    return pimpl->dinterface->instance<T>(instance_id);
  }
  ///@}

  ///@{
  /**
   * @name Retrieves the physical data from this logical data for the default instance
   */
  template <typename T>
  decltype(auto) instance(task& tp)
  {
    return instance<T>(pimpl->dinterface->get_default_instance_id(pimpl->ctx, *this, tp));
  }

  template <typename T>
  decltype(auto) instance(task& tp) const
  {
    return instance<T>(pimpl->dinterface->get_default_instance_id(pimpl->ctx, *this, tp));
  }
  ///@}

  void freeze(access_mode freeze_mode, data_place place = data_place::invalid())
  {
    pimpl->freeze(freeze_mode, place);
  }

  template <typename T>
  ::std::pair<T, event_list> get_frozen(task& fake_task, const data_place& dplace, access_mode m)
  {
    return pimpl->template get_frozen<T>(fake_task, dplace, m);
  }

  void unfreeze(task& fake_task, event_list prereqs = event_list())
  {
    pimpl->unfreeze(fake_task, mv(prereqs));
  }

  void set_automatic_unfreeze(task& unfreeze_fake_task_, bool flag)
  {
    pimpl->set_automatic_unfreeze(unfreeze_fake_task_, flag);
  }

  /**
   * @brief Indicates whether a logical data is frozen or not, and what is the corresponding mode
   *
   */
  ::std::pair<bool, access_mode> is_frozen() const
  {
    return pimpl->is_frozen();
  }

  /**
   * @brief Allocate memory for this logical data
   *
   * @param ctx
   * @param memory_node
   * @param instance_id
   * @param s
   * @param extra_args
   * @param prereqs
   */
  void allocate(const data_place& memory_node,
                instance_id_t instance_id,
                ::std::ptrdiff_t& s,
                void** extra_args,
                event_list& prereqs)
  {
    pimpl->allocate(memory_node, instance_id, s, extra_args, prereqs);
  }

  /**
   * @brief Deallocate memory previously allocated with `allocate`
   *
   * @param ctx
   * @param memory_node
   * @param instance_id
   * @param extra_args
   * @param prereqs
   */
  void deallocate(const data_place& memory_node, instance_id_t instance_id, void* extra_args, event_list& prereqs)
  {
    pimpl->deallocate(memory_node, instance_id, extra_args, prereqs);
  }

  /**
   * @brief Copy data
   *
   * @param ctx
   * @param dst_node
   * @param dst_instance_id
   * @param src_node
   * @param src_instance_id
   * @param arg
   * @param prereqs
   */
  void data_copy(const data_place& dst_node,
                 instance_id_t dst_instance_id,
                 const data_place& src_node,
                 instance_id_t src_instance_id,
                 event_list& prereqs)
  {
    pimpl->data_copy(dst_node, dst_instance_id, src_node, src_instance_id, prereqs);
  }

  /**
   * @brief Writes back data
   *
   * @param ctx
   * @param src_node
   * @param instance_id
   * @param prereqs
   */
  void write_back(const data_place& src_node, instance_id_t instance_id, event_list& prereqs)
  {
    pimpl->write_back(src_node, instance_id, prereqs);
  }

  /**
   * @brief Get the index of the first available instance_id
   *
   * @param dplace
   * @return The found instance_id
   */
  instance_id_t find_unused_instance_id(const data_place& dplace)
  {
    return pimpl->find_unused_instance_id(dplace);
  }

  /**
   * @brief Get the index of the first used instance_id that matches the data place, or get a new one
   *
   * @param dplace
   * @return The found instance_id
   */
  instance_id_t find_instance_id(const data_place& dplace)
  {
    return pimpl->find_instance_id(dplace);
  }

  /**
   * @brief Make it possible to reuse an id
   *
   * @param instance_id
   */
  void release_instance_id(instance_id_t instance_id)
  {
    pimpl->release_instance_id(instance_id);
  }

  /**
   * @brief Get the data pace associated with a given instance
   *
   * @param instance_id
   * @return const data_place&
   */
  const data_place& get_instance_dplace(instance_id_t instance_id) const
  {
    assert(pimpl);
    return pimpl->get_instance_dplace(instance_id);
  }

  ///@{
  /**
   * @name Get the data pace associated with a given instance
   */
  const data_instance& get_data_instance(instance_id_t instance_id) const
  {
    assert(pimpl);
    return pimpl->get_data_instance(instance_id);
  }
  data_instance& get_data_instance(instance_id_t instance_id)
  {
    return pimpl->get_data_instance(instance_id);
  }
  ///@}

  ///@{ @name Reference count query and manipulation
  bool has_ref() const
  {
    return pimpl->has_ref();
  }
  void add_ref()
  {
    pimpl->add_ref();
  }
  void remove_ref()
  {
    pimpl->remove_ref();
  }
  ///@}

  /**
   * @name Returns a dependency object for read/write/read and write access to this logical data.
   *
   * @return task_dep_untyped The dependency object corresponding to this logical data
   */
  ///@{
  task_dep_untyped read(data_place dp = data_place::affine())
  {
    return task_dep_untyped(*this, access_mode::read, mv(dp));
  }

  task_dep_untyped write(data_place dp = data_place::affine())
  {
    return task_dep_untyped(*this, access_mode::write, mv(dp));
  }

  task_dep_untyped rw(data_place dp = data_place::affine())
  {
    return task_dep_untyped(*this, access_mode::rw, mv(dp));
  }

  task_dep_untyped relaxed(::std::shared_ptr<reduction_operator_base> op, data_place dp = data_place::affine())
  {
    return task_dep_untyped(*this, access_mode::relaxed, mv(dp), op);
  }

  ///@}

  /**
   * @brief Returns true if the data interface of this object has been set
   */
  bool has_interface() const
  {
    assert(pimpl);
    return pimpl->dinterface != nullptr;
  }

  /**
   * @brief Returns true if the data is a void data interface
   */
  bool is_void_interface() const
  {
    assert(pimpl);
    return pimpl->is_void_interface();
  }

  // This function applies the reduction operator over 2 instances, the one
  // identified by "in_instance_id" is not modified, the one identified as
  // "inout_instance_id" is where the result is put.
  // This method assumes that instances are properly allocated, but ignores
  // the MSI state which is managed at other places.
  // The reduction occurs on the execution place e_place, and we assume that
  // both instances are located at the same data place.
  void apply_redux_op(const data_place& memory_node,
                      const exec_place& e_place,
                      instance_id_t inout_instance_id,
                      instance_id_t in_instance_id,
                      event_list& prereqs)
  {
    //        fprintf(stderr, "APPLY REDUX op(in: %d -> inout %d)\n", in_instance_id, inout_instance_id);

    assert(in_instance_id != inout_instance_id);

    auto& inst_in    = get_data_instance(in_instance_id);
    auto& inst_inout = get_data_instance(inout_instance_id);

    assert(inst_in.is_allocated());
    assert(inst_inout.is_allocated());

    // In addition to existing prerequisite, we must make sure all pending operations are done
    prereqs.merge(inst_in.get_read_prereq(), inst_inout.get_read_prereq(), inst_inout.get_write_prereq());

    // Apply reduction operator
    ::std::shared_ptr<reduction_operator_base> ops_in    = get_data_instance(in_instance_id).get_redux_op();
    ::std::shared_ptr<reduction_operator_base> ops_inout = get_data_instance(inout_instance_id).get_redux_op();
    assert(ops_in != nullptr || ops_inout != nullptr);

    ::std::shared_ptr<reduction_operator_base> ops = (ops_in != nullptr) ? ops_in : ops_inout;

    ops->op_untyped(*this, memory_node, inout_instance_id, memory_node, in_instance_id, e_place, prereqs);

    // Both instances now depend on the completion of this operation
    inst_in.set_read_prereq(prereqs);
    inst_inout.set_read_prereq(prereqs);
  }

  // Perform reductions within a node over a set of instances
  // The last item of the list will contain the reduction result
  void apply_redux_on_node(
    const data_place& memory_node, const exec_place& e_place, ::std::vector<instance_id_t>& ids, event_list& prereqs)
  {
    if (ids.size() < 2)
    {
      return;
    }

    // Apply reduction operator over all instances (there are at least two)
    for (size_t i = 1; i < ids.size(); i++)
    {
      auto in_id    = ids[i - 1];
      auto inout_id = ids[i];
      apply_redux_op(memory_node, e_place, inout_id, in_id, prereqs);
    }
  }

  // Asynchronously reconstruct a piece of data into the data instance
  // identified by instance_id. This may require to perform reduction on
  // differement data places, to perform data transfers, and to reduce these
  // temporary results as well.
  // If we have a non relaxed type of access after a reduction
  // instance_id is the data instance which should have a coherent copy after the reduction
  void reconstruct_after_redux(
    backend_ctx_untyped& bctx, instance_id_t instance_id, const exec_place& e_place, event_list& prereqs)
  {
    // @@@@TODO@@@@ get from somewhere else (machine ?)
    const size_t max_nodes = cuda_try<cudaGetDeviceCount>() + 2;
    ::std::vector<::std::vector<instance_id_t>> per_node(max_nodes);

    // Valid copies, we will select only one
    ::std::vector<instance_id_t> ref_copies;

    // These are instances on the target memory node which we can use as
    // temporary storage (ie. to receive copies)
    ::std::vector<instance_id_t> available_instances;

    auto& target_instance            = get_data_instance(instance_id);
    auto& target_dplace              = target_instance.get_dplace();
    auto target_memory_node          = target_dplace;
    instance_id_t target_instance_id = instance_id;

    //        fprintf(stderr, "Start to plan reduction of %p on instance %d (memory_node %d)\n", d, instance_id,
    //                e_place->memory_node);

    // ALL is serialized for now ...
    const auto nnodes = instance_id_t(get_data_instance_count());

#ifdef REDUCTION_DEBUG
    fprintf(stderr, "make_reduction_plan :: target instance %d\n", target_instance_id);
    for (size_t i = 0; i < nnodes; i++)
    {
      auto& instance_i = get_data_instance(i);
      fprintf(stderr,
              "make_reduction_plan :: instance %d : ops %p dplace %d\n",
              i,
              instance_i.get_redux_op().get(),
              int(instance_i.get_dplace()));
    }
#endif

    // We go through all data instances
    for (auto i : each(nnodes))
    {
      auto& instance_i = get_data_instance(i);

#ifdef REDUCTION_DEBUG
      fprintf(stderr,
              "make_reduction_plan :: instance %d status %s\n",
              i,
              reserved::status_to_string(instance_i.get_msir()).c_str());
#endif

      // We exclude the target instance id because we want to reduce to this id, not from this id
      if (i != instance_id && instance_i.get_msir() == reserved::msir_state_id::reduction)
      {
        const data_place& dplace = instance_i.get_dplace();
        per_node[to_index(dplace)].push_back(i);
        // fprintf(stderr, "instance %d : get_msir() == reserved::msir_state_id::reduction memory_node = %d\n", i,
        //        memory_node);
        continue;
      }

      // If this is a valid copy, we put this aside and will select one of these
      if (instance_i.get_msir() == reserved::msir_state_id::modified
          || instance_i.get_msir() == reserved::msir_state_id::shared)
      {
        ref_copies.push_back(i);
        continue;
      }

      // If this is an invalid copy, and that it is on the target memory node, we can use this as a temporary
      // storage
      if (instance_i.get_msir() == reserved::msir_state_id::invalid && instance_i.get_dplace() == target_memory_node)
      {
        available_instances.push_back(i);
        continue;
      }
    }

#ifdef REDUCTION_DEBUG
    for (auto r : ref_copies)
    {
      fprintf(stderr, "make_reduction_plan :: ref copy %d\n", r);
    }
#endif

    // Add "the" reference copy (if any) to the list of items to reduce
    if (ref_copies.size() > 0)
    {
      // TODO : we should sort ref_copies to avoid transfers if possible
      auto ref_instance_id = ref_copies[0];

      auto& ref_instance         = get_data_instance(ref_instance_id);
      const auto ref_memory_node = ref_instance.get_dplace();

#ifdef REDUCTION_DEBUG
      fprintf(stderr, "Using %d as the reference\n", ref_instance_id);
#endif

      // The target instance id is implicitly the one where to reduce last, so we don't add it
      if (ref_instance_id != target_instance_id)
      {
        // fprintf(stderr, "...adding %d to per_node[%d]\n", ref_instance_id, ref_memory_node);
        per_node[to_index(ref_memory_node)].push_back(ref_instance_id);
      }

      // Consider whether remaining copies can be used as temporary storage
      // A copy that is not used as "the" reference copy, but which is on
      // the target memory node, can be used
      if (ref_copies.size() > 1)
      {
        for (auto i : each(1, ref_copies.size()))
        {
          auto& inst = get_data_instance(instance_id_t(i));
          if (inst.get_dplace() == target_memory_node)
          {
            available_instances.push_back(ref_copies[i]);
          }
        }
      }
    }

#ifdef REDUCTION_DEBUG
    for (auto a : available_instances)
    {
      fprintf(stderr, "make_reduction_plan :: available instance %d (can be overwritten)\n", a);
    }
#endif

    // First start by invalidating all data instances !
    for (auto i : each(nnodes))
    {
      get_data_instance(i).set_msir(reserved::msir_state_id::invalid);
    }

    // Naive plan : reduce all local instances (except on target node), copy to target node, reduce local instances
    // on target node
    for (auto n : each(max_nodes))
    {
#ifdef REDUCTION_DEBUG
      fprintf(stderr, "make_reduction_plan :: node %d per_node size %d\n", n, per_node[n].size());
#endif
      // Skip if there is nothing to do on that memory node
      size_t per_node_size = per_node[n].size();
      if (per_node_size == 0)
      {
        continue;
      }

      // TODO THIS MAY BE A BUG: do we care about managed devices or host?
      const auto memory_node = data_place::device(static_cast<int>(n - 2));
      // Skip the target memory node in this step
      if (memory_node == target_memory_node)
      {
        continue;
      }

      // fprintf(stderr, "SET EXEC CTX associated to MEMORY NODE %d \n", n);

      // We now ensure that the current execution place is appropriate to
      // manipulate data on that memory node (data place). This will
      // typically change the current device if needed, or select an
      // appropriate affinity mask.

      exec_place e_place_n = memory_node.get_affine_exec_place();

      auto saved_place = e_place_n.activate(pimpl->ctx);

      // Reduce instances if there are more than one
      if (per_node[n].size() > 1)
      {
        // Apply reduction operator over all instances on node n (there are at least two)
#ifdef REDUCTION_DEBUG
        fprintf(stderr, "apply_redux_on_node %d\n", int(memory_node));
#endif
        apply_redux_on_node(memory_node, e_place_n, per_node[n], prereqs);
      }

      instance_id_t copy_instance_id;

      // We first try to get an available instance
      if (available_instances.size() > 0)
      {
        copy_instance_id = available_instances.back();
        available_instances.pop_back();

        // This instance will be used, and we add the current list of events to its existing one
        auto& inst = get_data_instance(copy_instance_id);
        inst.add_read_prereq(bctx, prereqs);

        // fprintf(stderr, "REUSE INSTANCE %d to copy\n", copy_instance_id);
#ifdef REDUCTION_DEBUG
        fprintf(stderr, "make_reduction_plan :: reuse instance %d for copy\n", copy_instance_id);
#endif
      }
      else
      {
        // There was no available instance, so we allocate a new one, and assign a new instance id
        copy_instance_id = find_unused_instance_id(target_dplace);
        // fprintf(stderr, "RESERVE ID %d on node %d\n", copy_instance_id, n);

#ifdef REDUCTION_DEBUG
        fprintf(stderr, "make_reduction_plan :: find_unused_instance_id => %d\n", copy_instance_id);
#endif
      }

      auto& copy_inst = get_data_instance(copy_instance_id);
      if (!copy_inst.is_allocated())
      {
        // Allocate an instance on the memory node (data place)
        // fprintf(stderr, "ALLOCATE ID %d on node %d\n", copy_instance_id, n);

        // mode is rather meaningless here (?)
        // fprintf(stderr, "ALLOCATE ID %d on node %d\n", copy_instance_id, int(target_memory_node));
        reserved::dep_allocate(
          pimpl->ctx, *this, access_mode::read, target_memory_node, e_place_n, copy_instance_id, prereqs);
      }

      // Copy the last instance to the destination
      instance_id_t src_instance_id = per_node[n].back();
      // fprintf(stderr, "COPY id %d (node %d) => id %d (node %d)\n", src_instance_id, n, copy_instance_id,
      //        target_memory_node);

      assert(get_data_instance(src_instance_id).is_allocated());
      assert(get_data_instance(copy_instance_id).is_allocated());
      // fprintf(stderr, "REDUCTION : copy id %d to id %d\n", src_instance_id, copy_instance_id);
      data_copy(target_memory_node, copy_instance_id, memory_node, src_instance_id, prereqs);

      // The copied instance will use the same operator as the source
      auto ops = get_data_instance(src_instance_id).get_redux_op();
      if (ops.get())
      {
        // fprintf(stderr, "SET REDUX OP for id %d (use that of id %d)\n", copy_instance_id, src_instance_id);
        get_data_instance(copy_instance_id).set_redux_op(ops);
      }

      // Add this instance to the list of instances on the target node
      // If this is the reference id, we don't add it because it will be implicitly the last element where to
      // reduce !
      if (copy_instance_id != target_instance_id)
      {
        //    fprintf(stderr, "ADD instance %d to per_node[%d]\n", copy_instance_id, target_memory_node);
        per_node[to_index(target_memory_node)].push_back(copy_instance_id);
      }

      // Restore the execution place to its previous state (e.g. current CUDA device)
      // fprintf(stderr, "RESET CTX\n");
      e_place_n.deactivate(pimpl->ctx, saved_place);
    }

    if (per_node[to_index(target_memory_node)].size() > 1)
    {
      // Reduce all instances on the target node, including temporary ones
      apply_redux_on_node(target_memory_node, e_place, per_node[to_index(target_memory_node)], prereqs);
    }

    // Reduce to the target id, unless it was obtained from a copy
    if (per_node[to_index(target_memory_node)].size() > 0)
    {
      instance_id_t in_id    = per_node[to_index(target_memory_node)].back();
      instance_id_t inout_id = target_instance_id;

      // fprintf(stderr, "REDUCE TO TARGET ID=%d....\n", target_instance_id);
      // fprintf(stderr, "\top(in: %d -> inout %d)\n", in_id, inout_id);

      // It is possible (likely) that the target id was not used for a
      // reduction, so we use the same operator as the other data
      // instance which we are reducing with
      auto ops = get_data_instance(in_id).get_redux_op();
      if (ops.get())
      {
        get_data_instance(inout_id).set_redux_op(ops);
      }

      apply_redux_op(target_memory_node, e_place, inout_id, in_id, prereqs);
    }

    // This is likely not optimal : but we assume the only valid copy is
    // the last one. Note that in some situations, we may have other valid
    // copies if we did copy the reduced data, and did not modify it again
    get_data_instance(target_instance_id).set_msir(reserved::msir_state_id::modified);
  }

  size_t get_data_instance_count() const
  {
    return pimpl->get_data_instance_count();
  }

  // prereqs is used to record which prereqs are expected before using a
  // piece of data.
  //
  // Returns prereq_out
  void enforce_msi_protocol(instance_id_t instance_id, access_mode mode, event_list& prereqs)
  {
    pimpl->enforce_msi_protocol(instance_id, mode, prereqs);
  }

  // Find a valid copy given a hint and return its instance id. Currently the hint parameter is not used.
  instance_id_t find_source_instance_id(instance_id_t dst_instance_id)
  {
    return pimpl->find_source_instance_id(dst_instance_id);
  }

  size_t hash() const
  {
    return pimpl->hash();
  }

  // Enable or disable write-back. Enabling write-back will cause an error if there is no reference data instance.
  void set_write_back(bool flag)
  {
    pimpl->set_write_back(flag);
  }

  reserved::logical_data_state& get_state()
  {
    return pimpl->get_state();
  }

  auto& get_ctx() const
  {
    return pimpl->ctx;
  }

  bool is_initialized() const
  {
    return pimpl.get() != nullptr;
  }

  bool operator==(const logical_data_untyped& other) const
  {
    return pimpl == other.pimpl;
  }

  friend inline ::std::shared_ptr<void> pack_state(const logical_data_untyped& d)
  {
    return d.pimpl;
  }

  friend inline logical_data_untyped unpack_state(const ::std::shared_ptr<void>& p)
  {
    assert(p);
    return logical_data_untyped(::std::static_pointer_cast<reserved::logical_data_untyped_impl>(p));
  }

  auto& get_mutex()
  {
    return pimpl->get_mutex();
  }

private:
  int get_unique_id() const
  {
    return pimpl->get_unique_id();
  }

  ::std::shared_ptr<reserved::logical_data_untyped_impl> pimpl;
};

// This implementation is deferred because we need the logical_data_untyped type in it
inline void reserved::logical_data_untyped_impl::erase()
{
  // fprintf(stderr, "ERASING ... %s - get_unique_id() %d was_erased %d\n", get_symbol().c_str(),
  // int(get_unique_id()),
  //         was_erased ? 1 : 0);

  // Early exit if:
  if (get_unique_id() == -1 // this logical_data_untyped was moved to another logical_data_untyped
                            // class which will take care of the necessary cleanups later on, OR
      || !ctx // the context was already null, OR
      || was_erased)
  { // the logical_data_untyped was already erased
    return;
  }

  if (frozen_flag)
  {
    if (!automatic_unfreeze)
    {
      fprintf(stderr, "Error: destroying frozen logical data without unfreeze and no automatic unfreeze\n");
      abort();
    }

    // Freeze data automatically : we assume all dependencies on that
    // frozen data are solved by other means (this is the requirement of
    // the set_automatic_unfreeze API)
    assert(unfreeze_fake_task.has_value());
    unfreeze(unfreeze_fake_task.value(), event_list());
  }

  auto& ctx_st = ctx.get_state();

  auto wb_prereqs = event_list();
  auto& h_state   = get_state();

  const bool track_dangling_events = ctx.track_dangling_events();

  /* If there is a reference instance id, it needs to be updated with a
   * valid copy if that is not the case yet */
  if (enable_write_back && !is_void_interface())
  {
    instance_id_t ref_id = reference_instance_id;
    assert(ref_id != instance_id_t::invalid);

    // Get the state in which we store previous writer, readers, ...
    if (h_state.current_mode == access_mode::relaxed)
    {
      // Reconstruction of the data on the reference data place needed

      // We create a logical_data from a pointer to an implementation
      logical_data_untyped l(shared_from_this());

      data_instance& ref_instance  = get_data_instance(ref_id);
      const data_place& ref_dplace = ref_instance.get_dplace();
      auto e                       = ref_dplace.get_affine_exec_place();
      l.reconstruct_after_redux(ctx, ref_id, e, wb_prereqs);

      h_state.current_mode = access_mode::none;
    }

    auto s = used_instances[size_t(ref_id)].get_msir();
    if (s == reserved::msir_state_id::invalid)
    {
      // Write-back needed
      // fprintf(stderr, "Write-back needed %s\n", get_symbol().c_str());

      // Look where to take the valid copy from
      instance_id_t src_id         = find_source_instance_id(ref_id);
      data_instance& src_instance  = get_data_instance(src_id);
      const data_place& src_dplace = src_instance.get_dplace();

      data_instance& dst_instance = get_data_instance(ref_id);

      // Initiate the copy once src and dst are ready
      auto reqs = src_instance.get_read_prereq();
      reqs.merge(dst_instance.get_read_prereq(), dst_instance.get_write_prereq());

      write_back(src_dplace, src_id, reqs);

      src_instance.add_write_prereq(ctx, reqs);
      dst_instance.set_read_prereq(reqs);

      if (track_dangling_events)
      {
        // nobody waits for these events, so we put them in the list of dangling events
        ctx_st.add_dangling_events(ctx, reqs);
      }
    }
  }

  for (auto i : each(instance_id_t(used_instances.size())))
  {
    auto& inst_i = used_instances[size_t(i)];
    // Unpin an instance if if was automatically pinned
    if (inst_i.automatically_pinned)
    {
      _CCCL_ASSERT(inst_i.get_dplace().is_host(), "");
      _CCCL_ASSERT(dinterface, "");
      dinterface->unpin_host_memory(i);
    }

    if (inst_i.is_allocated() && inst_i.reclaimable)
    {
      // Make sure copies or reduction initiated by the erase are finished
      auto inst_prereqs = wb_prereqs;
      // Wait for preceding tasks
      inst_prereqs.merge(get_pending_done_prereqs(inst_i.get_dplace()));

      inst_prereqs.merge(inst_i.get_read_prereq(), inst_i.get_write_prereq());

      // We now ask to deallocate that piece of data
      deallocate(inst_i.get_dplace(), i, inst_i.get_extra_args(), inst_prereqs);

      if (track_dangling_events)
      {
        ctx_st.add_dangling_events(ctx, inst_prereqs);
      }
    }

    inst_i.clear();
  }

  if (track_dangling_events)
  {
    if (wb_prereqs.size() > 0)
    {
      // nobody waits for these events, so we put them in the list of dangling events
      ctx_st.add_dangling_events(ctx, wb_prereqs);
    }
  }

  // Clear the state which may contain references (eg. shared_ptr) to other
  // resources. This must be done here because the destructor of the logical
  // data may be called after finalize()
  h_state.clear();

  ctx_st.logical_data_ids_mutex.lock();

  // This unique ID is not associated to a pointer anymore (and should never be reused !)
  //
  // This SHOULD be in the table because that piece of data was created
  // in this context and cannot already have been destroyed.
  auto erased = ctx_st.logical_data_ids.erase(get_unique_id());
  EXPECT(erased == 1UL, "ERROR: prematurely destroyed data");

  if (ctx_st.logical_data_stats_enabled)
  {
    ctx_st.previous_logical_data_stats.push_back(::std::make_pair(get_symbol(), dinterface->data_footprint()));
  }

  ctx_st.logical_data_ids_mutex.unlock();

  // Make sure this we do not erase this twice. For example after calling
  // finalize() there is no need to erase it again in the constructor
  was_erased = true;
}

namespace reserved
{

/**
 * @brief Implements STF dependencies.
 *
 * This method ensures that the current task (task) depends on the appropriate
 * predecessors. A second method enforce_stf_deps_after will be called to make
 * sure future tasks depend on that task.
 */
template <typename task_type>
inline event_list enforce_stf_deps_before(
  backend_ctx_untyped& bctx,
  logical_data_untyped& handle,
  const instance_id_t instance_id,
  const task_type& task,
  const access_mode mode,
  const ::std::optional<exec_place> eplace)
{
  auto result  = event_list();
  auto& ctx_st = bctx.get_state();
  // Get the context in which we store previous writer, readers, ...
  auto& ctx_ = handle.get_state();

  auto& dot                 = *bctx.get_dot();
  const bool dot_is_tracing = dot.is_tracing();

  if (mode == access_mode::relaxed)
  {
    // A reduction only needs to wait for previous accesses on the data instance
    ctx_.current_mode = access_mode::relaxed;

    if (dot_is_tracing)
    {
      // Add this task to the list of task accessing the logical data in relaxed mode
      // We only store its id since this is used for dot
      ctx_.pending_redux_id.push_back(task.get_unique_id());
    }

    // XXX with a mv we may avoid copies
    const auto& data_instance = handle.get_data_instance(instance_id);
    result.merge(data_instance.get_read_prereq(), data_instance.get_write_prereq());
    return result;
  }

  // This is not a reduction, but perhaps we need to reconstruct the data first?
  if (ctx_.current_mode == access_mode::relaxed)
  {
    assert(eplace.has_value());
    if (dot_is_tracing)
    {
      // Add a dependency between previous tasks accessing the handle
      // in relaxed mode, and this task which forces its
      // reconstruction.
      for (const int redux_task_id : ctx_.pending_redux_id)
      {
        dot.add_edge(redux_task_id, task.get_unique_id());
      }
      ctx_.pending_redux_id.clear();
    }
    handle.reconstruct_after_redux(bctx, instance_id, eplace.value(), result);
    ctx_.current_mode = access_mode::none;
  }

  // @@@TODO@@@ cleaner ...
  const bool write = (mode == access_mode::rw || mode == access_mode::write);

  // ::std::cout << "Notifying " << (write?"W":"R") << " access on " << get_symbol() << " by task " <<
  // task->get_symbol() << ::std::endl;
  if (write)
  {
    if (ctx_.current_mode == access_mode::write)
    {
      // Write after Write (WAW)
      assert(ctx_.current_writer.has_value());

      const auto& cw = ctx_.current_writer.value();
      result.merge(cw.get_done_prereqs());

      const auto cw_id = cw.get_unique_id();

      if (dot_is_tracing)
      {
        dot.add_edge(cw_id, task.get_unique_id());
      }

      ctx_st.leaves.remove(cw_id);

      // Replace previous writer
      ctx_.previous_writer = cw;
    }
    else
    {
      // Write after read

      // The writer depends on all current readers
      auto& current_readers = ctx_.current_readers;
      result.merge(current_readers.get_done_prereqs());

      for (const int reader_task_id : current_readers.get_ids())
      {
        if (dot_is_tracing)
        {
          dot.add_edge(reader_task_id, task.get_unique_id());
        }
        ctx_st.leaves.remove(reader_task_id);
      }

      current_readers.clear();
      ctx_.current_mode = access_mode::write;
    }
    // Note the task will later be set as the current writer
  }
  else
  {
    // This is a read access
    if (ctx_.current_mode == access_mode::write)
    {
      // Read after Write
      // Current writer becomes the previous writer, and all future readers will depend on this previous
      // writer
      assert(ctx_.current_writer.has_value());
      ctx_.previous_writer = mv(ctx_.current_writer);

      if (ctx_.previous_writer.has_value())
      {
        const auto& pw = ctx_.previous_writer.value();
        result.merge(pw.get_done_prereqs());

        const int pw_id = pw.get_unique_id();

        if (dot_is_tracing)
        {
          dot.add_edge(pw_id, task.get_unique_id());
        }

        ctx_st.leaves.remove(pw_id);
      }
      else
      {
        EXPECT(false, "Internal error: previous_writer must be set");
      }

      ctx_.current_mode = access_mode::none;
      // ::std::cout << "CHANGING to FALSE for " << symbol << ::std::endl;
    }
    else if (ctx_.previous_writer.has_value())
    {
      const auto& pw = ctx_.previous_writer;
      result.merge(pw->get_done_prereqs());

      const int pw_id = pw->get_unique_id();
      if (dot_is_tracing)
      {
        dot.add_edge(pw_id, task.get_unique_id());
      }

      ctx_st.leaves.remove(pw_id);
    }

    // Note : the task will later be added to the list of readers
  }

  return result;
}

template <typename task_type>
inline void enforce_stf_deps_after(
  backend_ctx_untyped& bctx, logical_data_untyped& handle, const task_type& task, const access_mode mode)
{
  if (mode == access_mode::relaxed)
  {
    // no further action is required
    return;
  }

  // Get the context in which we store previous writer, readers, ...
  auto& ctx_ = handle.get_state();

  if (mode == access_mode::rw || mode == access_mode::write)
  {
    ctx_.current_writer = task;
  }
  else
  {
    // Add to the list of readers
    ctx_.current_readers.add(bctx, task);
  }
}

/* Enforce task dependencies, allocations, and copies ... */
inline void fetch_data(
  backend_ctx_untyped& bctx,
  logical_data_untyped& d,
  const instance_id_t instance_id,
  task& t,
  access_mode mode,
  const ::std::optional<exec_place> eplace,
  const data_place& dplace,
  event_list& result)
{
  event_list stf_prereq = reserved::enforce_stf_deps_before(bctx, d, instance_id, t, mode, eplace);

  if (d.has_interface() && !d.is_void_interface())
  {
    // Allocate data if needed (and possibly reclaim memory to do so)
    reserved::dep_allocate(bctx, d, mode, dplace, eplace, instance_id, stf_prereq);

    /*
     * DATA LAZY UPDATE (relying on the MSI protocol)
     */

    // This will initiate a copy if the data was not valid
    d.enforce_msi_protocol(instance_id, mode, stf_prereq);
  }

  stf_prereq.optimize(bctx);

  // Gather all prereqs required to fetch this piece of data into the
  // dependencies of the task.
  // Even temporary allocation may require to enforce dependencies
  // because we are reclaiming data for instance.
  result.merge(mv(stf_prereq));
}

}; // namespace reserved

// This implementation is deferred because we need the logical_data_untyped type in it
template <typename T>
::std::pair<T, event_list>
reserved::logical_data_untyped_impl::get_frozen(task& fake_task, const data_place& dplace, access_mode m)
{
  event_list prereqs;

  // This will pick an instance id, either already valid or which needs to be allocated/populated
  auto id = find_instance_id(dplace);

  // Get the logical_data_untyped from the current impl
  logical_data_untyped d(shared_from_this());

  // Exactly like a task that would fetch a piece of data, this introduce
  // appropriate dependencies with previous readers/writers (or frozen data
  // deps !). Then, if the data wasn't available on the data place, it can be
  // allocated and a copy from a valid source can be made.
  // This will also update the MSI states of the logical data instances.
  reserved::fetch_data(ctx, d, id, fake_task, m, ::std::nullopt, dplace, prereqs);

  // Make sure we now have a valid copy (unless this is a token, because
  // fetch_data will only enforce dependencies and will not move or allocate
  // data)
  if constexpr (!::std::is_same_v<T, void_interface>)
  {
    assert(used_instances[int(id)].is_allocated());
    assert(used_instances[int(id)].get_msir() != reserved::msir_state_id::invalid);
  }

  return ::std::pair<T, event_list>(dinterface->instance<T>(id), mv(prereqs));
}

// This implementation is deferred because we need the logical_data_untyped type in it
inline void reserved::logical_data_untyped_impl::unfreeze(task& fake_task, event_list prereqs)
{
  assert(frozen_flag);

  // Unlike regular tasks, unfreeze may affects multiple data instances, so
  // we do not reuse the same code path to update prereqs on data instances
  for (auto i : each(used_instances.size()))
  {
    if (frozen_mode == access_mode::read)
    {
      used_instances[i].add_write_prereq(ctx, prereqs);
    }
    else
    {
      // rw or write
      used_instances[i].set_read_prereq(prereqs);
      used_instances[i].clear_write_prereq();
    }
  }

  // Get the logical_data_untyped from the current impl
  logical_data_untyped d(shared_from_this());

  // Keep track of the previous readers/writers and generate dot
  reserved::enforce_stf_deps_after(ctx, d, fake_task, frozen_mode);

  frozen_flag = false;
}

inline void backend_ctx_untyped::impl::erase_all_logical_data()
{
  /* Since we modify the map while iterating on it, we will copy it */
  logical_data_ids_mutex.lock();
  auto logical_data_ids_cpy = logical_data_ids;
  logical_data_ids_mutex.unlock();

  /* Erase all logical data created in this context */
  for (auto p : logical_data_ids_cpy)
  {
    auto& d = p.second;
    d.erase();
  }
}

inline void backend_ctx_untyped::impl::print_logical_data_summary() const
{
  ::std::lock_guard<::std::mutex> guard(logical_data_ids_mutex);

  fprintf(stderr, "Context current logical data summary\n");
  fprintf(stderr, "====================================\n");

  size_t total_footprint = 0;

  // Map to aggregate counts based on (symbol, footprint)
  ::std::map<::std::string, ::std::map<size_t, size_t>> data_summary;

  for (const auto& [id, data_impl] : logical_data_ids)
  {
    size_t footprint = data_impl.dinterface->data_footprint();
    total_footprint += footprint;

    ::std::string symbol = data_impl.get_symbol();
    data_summary[symbol][footprint]++;
  }

  // Print summary
  for (const auto& [symbol, footprints] : data_summary)
  {
    fprintf(stderr, "- %s,", symbol.c_str());
    bool first = true;
    for (const auto& [footprint, count] : footprints)
    {
      if (!first)
      {
        fprintf(stderr, " |");
      }
      first = false;

      fprintf(stderr, " %s", pretty_print_bytes(footprint).c_str());
      if (count > 1)
      {
        fprintf(stderr, " (x%zu)", count);
      }
    }
    fprintf(stderr, "\n");
  }

  fprintf(stderr, "====================================\n");
  fprintf(stderr, "Current footprint : %s\n", pretty_print_bytes(total_footprint).c_str());
  fprintf(stderr, "====================================\n");

  size_t total_footprint_destroyed = 0;

  // Map to aggregate counts based on (symbol, footprint)
  ::std::map<::std::string, ::std::map<size_t, size_t>> destroyed_data_summary;

  for (const auto& [symbol, footprint] : previous_logical_data_stats)
  {
    total_footprint_destroyed += footprint;
    destroyed_data_summary[symbol][footprint]++;
  }

  // Print summary
  for (const auto& [symbol, footprints] : destroyed_data_summary)
  {
    fprintf(stderr, "- %s,", symbol.c_str());
    bool first = true;
    for (const auto& [footprint, count] : footprints)
    {
      if (!first)
      {
        fprintf(stderr, " |");
      }
      first = false;

      fprintf(stderr, " %s", pretty_print_bytes(footprint).c_str());
      if (count > 1)
      {
        fprintf(stderr, " (x%zu)", count);
      }
    }
    fprintf(stderr, "\n");
  }

  fprintf(stderr, "====================================\n");
  fprintf(stderr, "Destroyed footprint : %s\n", pretty_print_bytes(total_footprint_destroyed).c_str());
  fprintf(stderr, "====================================\n");
  fprintf(stderr, "Total footprint : %s\n", pretty_print_bytes(total_footprint + total_footprint_destroyed).c_str());
  fprintf(stderr, "====================================\n");
}

// Defined here to avoid circular dependencies
inline logical_data_untyped task_dep_untyped::get_data() const
{
  assert(data);
  return unpack_state(data);
}

// Defined here to avoid circular dependencies
// (also, don't document this because Doxygen doesn't know `decltype`)
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <class T>
inline decltype(auto) task_dep<T, void, false>::instance(task& tp) const
{
  auto t = get_data();
  return static_cast<logical_data<T>&>(t).instance(tp);
}
#endif // !_CCCL_DOXYGEN_INVOKED

// Defined here to avoid circular dependencies
inline instance_id_t task::find_data_instance_id(const logical_data_untyped& d) const
{
  for (auto& it : pimpl->deps)
  {
    if (d == it.get_data())
    {
      // We found the data
      return it.get_instance_id();
    }
  }

  // This task does not has d in its dependencies
  fprintf(stderr, "FATAL: could not find this piece of data in the current task.\n");
  abort();

  return instance_id_t::invalid;
}

// Don't document this because Doxygen doesn't know `decltype`
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <typename T, typename logical_data_untyped>
inline decltype(auto) task::get(size_t submitted_index) const
{
  if (pimpl->reordered_indexes.empty())
  {
    pimpl->initialize_reordered_indexes();
  }

  EXPECT(submitted_index < pimpl->reordered_indexes.size());
  size_t reordered_id       = pimpl->reordered_indexes[submitted_index];
  instance_id_t instance_id = pimpl->deps[reordered_id].get_instance_id();
  logical_data_untyped d    = pimpl->deps[reordered_id].get_data();
  return d.template instance<T>(instance_id);
}
#endif // !_CCCL_DOXYGEN_INVOKED

/**
 * @brief Represents typed logical data.
 *
 * @tparam T The type of the underlying data.
 */
template <class T>
class logical_data : public logical_data_untyped
{
public:
  /// @brief Alias for `T`
  using element_type = T;
  /// @brief Alias for `shape_of<T>`
  using shape_t = shape_of<T>;

  /// @brief Default constructor
  logical_data() = default;

  /// @brief Constructor from an untyped logical data
  ///
  /// Warning : no checks are done to ensure the type used to create the
  /// untyped logical data matches, it is the responsibility of the caller to
  /// ensure this is a valid conversion
  logical_data(logical_data_untyped&& u)
      : logical_data_untyped(u)
  {}

  /**
   * @brief Constructor
   *
   * @tparam U Backend type
   * @param ctx Backend context
   * @param instance Reference instance used for initializing this logical data
   * @param dp Data place
   * @param data_prereq
   */
  template <typename U>
  logical_data(backend_ctx_untyped ctx, ::std::shared_ptr<U> instance, data_place dp)
      : logical_data_untyped(mv(ctx), mv(instance), mv(dp))
  {
    // Note that we did not put this static_assertion in the default
    // constructor to keep using  = default which is preferred
    static_assert(sizeof(logical_data) == sizeof(logical_data_untyped),
                  "Cannot add state here because it would be lost through slicing");

    EXPECT(get_ctx());
    static_assert(::std::is_same_v<T, typename U::element_type>);
    static_assert(::std::is_same_v<shape_of<T>, typename U::shape_t>);
  }

  ///@{ @name Execution place getter
  const shape_t& shape() const
  {
    return logical_data_untyped::common<shape_t>();
  }
  ///@}

  ///@{ @name Instance getter for a given instance or the default instance
  decltype(auto) instance(instance_id_t instance_id)
  {
    return logical_data_untyped::instance<T>(instance_id);
  }
  decltype(auto) instance(task& tp)
  {
    return instance(get_data_interface().get_default_instance_id(get_ctx(), *this, tp));
  }
  ///@}

  ///@{ @name Assign a symbolic name for this object
  logical_data& set_symbol(::std::string str)
  {
    logical_data_untyped::set_symbol(mv(str));
    return *this;
  }
  ///@}

  ///@{ @name Select a custom allocator for this logical data
  logical_data& set_allocator(block_allocator_untyped custom_allocator)
  {
    logical_data_untyped::set_allocator(mv(custom_allocator));
    return *this;
  }
  ///@}

  ///@{ @name Get hash value
  size_t hash() const
  {
    return logical_data_untyped::hash();
  }
  ///@}

  ///@{
  /**
   * @name Return a task_dep<T> object for reading and/or writing this logical data.
   *
   * @tparam Pack Additional parameter types for `task_dep<T>`'s constructor, if any
   * @param pack Additional arguments for `task_dep<T>`'s constructor, if any
   * @return task_dep<T> The object encapsulating access
   */
  template <typename... Pack>
  auto read(Pack&&... pack) const
  {
    using U = readonly_type_of<T>;
    return task_dep<U, ::std::monostate, false>(*this, access_mode::read, ::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto write(Pack&&... pack)
  {
    return task_dep<T, ::std::monostate, false>(*this, access_mode::write, ::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto rw(Pack&&... pack)
  {
    return task_dep<T, ::std::monostate, false>(*this, access_mode::rw, ::std::forward<Pack>(pack)...);
  }

  template <typename... Pack>
  auto relaxed(Pack&&... pack)
  {
    return task_dep<T, ::std::monostate, false>(*this, access_mode::relaxed, ::std::forward<Pack>(pack)...);
  }

  template <typename Op, typename... Pack>
  auto reduce(Op, no_init, Pack&&... pack)
  {
    return task_dep<T, Op, false>(*this, access_mode::reduce_no_init, ::std::forward<Pack>(pack)...);
  }

  /* If we do not pass the no_init{} tag type there, this is going to
   * initialize data, not accumulate with existing values. */
  template <typename Op, typename... Pack>
  auto reduce(Op, Pack&&... pack)
  {
    return task_dep<T, Op, true>(*this, access_mode::reduce, ::std::forward<Pack>(pack)...);
  }

  ///@}
};

/// @brief Shortcut type for the logical data produced by ctx.token()
using token = logical_data<void_interface>;

/**
 * @brief Reclaims memory from allocated data instances.
 *
 * Reclaims memory for requested size. It considers different passes depending on the specified criteria. Memory is
 * reclaimed from data instances that are not currently in use and meet the pass criteria.
 *
 * @param[in] ctx Pointer to the backend context state.
 * @param[in] place Data place from where the memory should be reclaimed.
 * @param[in] requested_s The size of memory to be reclaimed.
 * @param[out] reclaimed_s The size of memory that was successfully reclaimed.
 * @param[in,out] prereqs A unique pointer to a list of events that need to be completed before memory can be
 * reclaimed.
 */
inline void reclaim_memory(
  backend_ctx_untyped& ctx, const data_place& place, size_t requested_s, size_t& reclaimed_s, event_list& prereqs)
{
  const auto memory_node = to_index(place);

  auto& ctx_state = ctx.get_state();

  reclaimed_s = 0;

  ctx.get_dot()->set_current_color("red");
  SCOPE(exit)
  {
    ctx.get_dot()->set_current_color("white");
  };

  int first_pass = 0;
  if (getenv("NAIVE_RECLAIM"))
  {
    first_pass = 2;
  }

  /*
   * To implement a Last Recently Used (LRU) reclaiming policy, we
   * approximate access time by the largest prereq ID associated to a data
   * instance. We thus don't reclaim data in the order they appear in the
   * map, but accordingly to this largest ID.
   *
   * We therefore create a vector of data to select the most appropriate data
   * after sorting it.
   */
  using tup_t = ::std::tuple<data_instance*, reserved::logical_data_untyped_impl*, instance_id_t, int>;
  ::std::vector<tup_t> eligible_data;

  // Go through all entries
  // Pass 0 : only locally invalid instances
  // Pass 1 : only shared instances
  // Pass 2 : all instances
  // We stop as soon as we have reclaimed enough memory
  size_t eligible_data_size = 0;
  for (int pass = first_pass; (reclaimed_s < requested_s) && (eligible_data_size < requested_s) && (pass <= 2); pass++)
  {
    // Get the table of all logical data ids used in this context (the parent of the task)
    ::std::lock_guard<::std::mutex> guard(ctx_state.logical_data_ids_mutex);
    auto& logical_data_ids = ctx_state.logical_data_ids;
    for (auto& e : logical_data_ids)
    {
      auto& d = e.second;

      // fprintf(stderr, "Trying to reclaim logical data (id %d)\n", e.first);

      if (d.has_ref())
      {
        // Ignore logical data which are currently being used (e.g. those being acquired)
        continue;
      }

      // Do not try to reclaim memory from a frozen logical data (XXX we might restrict this for write-only or rw
      // frozen data)
      if (d.frozen_flag)
      {
        continue;
      }

      // We go through all data instances

      size_t ninstances = d.get_data_instance_count();
      for (auto i : each(instance_id_t(ninstances)))
      {
        auto& inst = d.get_data_instance(i);

        // Some data instances are not eligible for reclaiming (e.g. memory allocated by users)
        if (!inst.reclaimable)
        {
          continue;
        }

        // Only reclaim allocated memory
        if (!inst.is_allocated())
        {
          continue;
        }

        // Only reclaim on the requested data place
        if (to_index(inst.get_dplace()) != memory_node)
        {
          continue;
        }

        assert(inst.get_used());

        bool eligible      = true;
        auto current_state = inst.get_msir();

        switch (pass)
        {
          case 0:
            // Only process locally invalid instances
            if (current_state != reserved::msir_state_id::invalid)
            {
              eligible = false;
            }
            break;
          case 1:
            // Only process instances which will not require a write back (ie. exclude
            // reserved::msir_state_id::modified)
            if (current_state == reserved::msir_state_id::modified)
            {
              eligible = false;
            }
            break;
          default:
            // no-op
            break;
        }

        // If this data instance meets all criteria, do reclaim it
        if (eligible)
        {
          if (pass == 2 || pass == 1)
          {
            // In this case, we do not try to reclaim it
            // immediately, but we put it in a vector which will be
            // sorted after in order to implement a LRU policy
            eligible_data.emplace_back(&inst, &d, instance_id_t(i), inst.max_prereq_id());
            eligible_data_size += inst.allocated_size;
            continue;
          }
          // This does not consumes prereqs, but produces new ones
          prereqs.merge(d.do_reclaim(place, i));

          reclaimed_s += inst.allocated_size;
          // fprintf(stderr, "RECLAIMED %ld / %ld - last prereq id was %d\n", reclaimed_s, requested_s);
          if (reclaimed_s >= requested_s)
          {
            return;
          }
        }
      } // end loop over of data instances
    } // end loop over logical data ids
  } // end loop pass

  // Setting this variable to a non-zero value will disable the LRU policy
  const char* str = getenv("RECLAIM_NO_SORT");
  if (!str || atoi(str) == 0)
  {
    ::std::sort(eligible_data.begin(), eligible_data.end(), [](const auto& a, const auto& b) {
      return ::std::get<3>(a) < ::std::get<3>(b);
    });
  }

  for (auto& [inst, d, i, _] : eligible_data)
  {
    prereqs.merge(d->do_reclaim(place, i));
    reclaimed_s += inst->allocated_size;
    // fprintf(stderr, "RECLAIMED %ld / %ld - last prereq id was %d\n", reclaimed_s, requested_s);

    // Stop when we have reclaimed enough memory
    if (reclaimed_s >= requested_s)
    {
      return;
    }
  }
}

} // namespace cuda::experimental::stf
