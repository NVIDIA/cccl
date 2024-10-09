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
 * @brief Enforce dependencies before and after task submission to implement
 *        the STF model
 */

#pragma once

#include "cudastf/__stf/internal/logical_data.h"

namespace cuda::experimental::stf {

namespace reserved {

/**
 * @brief Implements STF dependencies.
 *
 * This method ensures that the current task (task) depends on the appropriate
 * predecessors. A second method enforce_stf_deps_after will be called to make
 * sure future tasks depend on that task.
 */
template <typename task_type>
inline event_list enforce_stf_deps_before(backend_ctx_untyped& bctx, logical_data_untyped& handle,
        const instance_id_t instance_id, const task_type& task, const access_mode mode,
        const ::std::optional<exec_place> eplace) {
    auto result = event_list();
    auto& cs = bctx.get_stack();
    // Get the context in which we store previous writer, readers, ...
    auto& ctx = handle.get_state();

    auto& dot = *bctx.get_dot();
    const bool dot_is_tracing = dot.is_tracing();

    if (mode == access_mode::redux) {
        // A reduction only needs to wait for previous accesses on the data instance
        ctx.current_mode = access_mode::redux;

        if (dot_is_tracing) {
            // Add this task to the list of task accessing the logical data in redux mode
            // We only store its id since this is used for dot
            ctx.pending_redux_id.push_back(task.get_unique_id());
        }

        // XXX with a mv we may avoid copies
        const auto& data_instance = handle.get_data_instance(instance_id);
        result.merge(data_instance.get_read_prereq(), data_instance.get_write_prereq());
        return result;
    }

    // This is not a reduction, but perhaps we need to reconstruct the data first?
    if (ctx.current_mode == access_mode::redux) {
        assert(eplace.has_value());
        if (dot_is_tracing) {
            // Add a dependency between previous tasks accessing the handle
            // in redux mode, and this task which forces its
            // reconstruction.
            for (const int redux_task_id: ctx.pending_redux_id) {
                dot.add_edge(redux_task_id, task.get_unique_id());
            }
            ctx.pending_redux_id.clear();
        }
        handle.reconstruct_after_redux(instance_id, eplace.value(), result);
        ctx.current_mode = access_mode::none;
    }

    // @@@TODO@@@ cleaner ...
    const bool write = (mode == access_mode::rw || mode == access_mode::write);

    // ::std::cout << "Notifying " << (write?"W":"R") << " access on " << get_symbol() << " by task " <<
    // task->get_symbol() << ::std::endl;
    if (write) {
        if (ctx.current_mode == access_mode::write) {
            // Write after Write (WAW)
            assert(ctx.current_writer.has_value());

            const auto& cw = ctx.current_writer.value();
            result.merge(cw.get_done_prereqs());

            const auto cw_id = cw.get_unique_id();

            if (dot_is_tracing) {
                dot.add_edge(cw_id, task.get_unique_id());
            }

            cs.remove_leaf_task(cw_id);

            // Replace previous writer
            ctx.previous_writer = cw;
        } else {
            // Write after read

            // The writer depends on all current readers
            auto& current_readers = ctx.current_readers;
            result.merge(current_readers.get_done_prereqs());

            for (const int reader_task_id: current_readers.get_ids()) {
                if (dot_is_tracing) {
                    dot.add_edge(reader_task_id, task.get_unique_id());
                }
                cs.remove_leaf_task(reader_task_id);
            }

            current_readers.clear();
            ctx.current_mode = access_mode::write;
        }
        // Note the task will later be set as the current writer
    } else {
        // This is a read access
        if (ctx.current_mode == access_mode::write) {
            // Read after Write
            // Current writer becomes the previous writer, and all future readers will depend on this previous
            // writer
            assert(ctx.current_writer.has_value());
            ctx.previous_writer = mv(ctx.current_writer);
            const auto& pw = ctx.previous_writer;

            result.merge(pw->get_done_prereqs());

            const int pw_id = pw->get_unique_id();

            if (dot_is_tracing) {
                dot.add_edge(pw_id, task.get_unique_id());
            }

            cs.remove_leaf_task(pw_id);

            ctx.current_mode = access_mode::none;
            // ::std::cout << "CHANGING to FALSE for " << symbol << ::std::endl;
        } else if (ctx.previous_writer.has_value()) {
            const auto& pw = ctx.previous_writer;
            result.merge(pw->get_done_prereqs());

            const int pw_id = pw->get_unique_id();
            if (dot_is_tracing) {
                dot.add_edge(pw_id, task.get_unique_id());
            }

            cs.remove_leaf_task(pw_id);
        }

        // Note : the task will later be added to the list of readers
    }

    return result;
}

template <typename task_type>
inline void enforce_stf_deps_after(logical_data_untyped& handle, const task_type& task, const access_mode mode) {
    if (mode == access_mode::redux) {
        // no further action is required
        return;
    }

    // Get the context in which we store previous writer, readers, ...
    auto& ctx = handle.get_state();

    if (mode == access_mode::rw || mode == access_mode::write) {
        ctx.current_writer = task;
    } else {
        // Add to the list of readers
        ctx.current_readers.add(task);
    }
}

/* Enforce task dependencies, allocations, and copies ... */
inline void fetch_data(backend_ctx_untyped& ctx, logical_data_untyped& d, const instance_id_t instance_id, task& t,
        access_mode mode, const ::std::optional<exec_place> eplace, const data_place& dplace, event_list& result) {
    event_list stf_prereq = reserved::enforce_stf_deps_before(ctx, d, instance_id, t, mode, eplace);

    if (d.has_interface()) {
        // Allocate data if needed (and possibly reclaim memory to do so)
        dep_allocate(ctx, d, mode, dplace, eplace, instance_id, stf_prereq);

        /*
         * DATA LAZY UPDATE (relying on the MSI protocol)
         */

        // This will initiate a copy if the data was not valid
        d.enforce_msi_protocol(instance_id, mode, stf_prereq);

        stf_prereq.optimize();

        // Gather all prereqs required to fetch this piece of data into the
        // dependencies of the task.
        // Even temporary allocation may require to enfore dependencies
        // because we are reclaiming data for instance.
        result.merge(mv(stf_prereq));
    }
}

}  // namespace reserved

/**
 * @brief Acquires necessary resources and dependencies for a task to run.
 *
 * This function prepares a task for execution by setting up its execution context,
 * sorting its dependencies to avoid deadlocks, and ensuring all necessary data
 * dependencies are fulfilled. It handles both small and large tasks by checking
 * the task size and adjusting its behavior accordingly. Dependencies are processed
 * to mark data usage, allocate necessary resources, and update data instances for
 * task execution. This function also handles the task's transition from the setup
 * phase to the running phase.
 *
 * @param ctx The backend context in which the task is executed. This context contains
 *            the execution stack and other execution-related information.
 * @param tsk The task to be prepared for execution. The task must be in the setup phase
 *            before calling this function.
 * @return An event_list containing all the input events and any additional events
 *         generated during the acquisition of dependencies. This list represents the
 *         prerequisites for the task to start execution.
 *
 * @note The function `EXPECT`s the task to be in the setup phase and the execution place
 *       not to be `exec_place::device_auto`.
 * @note Dependencies are sorted by logical data addresses to prevent deadlocks.
 * @note For tasks with multiple dependencies on the same logical data, only one
 *       instance of the data is used, and its access mode is determined by combining
 *       the access modes of all dependencies on that data.
 */
inline event_list task::acquire(backend_ctx_untyped& ctx) {
    EXPECT(get_task_phase() == task::phase::setup);

    const auto eplace = get_exec_place();
    EXPECT(eplace != exec_place::device_auto);
    // If there are any extra dependencies to fulfill
    auto result = get_input_events();

    // Automatically set the appropriate context (device, SM affinity, ...)
    pimpl->saved_place_ctx = eplace.activate(ctx);

    auto& task_deps = pimpl->deps;

    for (auto index: each(task_deps.size())) {
        assert(task_deps[index].get_data().is_initialized());
        // Save index before reordering
        task_deps[index].dependency_index = index;
        // Mark up data to avoid them being reclaimed while they are going to be used anyway
        task_deps[index].get_data().add_ref();
    }

    // Sort deps by logical data addresses to avoid deadlocks when multiple
    // threads are going to acquire the mutexes associated to the different
    // logical data
    ::std::sort(task_deps.begin(), task_deps.end());

    // Process all dependencies one by one
    for (auto it = task_deps.begin(); it != task_deps.end(); ++it) {
        logical_data_untyped d = it->get_data();
        access_mode mode = it->get_access_mode();

        auto [frozen, frozen_mode] = d.is_frozen();
        if (frozen) {
            // If we have a frozen data, we can only access it if we are making
            // a read only access, and if the data was frozen in read only mode.
            // Otherwise it's a mistake as we would modify some data possibly
            // being used outside the task
            if (!(frozen_mode == access_mode::read && mode == access_mode::read)) {
                fprintf(stderr, "Error: illegal access on frozen logical data\n");
                abort();
            }
        }

        // Make sure the context of the logical data and the context of the task match
        // This is done by comparing the addresses of the context states
        if (ctx != d.get_ctx()) {
            fprintf(stderr, "Error: mismatch between task context and logical data context\n");
            abort();
        }

        // We possibly "merge" multiple dependencies. If they have different modes, those are combined.
        // Since logical data are ordered by addresses, "mergeable" deps will be
        // stored contiguously, so we can stop as soon as there is another
        // dependency
        for (auto next = ::std::next(it); next != task_deps.end() && *it == *next; ++it, ++next) {
            mode |= next->get_access_mode();
            it->skipped = true;
        }

        /* Get of this dependency which is not skipped, and save it in a vector. We also save the equivalent merged
         * mode. */
        size_t d_index = it - task_deps.begin();
        pimpl->unskipped_indexes.push_back(::std::make_pair(d_index, mode));

        /* Make sure the logical data is locked until the task is released */
        d.get_mutex().lock();

        // The affine data place is set at the task level, it can be inherited
        // from the execution place, or be some composite data place set up in
        // a parallel_for construct, for example
        const data_place& dplace = it->get_dplace() == data_place::affine ? get_affine_data_place() : it->get_dplace();

        const instance_id_t instance_id =
                mode == access_mode::redux ? d.find_unused_instance_id(dplace) : d.find_instance_id(dplace);

        if (mode == access_mode::redux) {
            d.get_data_instance(instance_id).set_redux_op(it->get_redux_op());
        }

        // We will need to remind this when we need to access that
        // piece of data, or when we release it to manipulate the
        // proper instance.
        it->set_instance_id(instance_id);

        // This enforces dependencies with previous tasks, and ensures the data
        // instances can be used (there can be extra prereqs for an instance to
        // ensure it's properly allocated for example, or if there is a pending
        // copy from that instance to another, so we need to keep track of both
        // task dependencies (STF) and these instance-specific dependencies.
        reserved::fetch_data(ctx, d, instance_id, *this, mode, eplace, dplace, result);
    }

    // In the (rare case) where there is no data dependency for a task, the
    // task would still depend on the entry events of the context, if any
    if ((task_deps.size() == 0) && ctx.has_start_events()) {
        result.merge(ctx.get_start_events());
    }

    // @@@@ TODO@@@@ explain this algorithm, and why we need to go in reversed
    // order because we skipped equivalent dependencies that were stored
    // contiguously after sorting.
    instance_id_t previous_instance_id = instance_id_t::invalid;
    for (auto it: each(task_deps.rbegin(), task_deps.rend())) {
        if (!it->skipped) {
            previous_instance_id = it->get_instance_id();
        } else {
            assert(previous_instance_id != instance_id_t::invalid);
            // @@@@ TODO @@@@ make a unit test to make sure we have the same instance id for different acquired_data
            // ? fprintf(stderr, "SETTING SKIPPED INSTANCE ID ... %d\n", previous_instance_id);
            it->set_instance_id(previous_instance_id);
        }
    }

    for (const auto& e: task_deps) {
        logical_data_untyped d = e.get_data();

        if (e.get_access_mode() == access_mode::redux) {
            // Save the last task accessing the instance in with a relaxed coherency mode
            d.get_data_instance(e.get_instance_id()).set_last_task_relaxed(*this);
        }

        // Remove temporary reference
        d.remove_ref();
    }

    auto& dot = *ctx.get_dot();
    if (dot.is_tracing()) {
        // Declare that the node identified by unique_id depends on these prereqs
        result.dot_declare_prereqs(dot, get_unique_id(), 1);
    }

    // We consider the setup phase is over
    pimpl->phase = task::phase::running;

    return result;
}

/*
 * TODO insert description here.
 */
inline void task::release(backend_ctx_untyped& ctx, event_list& done_prereqs) {
    // After release(), the task is over
    assert(get_task_phase() == task::phase::running);
    assert(get_done_prereqs().size() == 0);

    auto& task_deps = pimpl->deps;
    auto& cs = ctx.get_stack();

    // We copy the list of prereqs into the task
    merge_event_list(done_prereqs);

    // Get the indices of logical data which were not skipped (redundant
    // dependencies are merged).
    for (auto& [ind, mode]: pimpl->unskipped_indexes) {
        auto& e = task_deps[ind];
        logical_data_untyped d = e.get_data();

        auto&& data_instance = d.get_data_instance(e.get_instance_id());

        if (mode == access_mode::read) {
            // If we have a read-only task, we only need to make sure that write accesses waits for this task
            data_instance.add_write_prereq(done_prereqs);
        } else {
            data_instance.set_read_prereq(done_prereqs);
            data_instance.clear_write_prereq();
        }

        // Update last reader/writer tasks
        reserved::enforce_stf_deps_after(d, *this, mode);
    }

    // Automatically reset the context to its original configuration (device, SM affinity, ...)
    get_exec_place().deactivate(ctx, pimpl->saved_place_ctx);

    auto& dot = *ctx.get_dot();
    if (dot.is_tracing()) {
        // These prereqs depend on the task identified by unique_id
        auto& done_prereqs = get_done_prereqs();
        done_prereqs.dot_declare_prereqs_from(dot, get_unique_id(), 1);
    }

    // This task becomes a new "leaf task" until another task depends on it
    cs.add_leaf_task(*this);

    pimpl->phase = task::phase::finished;

    /* We unlock the mutex which were locked. We only locked each logical data
     * once, so merged dependencies are ignored and we loop over unskipped indices.
     */
    for (auto& [ind, _]: pimpl->unskipped_indexes) {
        logical_data_untyped d = task_deps[ind].get_data();
        d.get_mutex().unlock();
    }

    // Doing this shunts a circular dependency that would otherwise leak logical_data objects.
    // Note that we do this for all dependencies, including redundant ones.
    for (auto& dep: task_deps) {
        dep.reset_logical_data();
    }

    /* Execute hooks (if any) */
    for (auto& hook: pimpl->post_submission_hooks) {
        hook();
    }

    // This will, in particular, release shared_ptr to logical data captured in
    // the dependencies.
    pimpl->post_submission_hooks.clear();
}

}  // namespace cuda::experimental::stf
