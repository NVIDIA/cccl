//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file stf.h
//! \brief CUDA STF (Sequential Task Flow) C Interface
//!
//! \details
//! This header provides a C interface to the CUDA STF C++ library, enabling
//! task-based parallel programming with automatic data movement and dependency management.
//!
//! The Sequential Task Flow programming model involves defining logical data and
//! submitting tasks that operate on this data. STF automatically deduces dependencies
//! between tasks and orchestrates both computation and data movement to ensure
//! efficient execution with maximum concurrency.
//!
//! \par Key Concepts:
//! - **Logical Data**: Abstract handles for data that may exist in multiple locations
//! - **Tasks**: Operations that consume and produce logical data with specified access modes
//! - **Dependencies**: Automatically inferred from data access patterns (RAW, WAR, WAW)
//! - **Execution Places**: Specify where tasks run (CPU, specific GPU devices)
//! - **Data Places**: Specify where data should be located in memory hierarchy
//!
//! \par Basic Usage Pattern:
//! \code
//! // 1. Create STF context
//! stf_ctx_handle ctx = stf_ctx_create();
//!
//! // 2. Create logical data from arrays
//! float X[1024], Y[1024];
//! stf_logical_data_handle lX = stf_logical_data(ctx, X, sizeof(X));
//! stf_logical_data_handle lY = stf_logical_data(ctx, Y, sizeof(Y));
//!
//! // 3. Create and configure task
//! stf_task_handle task = stf_task_create(ctx);
//! stf_task_add_dep(task, lX, STF_READ);  // X is read-only
//! stf_task_add_dep(task, lY, STF_RW);    // Y is read-write
//!
//! // 4. Execute task
//! stf_task_start(task);
//! CUstream stream = stf_task_get_custream(task);
//! float* x_ptr = (float*)stf_task_get(task, 0);
//! float* y_ptr = (float*)stf_task_get(task, 1);
//! // ... launch CUDA operations using stream ...
//! stf_task_end(task);
//!
//! // 5. Cleanup
//! stf_task_destroy(task);
//! stf_logical_data_destroy(lX);
//! stf_logical_data_destroy(lY);
//! stf_ctx_finalize(ctx);
//! \endcode
//!
//! \warning This API is experimental and subject to change.
//!          Define CCCL_C_EXPERIMENTAL to acknowledge this.

#pragma once

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//! \defgroup AccessMode Data Access Modes
//! \brief Specifies how tasks access logical data
//! \{

//! \brief Data access mode for task dependencies
//!
//! Specifies how a task will access logical data, which determines
//! synchronization requirements and concurrency opportunities.
typedef enum stf_access_mode
{
  STF_NONE  = 0, //!< No access (invalid)
  STF_READ  = 1 << 0, //!< Read-only access - allows concurrent readers
  STF_WRITE = 1 << 1, //!< Write-only access - requires exclusive access
  STF_RW    = STF_READ | STF_WRITE //!< Read-write access - requires exclusive access
} stf_access_mode;

//! \}

//! \defgroup Places Opaque execution and data places
//! \brief Heap-allocated handles wrapping C++ \c exec_place and \c data_place.
//! Callers own handles: every successful \c stf_*_place_* factory or \c *_clone allocates;
//! release with the matching \c *_destroy (idempotent for \c NULL).
//! \{

//! \brief Opaque handle to an \c exec_place (including grids — all execution places are grids in C++).
typedef struct stf_exec_place_opaque_t* stf_exec_place_handle;

//! \brief Opaque handle to a \c data_place.
typedef struct stf_data_place_opaque_t* stf_data_place_handle;

//! \brief 4D position (coordinates) for partition mapping.
//! Layout matches C++ pos4 for use as partition function arguments/result.
typedef struct stf_pos4
{
  int64_t x; //!< Coordinate along first axis
  int64_t y; //!< Coordinate along second axis
  int64_t z; //!< Coordinate along third axis
  int64_t t; //!< Coordinate along fourth axis
} stf_pos4;

//! \brief 4D dimensions (extents) for partition mapping.
//! Layout matches C++ dim4 for use as partition function arguments.
typedef struct stf_dim4
{
  uint64_t x; //!< Extent along first axis
  uint64_t y; //!< Extent along second axis
  uint64_t z; //!< Extent along third axis
  uint64_t t; //!< Extent along fourth axis
} stf_dim4;

//! \brief Partition (mapper) function: data coordinates -> grid position.
//! Can be implemented in C or provided from Python via ctypes/cffi.
typedef stf_pos4 (*stf_get_executor_fn)(stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims);

//! \brief Create host execution place (CPU).
stf_exec_place_handle stf_exec_place_host(void);

//! \brief Create device execution place for CUDA device \p dev_id.
stf_exec_place_handle stf_exec_place_device(int dev_id);

//! \brief Create execution place for the current CUDA device.
stf_exec_place_handle stf_exec_place_current_device(void);

//! \brief Deep copy of an execution place handle (caller must stf_exec_place_destroy the result).
stf_exec_place_handle stf_exec_place_clone(stf_exec_place_handle h);

//! \brief Release an execution place handle (including grids from stf_exec_place_grid_*).
void stf_exec_place_destroy(stf_exec_place_handle h);

//! \return Non-zero if this place is host execution.
int stf_exec_place_is_host(stf_exec_place_handle h);

//! \return Non-zero if this place is a CUDA device execution context.
int stf_exec_place_is_device(stf_exec_place_handle h);

//! \brief Writes grid dimensions into \p out_dims (all scalars are 1x1x1x1 for non-grid places).
void stf_exec_place_get_dims(stf_exec_place_handle h, stf_dim4* out_dims);

//! \brief Number of sub-places in the grid (1 for scalar places).
size_t stf_exec_place_size(stf_exec_place_handle h);

//! \brief Sets the affine data place used when logical data uses affine placement with this exec grid.
void stf_exec_place_set_affine_data_place(stf_exec_place_handle h, stf_data_place_handle affine_dplace);

//! \brief Build a grid of device execution places from device IDs (one scalar place per ID).
stf_exec_place_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count);

//! \brief Build a grid from an array of execution place handles.
stf_exec_place_handle
stf_exec_place_grid_create(const stf_exec_place_handle* places, size_t count, const stf_dim4* grid_dims);

//! \brief Same as stf_exec_place_destroy (grids are exec_place handles).
void stf_exec_place_grid_destroy(stf_exec_place_handle grid);

//! \brief Host (CPU/pinned) data placement.
stf_data_place_handle stf_data_place_host(void);

//! \brief Device-local memory for \p dev_id.
stf_data_place_handle stf_data_place_device(int dev_id);

//! \brief CUDA managed (unified) memory.
stf_data_place_handle stf_data_place_managed(void);

//! \brief Affine placement (follows the task execution place).
stf_data_place_handle stf_data_place_affine(void);

//! \brief Data on the current CUDA device.
stf_data_place_handle stf_data_place_current_device(void);

//! \brief Composite partitioned placement over a grid of execution places.
stf_data_place_handle stf_data_place_composite(stf_exec_place_handle grid, stf_get_executor_fn mapper);

//! \brief Deep copy (caller must stf_data_place_destroy).
stf_data_place_handle stf_data_place_clone(stf_data_place_handle h);

//! \brief Release a data place handle.
void stf_data_place_destroy(stf_data_place_handle h);

//! \brief Device ordinal from \c data_place_interface::get_device_ordinal() (see C++ docs for sentinel values).
int stf_data_place_get_device_ordinal(stf_data_place_handle h);

//! \brief Human-readable description; pointer valid until the next call on this thread.
const char* stf_data_place_to_string(stf_data_place_handle h);

//! \}

//! \defgroup Handles Opaque Handles
//! \brief Opaque handle types for STF objects
//! \{

//!
//! \brief Opaque handle for STF context
//!
//! Context stores the state of the STF library and serves as entry point for all API calls.
//! Must be created with stf_ctx_create() or stf_ctx_create_graph() and destroyed with stf_ctx_finalize().

typedef struct stf_ctx_handle_t* stf_ctx_handle;

//!
//! \brief Opaque handle for logical data
//!
//! Represents abstract data that may exist in multiple memory locations.
//! Created with stf_logical_data() or stf_logical_data_empty() and destroyed with stf_logical_data_destroy().

typedef struct stf_logical_data_handle_t* stf_logical_data_handle;

//!
//! \brief Opaque handle for task
//!
//! Represents a computational task that operates on logical data.
//! Created with stf_task_create() and destroyed with stf_task_destroy().

typedef struct stf_task_handle_t* stf_task_handle;

//!
//! \brief Opaque handle for CUDA kernel task
//!
//! Specialized task optimized for CUDA kernel execution.
//! Created with stf_cuda_kernel_create() and destroyed with stf_cuda_kernel_destroy().

typedef struct stf_cuda_kernel_handle_t* stf_cuda_kernel_handle;

//!
//! \brief Opaque handle for a host launch scope
//!
//! A host launch scope schedules a user-provided C callback on the host
//! as a proper task graph node, with full dependency tracking.
//! Created with stf_host_launch_create() and destroyed with stf_host_launch_destroy().

typedef struct stf_host_launch_handle_t* stf_host_launch_handle;

//!
//! \brief Opaque handle for host launch dependency data
//!
//! Passed to the host callback at invocation time.  Provides indexed
//! access to the data of each dependency and to optional user data.

typedef struct stf_host_launch_deps_handle_t* stf_host_launch_deps_handle;

//!
//! \brief C callback type for host launch
//!
//! \param deps Opaque handle to dependency data and user data

typedef void (*stf_host_callback_fn)(stf_host_launch_deps_handle deps);

//! \}

//! \defgroup Context Context Management
//! \brief Create, configure, and finalize STF contexts
//! \{

//!
//! \brief Create STF context with stream backend
//!
//! Creates a new STF context using the default stream-based backend.
//! Tasks are executed eagerly using CUDA streams and events.
//!
//! \return Context handle, or NULL if allocation failed
//!
//! \post On success, caller must finalize with stf_ctx_finalize()
//!
//! \par Example:
//! \code
//! stf_ctx_handle ctx = stf_ctx_create();
//! // ... use context ...
//! stf_ctx_finalize(ctx);
//! \endcode
//!
//! \see stf_ctx_create_graph(), stf_ctx_finalize()

stf_ctx_handle stf_ctx_create(void);

//!
//! \brief Create STF context with graph backend
//!
//! Creates a new STF context using the CUDA graph backend.
//! Tasks are captured into CUDA graphs and launched when needed,
//! potentially providing better performance for repeated patterns.
//!
//! \return Context handle, or NULL if allocation failed
//!
//! \post On success, caller must finalize with stf_ctx_finalize()
//!
//! \note Graph backend has restrictions on stream synchronization within tasks
//!
//! \par Example:
//! \code
//! stf_ctx_handle ctx = stf_ctx_create_graph();
//! // ... use context ...
//! stf_ctx_finalize(ctx);
//! \endcode
//!
//! \see stf_ctx_create(), stf_ctx_finalize()

stf_ctx_handle stf_ctx_create_graph(void);

//!
//! \brief Finalize STF context
//!
//! Waits for all pending operations to complete, performs write-back
//! of modified data to host, and releases all associated resources.
//!
//! \param ctx Context handle to finalize
//!
//! \pre ctx must be valid context handle
//! \post All pending operations completed, resources released, ctx becomes invalid
//!
//! \note This function blocks until all asynchronous operations complete
//!
//! \par Example:
//! \code
//! stf_ctx_handle ctx = stf_ctx_create();
//! // ... submit tasks ...
//! stf_ctx_finalize(ctx);  // Blocks until completion
//! \endcode
//!
//! \see stf_ctx_create(), stf_ctx_create_graph(), stf_fence()

void stf_ctx_finalize(stf_ctx_handle ctx);

//!
//! \brief Get synchronization fence for context
//!
//! Returns a CUDA stream that will be signaled when all pending
//! operations in the context complete. Provides non-blocking
//! alternative to stf_ctx_finalize() for synchronization queries.
//!
//! \param ctx Context handle
//! \return CUDA stream for synchronization
//!
//! \pre ctx must be valid context handle
//!
//! \par Example:
//! \code
//! stf_ctx_handle ctx = stf_ctx_create();
//! // ... submit tasks ...
//!
//! cudaStream_t fence = stf_fence(ctx);
//! cudaStreamSynchronize(fence);  // Wait for completion
//! stf_ctx_finalize(ctx);
//! \endcode
//!
//! \see stf_ctx_finalize()

cudaStream_t stf_fence(stf_ctx_handle ctx);

//! \}

//! \defgroup LogicalData Logical Data Management
//! \brief Create and manage abstract data handles
//! \{

//!
//! \brief Create logical data from existing memory buffer
//!
//! Creates logical data handle from existing memory buffer, assuming host data place.
//! This is a convenience wrapper around stf_logical_data_with_place() with host placement.
//!
//! \param ctx Context handle
//! \param addr Pointer to existing data buffer (assumed to be host memory)
//! \param sz Size of data in bytes
//!
//! \pre ctx must be valid context handle
//! \pre addr must not be NULL and point to host-accessible memory
//! \pre sz must be greater than 0
//! \return Logical data handle, or NULL on allocation failure
//!
//! \note This function assumes host memory. For device/managed memory, use stf_logical_data_with_place()
//! \note Equivalent to host placement via stf_data_place_host() passed to stf_logical_data_with_place()
//!
//! \par Example:
//! \code
//! float data[1024];
//! stf_logical_data_handle ld = stf_logical_data(ctx, data, sizeof(data));  // Assumes host memory
//! // ... use in tasks ...
//! stf_logical_data_destroy(ld);
//! \endcode
//!
//! \see stf_logical_data_with_place(), stf_logical_data_empty(), stf_logical_data_destroy()

stf_logical_data_handle stf_logical_data(stf_ctx_handle ctx, void* addr, size_t sz);

//!
//! \brief Create logical data handle from address with data place specification
//!
//! Creates logical data handle from existing memory buffer, explicitly specifying where
//! the memory is located (host, device, managed, etc.). This is the primary and recommended
//! logical data creation function as it provides STF with essential memory location information
//! for optimal data movement and placement strategies.
//!
//! \param ctx Context handle
//! \param addr Pointer to existing memory buffer
//! \param sz Size of buffer in bytes
//! \param dplace Data place specifying memory location
//!
//! \pre ctx must be valid context handle
//! \pre addr must point to valid memory of at least sz bytes
//! \pre sz must be greater than 0
//! \pre dplace must be valid data place (not invalid)
//!
//! \return Logical data handle on success, or NULL on allocation failure
//! \post Caller owns returned handle (must call stf_logical_data_destroy())
//!
//! \par Examples:
//! \code
//! // GPU device memory (recommended for CUDA arrays)
//! float* device_ptr;
//! cudaMalloc(&device_ptr, 1000 * sizeof(float));
//! stf_data_place_handle dplace = stf_data_place_device(0);
//! stf_logical_data_handle ld =
//!   stf_logical_data_with_place(ctx, device_ptr, 1000 * sizeof(float), dplace);
//! stf_data_place_destroy(dplace);
//!
//! // Host memory
//! float* host_data = new float[1000];
//! stf_data_place_handle host_place = stf_data_place_host();
//! stf_logical_data_handle ld_host =
//!   stf_logical_data_with_place(ctx, host_data, 1000 * sizeof(float), host_place);
//! stf_data_place_destroy(host_place);
//!
//! // Managed memory
//! float* managed_ptr;
//! cudaMallocManaged(&managed_ptr, 1000 * sizeof(float));
//! stf_data_place_handle managed_place = stf_data_place_managed();
//! stf_logical_data_handle ld_managed =
//!   stf_logical_data_with_place(ctx, managed_ptr, 1000 * sizeof(float), managed_place);
//! stf_data_place_destroy(managed_place);
//! \endcode
//!
//! \see stf_data_place_device(), stf_data_place_host(), stf_data_place_managed()

stf_logical_data_handle
stf_logical_data_with_place(stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace);

//!
//! \brief Set symbolic name for logical data
//!
//! Associates a human-readable name with logical data for debugging
//! and task graph visualization.
//!
//! \param ld Logical data handle
//! \param symbol Null-terminated string name
//!
//! \pre ld must be valid logical data handle
//! \pre symbol must not be NULL
//!
//! \note Symbol appears in DOT graph output when CUDASTF_DOT_FILE is set
//!
//! \par Example:
//! \code
//! stf_logical_data_handle ld = stf_logical_data(ctx, data, size);
//! stf_logical_data_set_symbol(ld, "input_matrix");
//! \endcode
//!
//! \see stf_task_set_symbol()

void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol);

//!
//! \brief Destroy logical data handle
//!
//! Destroys logical data handle and releases associated resources.
//! Triggers write-back to host if data was modified.
//!
//! \param ld Logical data handle to destroy
//!
//! \pre ld must be valid logical data handle
//! \post ld becomes invalid, resources released
//!
//! \note Must be called for every created logical data handle
//!
//! \par Example:
//! \code
//! stf_logical_data_handle ld = stf_logical_data(ctx, data, size);
//! // ... use in tasks ...
//! stf_logical_data_destroy(ld);  // Cleanup
//! \endcode
//!
//! \see stf_logical_data(), stf_logical_data_empty()

void stf_logical_data_destroy(stf_logical_data_handle ld);

//!
//! \brief Create empty logical data (temporary)
//!
//! Creates logical data of specified size without backing host memory.
//! Useful for temporary buffers in multi-stage computations.
//!
//! \param ctx Context handle
//! \param length Size in bytes
//!
//! \pre ctx must be valid context handle
//! \pre length must be greater than 0
//! \return Logical data handle, or NULL on allocation failure
//!
//! \note First access must be write-only (STF_WRITE)
//! \note No write-back occurs since there's no host backing
//!
//! \par Example:
//! \code
//! stf_logical_data_handle temp = stf_logical_data_empty(ctx, 1024 * sizeof(float));
//!
//! // First access must be write-only
//! stf_task_add_dep(task, temp, STF_WRITE);
//! \endcode
//!
//! \see stf_logical_data(), stf_logical_data_destroy()

stf_logical_data_handle stf_logical_data_empty(stf_ctx_handle ctx, size_t length);

//!
//! \brief Create synchronization token
//!
//! Creates a logical data handle for synchronization purposes only.
//! Contains no actual data but can be used to enforce execution order.
//!
//! \param ctx Context handle
//!
//! \pre ctx must be valid context handle
//! \return Token handle, or NULL on allocation failure
//!
//! \note More efficient than using dummy data for synchronization
//! \note Can be accessed with any access mode
//!
//! \par Example:
//! \code
//! stf_logical_data_handle sync_token = stf_token(ctx);
//!
//! // Task 1 signals completion
//! stf_task_add_dep(task1, sync_token, STF_WRITE);
//!
//! // Task 2 waits for task1
//! stf_task_add_dep(task2, sync_token, STF_READ);
//! \endcode
//!
//! \see stf_logical_data(), stf_logical_data_destroy()

stf_logical_data_handle stf_token(stf_ctx_handle ctx);

//! \}

//! \defgroup TaskManagement Task Management
//! \brief Create, configure, and execute computational tasks
//! \{

//!
//! \brief Create new task
//!
//! Creates a new task within the specified context. Task is created
//! but not configured or executed. Use other stf_task_* functions
//! to configure execution place, add dependencies, and execute.
//!
//! \param ctx Context handle
//!
//! \pre ctx must be valid context handle
//! \return Task handle, or NULL on allocation failure
//!
//! \par Example:
//! \code
//! stf_task_handle task = stf_task_create(ctx);
//! // ... configure task ...
//! stf_task_destroy(task);
//! \endcode
//!
//! \see stf_task_destroy(), stf_task_set_exec_place(), stf_task_add_dep()

stf_task_handle stf_task_create(stf_ctx_handle ctx);

//!
//! \brief Set task execution place
//!
//! Specifies where the task should execute (device or host).
//! If not called, defaults to current device.
//!
//! \param t Task handle
//! \param exec_p Pointer to execution place specification
//!
//! \pre t must be valid task handle
//! \pre exec_p must not be NULL
//! \pre Must be called before stf_task_start()
//!
//! \par Example:
//! \code
//! stf_task_handle task = stf_task_create(ctx);
//!
//! // Execute on device 1
//! stf_exec_place_handle place = stf_exec_place_device(1);
//! stf_task_set_exec_place(task, place);
//! stf_exec_place_destroy(place);
//! \endcode
//!
//! \see stf_exec_place_device(), stf_exec_place_host()

void stf_task_set_exec_place(stf_task_handle t, stf_exec_place_handle exec_p);

//!
//! \brief Set symbolic name for task
//!
//! Associates a human-readable name with task for debugging
//! and task graph visualization.
//!
//! \param t Task handle
//! \param symbol Null-terminated string name
//!
//! \pre t must be valid task handle
//! \pre symbol must not be NULL
//!
//! \note Symbol appears in DOT graph output when CUDASTF_DOT_FILE is set
//!
//! \par Example:
//! \code
//! stf_task_handle task = stf_task_create(ctx);
//! stf_task_set_symbol(task, "matrix_multiply");
//! \endcode
//!
//! \see stf_logical_data_set_symbol()

void stf_task_set_symbol(stf_task_handle t, const char* symbol);

//!
//! \brief Add data dependency to task
//!
//! Adds a data dependency with specified access mode. Order of calls
//! determines index for stf_task_get(). Dependencies determine
//! automatic task synchronization.
//!
//! \param t Task handle
//! \param ld Logical data handle
//! \param m Access mode (STF_READ, STF_WRITE, STF_RW)
//!
//! \pre t must be valid task handle
//! \pre ld must be valid logical data handle
//! \pre m must be valid access mode
//!
//! \par Example:
//! \code
//! stf_task_add_dep(task, input_data, STF_READ);    // Index 0
//! stf_task_add_dep(task, output_data, STF_WRITE);  // Index 1
//! stf_task_add_dep(task, temp_data, STF_RW);       // Index 2
//! \endcode
//!
//! \see stf_task_add_dep_with_dplace(), stf_task_get()

void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m);

//!
//! \brief Add data dependency with explicit data placement
//!
//! Adds data dependency with specified access mode and explicit
//! data placement. Overrides default affine placement.
//!
//! \param t Task handle
//! \param ld Logical data handle
//! \param m Access mode (STF_READ, STF_WRITE, STF_RW)
//! \param data_p Pointer to data place specification
//!
//! \pre t must be valid task handle
//! \pre ld must be valid logical data handle
//! \pre m must be valid access mode
//! \pre data_p must not be NULL
//!
//! \par Example:
//! \code
//! // Force data to device 0 even if task runs elsewhere
//! stf_data_place_handle dplace = stf_data_place_device(0);
//! stf_task_add_dep_with_dplace(task, ld, STF_READ, dplace);
//! stf_data_place_destroy(dplace);
//! \endcode
//!
//! \see stf_task_add_dep(), stf_data_place_device(), stf_data_place_host()

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p);

//!
//! \brief Begin task execution
//!
//! Starts task execution. After this call, use stf_task_get_custream()
//! and stf_task_get() to access CUDA stream and data pointers.
//!
//! \param t Task handle
//!
//! \pre t must be valid task handle
//! \pre Task dependencies must already be configured
//! \post Task is executing, stream and data available
//!
//! \par Example:
//! \code
//! // Configure task first
//! stf_task_add_dep(task, data, STF_RW);
//!
//! // Start execution
//! stf_task_start(task);
//!
//! // Now can access stream and data
//! CUstream stream = stf_task_get_custream(task);
//! float* ptr = (float*)stf_task_get(task, 0);
//! \endcode
//!
//! \see stf_task_end(), stf_task_get_custream(), stf_task_get()

void stf_task_start(stf_task_handle t);

//!
//! \brief End task execution
//!
//! Ends task execution. Call after all CUDA operations are
//! submitted to the task stream.
//!
//! \param t Task handle
//!
//! \pre t must be valid task handle
//! \pre stf_task_start() must have been called
//! \post Task execution ended, may continue asynchronously
//!
//! \par Example:
//! \code
//! stf_task_start(task);
//! CUstream stream = stf_task_get_custream(task);
//!
//! // Launch operations
//! my_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(args...);
//!
//! stf_task_end(task);  // Operations may still be running
//! \endcode
//!
//! \see stf_task_start()

void stf_task_end(stf_task_handle t);

//!
//! \brief Get CUDA stream for task
//!
//! Returns CUDA stream associated with the task. All CUDA operations
//! within task must use this stream for proper synchronization.
//!
//! \param t Task handle
//! \return CUDA stream for launching operations
//!
//! \pre t must be valid task handle
//! \pre stf_task_start() must have been called
//!
//! \par Example:
//! \code
//! stf_task_start(task);
//! CUstream stream = stf_task_get_custream(task);
//!
//! // Launch kernel using this stream
//! my_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(args...);
//! \endcode
//!
//! \see stf_task_start(), stf_task_get()

CUstream stf_task_get_custream(stf_task_handle t);

//!
//! \brief Get data pointer for task dependency
//!
//! Returns pointer to logical data instance for specified dependency.
//! Index corresponds to order of stf_task_add_dep() calls.
//!
//! \param t Task handle
//! \param submitted_index Dependency index (0-based)
//! \return Pointer to data (cast to appropriate type)
//!
//! \pre t must be valid task handle
//! \pre stf_task_start() must have been called
//! \pre submitted_index must be valid dependency index
//! \post Pointer valid until stf_task_end()
//!
//! \par Example:
//! \code
//! // Dependencies added in this order:
//! stf_task_add_dep(task, input, STF_READ);     // Index 0
//! stf_task_add_dep(task, output, STF_WRITE);   // Index 1
//!
//! stf_task_start(task);
//!
//! // Get data pointers
//! const float* in = (const float*)stf_task_get(task, 0);
//! float* out = (float*)stf_task_get(task, 1);
//! \endcode
//!
//! \see stf_task_add_dep(), stf_task_start()

void* stf_task_get(stf_task_handle t, int submitted_index);

//!
//! \brief Destroy task handle
//!
//! Destroys task handle and releases associated resources.
//! Task should be completed before destruction.
//!
//! \param t Task handle to destroy
//!
//! \pre t must be valid task handle
//! \post t becomes invalid, resources released
//!
//! \note Must be called for every created task
//!
//! \par Example:
//! \code
//! stf_task_handle task = stf_task_create(ctx);
//! // ... configure and execute task ...
//! stf_task_destroy(task);
//! \endcode
//!
//! \see stf_task_create()

void stf_task_destroy(stf_task_handle t);

//!
//! \brief Enable graph capture for task (advanced)
//!
//! Enables graph capture optimization for the task.
//! Advanced feature typically not needed for basic usage.
//!
//! \param t Task handle
//!
//! \pre t must be valid task handle
//!
//! \note Used internally for CUDA graph backend optimization

void stf_task_enable_capture(stf_task_handle t);

//! \}

//! \defgroup CUDAKernel CUDA Kernel Interface
//! \brief Optimized interface for CUDA kernel execution
//! \{

//!
//! \brief Create CUDA kernel task
//!
//! Creates a specialized task optimized for CUDA kernel execution.
//! More efficient than generic tasks for repeated kernel launches,
//! especially with CUDA graph backend.
//!
//! \param ctx Context handle
//!
//! \pre ctx must be valid context handle
//! \return Kernel handle, or NULL on allocation failure
//!
//! \par Example:
//! \code
//! stf_cuda_kernel_handle kernel = stf_cuda_kernel_create(ctx);
//! // ... configure kernel ...
//! stf_cuda_kernel_destroy(kernel);
//! \endcode
//!
//! \see stf_cuda_kernel_destroy(), stf_task_create()

stf_cuda_kernel_handle stf_cuda_kernel_create(stf_ctx_handle ctx);

//!
//! \brief Set kernel execution place
//!
//! Specifies where the CUDA kernel should execute.
//!
//! \param k Kernel handle
//! \param exec_p Pointer to execution place specification
//!
//! \pre k must be valid kernel handle
//! \pre exec_p must not be NULL
//!
//! \see stf_exec_place_device(), stf_task_set_exec_place()

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place_handle exec_p);

//!
//! \brief Set symbolic name for kernel
//!
//! Associates human-readable name with kernel for debugging.
//!
//! \param k Kernel handle
//! \param symbol Null-terminated string name
//!
//! \pre k must be valid kernel handle
//! \pre symbol must not be NULL
//!
//! \see stf_task_set_symbol(), stf_logical_data_set_symbol()

void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol);

//!
//! \brief Add data dependency to kernel
//!
//! Adds data dependency with specified access mode for kernel execution.
//!
//! \param k Kernel handle
//! \param ld Logical data handle
//! \param m Access mode (STF_READ, STF_WRITE, STF_RW)
//!
//! \pre k must be valid kernel handle
//! \pre ld must be valid logical data handle
//! \pre m must be valid access mode
//!
//! \see stf_task_add_dep()

void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m);

//!
//! \brief Start kernel execution
//!
//! Begins kernel execution phase. After this, add kernel descriptions
//! with stf_cuda_kernel_add_desc().
//!
//! \param k Kernel handle
//!
//! \pre k must be valid kernel handle
//! \pre Dependencies must already be configured
//!
//! \see stf_cuda_kernel_add_desc(), stf_cuda_kernel_end()

void stf_cuda_kernel_start(stf_cuda_kernel_handle k);

//!
//! \brief Add CUDA kernel launch description (driver API)
//!
//! Adds kernel launch specification using CUDA driver API function handle.
//! This is the low-level interface used internally.
//!
//! \param k Kernel handle
//! \param cufunc CUDA driver API function handle
//! \param grid_dim_ CUDA grid dimensions
//! \param block_dim_ CUDA block dimensions
//! \param shared_mem_ Shared memory size in bytes
//! \param arg_cnt Number of kernel arguments
//! \param args Array of pointers to kernel arguments
//!
//! \pre k must be valid kernel handle
//! \pre stf_cuda_kernel_start() must have been called
//! \pre cufunc must be valid CUfunction
//! \pre args must contain arg_cnt valid argument pointers
//!
//! \see stf_cuda_kernel_add_desc()

void stf_cuda_kernel_add_desc_cufunc(
  stf_cuda_kernel_handle k,
  CUfunction cufunc,
  dim3 grid_dim_,
  dim3 block_dim_,
  size_t shared_mem_,
  int arg_cnt,
  const void** args);

//!
//! \brief Add CUDA kernel launch description
//!
//! Adds kernel launch specification using runtime API function pointer.
//! Automatically converts to driver API internally.
//!
//! \param k Kernel handle
//! \param func Pointer to __global__ function
//! \param grid_dim_ CUDA grid dimensions
//! \param block_dim_ CUDA block dimensions
//! \param shared_mem_ Shared memory size in bytes
//! \param arg_cnt Number of kernel arguments
//! \param args Array of pointers to kernel arguments
//!
//! \return cudaSuccess on success, or appropriate cudaError_t on failure
//!
//! \pre k must be valid kernel handle
//! \pre stf_cuda_kernel_start() must have been called
//! \pre func must be valid __global__ function pointer
//! \pre args must contain arg_cnt valid argument pointers
//!
//! \note Converts function pointer to CUfunction automatically
//!
//! \par Example:
//! \code
//! // Kernel: __global__ void axpy(float alpha, float* x, float* y)
//! stf_cuda_kernel_start(kernel);
//!
//! // Prepare arguments
//! float alpha = 2.0f;
//! float* d_x = (float*)stf_cuda_kernel_get_arg(kernel, 0);
//! float* d_y = (float*)stf_cuda_kernel_get_arg(kernel, 1);
//! const void* args[] = {&alpha, &d_x, &d_y};
//!
//! // Launch kernel (caller must handle return values != cudaSuccess)
//! cudaError_t err = stf_cuda_kernel_add_desc(kernel, (void*)axpy,
//!                                           dim3(16), dim3(128), 0, 3, args);
//! stf_cuda_kernel_end(kernel);
//! \endcode
//!
//! \see stf_cuda_kernel_add_desc_cufunc(), stf_cuda_kernel_get_arg()

static inline cudaError_t stf_cuda_kernel_add_desc(
  stf_cuda_kernel_handle k,
  const void* func,
  dim3 grid_dim_,
  dim3 block_dim_,
  size_t shared_mem_,
  int arg_cnt,
  const void** args)
{
  CUfunction cufunc;
  cudaError_t res = cudaGetFuncBySymbol(&cufunc, func);
  if (res != cudaSuccess)
  {
    return res;
  }

  stf_cuda_kernel_add_desc_cufunc(k, cufunc, grid_dim_, block_dim_, shared_mem_, arg_cnt, args);
  return cudaSuccess;
}

//!
//! \brief Get kernel argument data pointer
//!
//! Returns pointer to logical data for use as kernel argument.
//! Index corresponds to order of stf_cuda_kernel_add_dep() calls.
//!
//! \param k Kernel handle
//! \param index Dependency index (0-based)
//! \return Pointer to data for kernel argument
//!
//! \pre k must be valid kernel handle
//! \pre stf_cuda_kernel_start() must have been called
//! \pre index must be valid dependency index
//!
//! \see stf_cuda_kernel_add_desc(), stf_task_get()

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index);

//!
//! \brief End kernel execution
//!
//! Ends kernel execution phase. Call after all kernel descriptions
//! are added with stf_cuda_kernel_add_desc().
//!
//! \param k Kernel handle
//!
//! \pre k must be valid kernel handle
//! \pre stf_cuda_kernel_start() must have been called
//!
//! \see stf_cuda_kernel_start()

void stf_cuda_kernel_end(stf_cuda_kernel_handle k);

//!
//! \brief Destroy kernel handle
//!
//! Destroys kernel handle and releases associated resources.
//!
//! \param k Kernel handle to destroy
//!
//! \pre k must be valid kernel handle
//! \post k becomes invalid, resources released
//!
//! \note Must be called for every created kernel
//!
//! \see stf_cuda_kernel_create()

void stf_cuda_kernel_destroy(stf_cuda_kernel_handle k);

//! \}

//! \defgroup HostLaunch Host Launch
//! \brief Schedule a host callback as a task graph node with dependency tracking
//!
//! \details
//! Host launch provides a way to run arbitrary host-side functions as part of
//! the task graph. Unlike generic tasks where the user manually launches work
//! on a stream, host launch automatically schedules a C callback via
//! `cudaLaunchHostFunc` (stream context) or `cudaGraphAddHostNode` (graph context).
//!
//! This is the untyped counterpart of the C++ `ctx.host_launch(deps...)->*lambda`
//! construct, designed for use from C and Python bindings.
//! \{

//! \brief Create a host launch scope on a regular context
//!
//! \param ctx Context handle
//! \return Host launch handle, or NULL on allocation failure
//!
//! \see stf_host_launch_destroy()
stf_host_launch_handle stf_host_launch_create(stf_ctx_handle ctx);

//! \brief Add a dependency to a host launch scope
//!
//! \param h Host launch handle
//! \param ld Logical data handle
//! \param m Access mode (STF_READ, STF_WRITE, STF_RW)
//!
//! \see stf_task_add_dep()
void stf_host_launch_add_dep(stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m);

//! \brief Set the debug symbol for a host launch scope
//!
//! \param h Host launch handle
//! \param symbol Null-terminated string
void stf_host_launch_set_symbol(stf_host_launch_handle h, const char* symbol);

//! \brief Copy user data into the host launch scope
//!
//! The data is copied and later accessible via
//! stf_host_launch_deps_get_user_data() inside the callback.
//! An optional destructor is called on the copied buffer when the
//! dependency handle is destroyed.
//!
//! \param h Host launch handle
//! \param data Pointer to user data
//! \param size Size of user data in bytes
//! \param dtor Optional destructor for the copied data (may be NULL)
void stf_host_launch_set_user_data(stf_host_launch_handle h, const void* data, size_t size, void (*dtor)(void*));

//! \brief Submit the host callback and finalize the scope
//!
//! After this call, the callback will be invoked on the host when all
//! read/write dependencies are satisfied.  The callback receives an
//! opaque deps handle for accessing dependency data and user data.
//!
//! \param h Host launch handle
//! \param callback Function pointer invoked on the host
//!
//! \see stf_host_launch_create()
void stf_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback);

//! \brief Destroy a host launch handle
//!
//! \param h Host launch handle
//!
//! \see stf_host_launch_create()
void stf_host_launch_destroy(stf_host_launch_handle h);

//! \brief Get the raw data pointer for a dependency
//!
//! Returns the host-side pointer to the data of the dependency at \p index.
//! The pointer is valid only during the callback execution.
//!
//! \param deps Dependency handle
//! \param index Zero-based dependency index
//! \return Pointer to the data (as `slice<char>` data handle)
void* stf_host_launch_deps_get(stf_host_launch_deps_handle deps, size_t index);

//! \brief Get the byte size of a dependency
//!
//! \param deps Dependency handle
//! \param index Zero-based dependency index
//! \return Size in bytes
size_t stf_host_launch_deps_get_size(stf_host_launch_deps_handle deps, size_t index);

//! \brief Get the number of dependencies
//!
//! \param deps Dependency handle
//! \return Number of dependencies
size_t stf_host_launch_deps_size(stf_host_launch_deps_handle deps);

//! \brief Get the user data pointer
//!
//! \param deps Dependency handle
//! \return Pointer to the copied user data, or NULL if none was set
void* stf_host_launch_deps_get_user_data(stf_host_launch_deps_handle deps);

//! \}

#ifdef __cplusplus
}
#endif
