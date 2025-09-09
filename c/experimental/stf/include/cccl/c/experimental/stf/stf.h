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
//! stf_ctx_handle ctx;
//! stf_ctx_create(&ctx);
//!
//! // 2. Create logical data from arrays
//! float X[1024], Y[1024];
//! stf_logical_data_handle lX, lY;
//! stf_logical_data(ctx, &lX, X, sizeof(X));
//! stf_logical_data(ctx, &lY, Y, sizeof(Y));
//!
//! // 3. Create and configure task
//! stf_task_handle task;
//! stf_task_create(ctx, &task);
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
//! stf_ctx_finalize(ctx);
//! stf_task_destroy(task);
//! stf_logical_data_destroy(lX);
//! stf_logical_data_destroy(lY);
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

//! \defgroup ExecPlace Execution Places
//! \brief Specify where tasks should execute
//! \{

//! \brief Device execution place configuration
typedef struct stf_exec_place_device
{
  int dev_id; //!< CUDA device ID (0-based)
} stf_exec_place_device;

//! \brief Host execution place configuration
typedef struct stf_exec_place_host
{
  char dummy; //!< Dummy field for standard C compatibility
} stf_exec_place_host;

//! \brief Execution place type discriminator
typedef enum stf_exec_place_kind
{
  STF_EXEC_PLACE_DEVICE, //!< Task executes on CUDA device
  STF_EXEC_PLACE_HOST //!< Task executes on host (CPU)
} stf_exec_place_kind;

//! \brief Execution place specification
//!
//! Tagged union specifying where a task should execute.
//! Use helper functions make_device_place() and make_host_place() to create.
typedef struct stf_exec_place
{
  enum stf_exec_place_kind kind; //!< Type of execution place
  union
  {
    stf_exec_place_device device; //!< Device configuration (when kind == STF_EXEC_PLACE_DEVICE)
    stf_exec_place_host host; //!< Host configuration (when kind == STF_EXEC_PLACE_HOST)
  } u; //!< Configuration union
} stf_exec_place;

//! \brief Create execution place for CUDA device
//!
//! \param dev_id CUDA device index (0-based)
//! \return Execution place configured for specified device
//!
//! \par Example:
//! \code
//! // Execute task on device 1
//! stf_exec_place place = make_device_place(1);
//! stf_task_set_exec_place(task, &place);
//! \endcode
static inline stf_exec_place make_device_place(int dev_id)
{
  stf_exec_place p;
  p.kind            = STF_EXEC_PLACE_DEVICE;
  p.u.device.dev_id = dev_id;
  return p;
}

//! \brief Create execution place for host (CPU)
//!
//! \return Execution place configured for host execution
//!
//! \par Example:
//! \code
//! // Execute task on host
//! stf_exec_place place = make_host_place();
//! stf_task_set_exec_place(task, &place);
//! \endcode
static inline stf_exec_place make_host_place()
{
  stf_exec_place p;
  p.kind         = STF_EXEC_PLACE_HOST;
  p.u.host.dummy = 0; /* to avoid uninitialized memory warnings */
  return p;
}

//! \}

//! \defgroup DataPlace Data Places
//! \brief Specify where logical data should be located
//! \{

//! \brief Device data place configuration
typedef struct stf_data_place_device
{
  int dev_id; //!< CUDA device ID for data placement
} stf_data_place_device;

//! \brief Host data place configuration
typedef struct stf_data_place_host
{
  char dummy; //!< Dummy field for standard C compatibility
} stf_data_place_host;

//! \brief Managed memory data place configuration
typedef struct stf_data_place_managed
{
  char dummy; //!< Dummy field for standard C compatibility
} stf_data_place_managed;

//! \brief Affine data place configuration
//!
//! Affine placement means data follows the execution location automatically.
typedef struct stf_data_place_affine
{
  char dummy; //!< Dummy field for standard C compatibility
} stf_data_place_affine;

//! \brief Data place type discriminator
typedef enum stf_data_place_kind
{
  STF_DATA_PLACE_DEVICE, //!< Data on specific device memory
  STF_DATA_PLACE_HOST, //!< Data on host (CPU) memory
  STF_DATA_PLACE_MANAGED, //!< Data in CUDA managed (unified) memory
  STF_DATA_PLACE_AFFINE //!< Data follows execution place (default)
} stf_data_place_kind;

//! \brief Data placement specification
//!
//! Tagged union specifying where logical data should be located.
//! Use helper functions to create (make_device_data_place(), etc.).
typedef struct stf_data_place
{
  enum stf_data_place_kind kind; //!< Type of data placement
  union
  {
    stf_data_place_device device; //!< Device placement configuration
    stf_data_place_host host; //!< Host placement configuration
    stf_data_place_managed managed; //!< Managed memory configuration
    stf_data_place_affine affine; //!< Affine placement configuration
  } u; //!< Configuration union
} stf_data_place;

//! \brief Create data place for specific CUDA device
//!
//! \param dev_id CUDA device index (0-based)
//! \return Data place configured for device memory
//!
//! \par Example:
//! \code
//! // Force data to device 1 even if task runs elsewhere
//! stf_data_place dplace = make_device_data_place(1);
//! stf_task_add_dep_with_dplace(task, data, STF_READ, &dplace);
//! \endcode
static inline stf_data_place make_device_data_place(int dev_id)
{
  stf_data_place p;
  p.kind            = STF_DATA_PLACE_DEVICE;
  p.u.device.dev_id = dev_id;
  return p;
}

//! \brief Create data place for host memory
//!
//! \return Data place configured for host (CPU) memory
//!
//! \par Example:
//! \code
//! // Keep data on host even for device tasks (sparse access)
//! stf_data_place dplace = make_host_data_place();
//! stf_task_add_dep_with_dplace(task, data, STF_READ, &dplace);
//! \endcode
static inline struct stf_data_place make_host_data_place()
{
  stf_data_place p;
  p.kind         = STF_DATA_PLACE_HOST;
  p.u.host.dummy = 0; /* to avoid uninitialized memory warnings */
  return p;
}

//!
//! \brief Create data place for CUDA managed memory
//!
//! \return Data place configured for managed (unified) memory
//!
//! \par Example:
//! \code
//! // Use managed memory for flexible access patterns
//! stf_data_place dplace = make_managed_data_place();
//! stf_task_add_dep_with_dplace(task, data, STF_RW, &dplace);
//! \endcode

static inline struct stf_data_place make_managed_data_place()
{
  stf_data_place p;
  p.kind            = STF_DATA_PLACE_MANAGED;
  p.u.managed.dummy = 0; /* to avoid uninitialized memory warnings */
  return p;
}

//!
//! \brief Create affine data place (follows execution location)
//!
//! \return Data place configured for affine placement (default behavior)
//!
//! \par Example:
//! \code
//! // Explicitly specify default behavior
//! stf_data_place dplace = make_affine_data_place();
//! stf_task_add_dep_with_dplace(task, data, STF_RW, &dplace);
//! \endcode

static inline struct stf_data_place make_affine_data_place()
{
  stf_data_place p;
  p.kind           = STF_DATA_PLACE_AFFINE;
  p.u.affine.dummy = 0; /* to avoid uninitialized memory warnings */
  return p;
}

//! \}

//! \defgroup Handles Opaque Handles
//! \brief Opaque handle types for STF objects
//! \{

//!
//! \brief Opaque handle for STF context
//!
//! Context stores the state of the STF library and serves as entry point for all API calls.
//! Must be created with stf_ctx_create() or stf_ctx_create_graph() and destroyed with stf_ctx_finalize().

typedef void* stf_ctx_handle;

//!
//! \brief Opaque handle for logical data
//!
//! Represents abstract data that may exist in multiple memory locations.
//! Created with stf_logical_data() or stf_logical_data_empty() and destroyed with stf_logical_data_destroy().

typedef void* stf_logical_data_handle;

//!
//! \brief Opaque handle for task
//!
//! Represents a computational task that operates on logical data.
//! Created with stf_task_create() and destroyed with stf_task_destroy().

typedef void* stf_task_handle;

//!
//! \brief Opaque handle for CUDA kernel task
//!
//! Specialized task optimized for CUDA kernel execution.
//! Created with stf_cuda_kernel_create() and destroyed with stf_cuda_kernel_destroy().

typedef void* stf_cuda_kernel_handle;

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
//! \param[out] ctx Pointer to receive context handle
//!
//! \pre ctx must not be NULL
//! \post *ctx contains valid context handle that must be finalized with stf_ctx_finalize()
//!
//! \par Example:
//! \code
//! stf_ctx_handle ctx;
//! stf_ctx_create(&ctx);
//! // ... use context ...
//! stf_ctx_finalize(ctx);
//! \endcode
//!
//! \see stf_ctx_create_graph(), stf_ctx_finalize()

void stf_ctx_create(stf_ctx_handle* ctx);

//!
//! \brief Create STF context with graph backend
//!
//! Creates a new STF context using the CUDA graph backend.
//! Tasks are captured into CUDA graphs and launched when needed,
//! potentially providing better performance for repeated patterns.
//!
//! \param[out] ctx Pointer to receive context handle
//!
//! \pre ctx must not be NULL
//! \post *ctx contains valid context handle that must be finalized with stf_ctx_finalize()
//!
//! \note Graph backend has restrictions on stream synchronization within tasks
//!
//! \par Example:
//! \code
//! stf_ctx_handle ctx;
//! stf_ctx_create_graph(&ctx);
//! // ... use context ...
//! stf_ctx_finalize(ctx);
//! \endcode
//!
//! \see stf_ctx_create(), stf_ctx_finalize()

void stf_ctx_create_graph(stf_ctx_handle* ctx);

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
//! stf_ctx_handle ctx;
//! stf_ctx_create(&ctx);
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
//! stf_ctx_handle ctx;
//! stf_ctx_create(&ctx);
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
//! Creates logical data handle from an existing host memory buffer.
//! STF takes ownership of data management during task execution.
//!
//! \param ctx Context handle
//! \param[out] ld Pointer to receive logical data handle
//! \param addr Pointer to existing data buffer
//! \param sz Size of data in bytes
//!
//! \pre ctx must be valid context handle
//! \pre ld must not be NULL
//! \pre addr must not be NULL
//! \pre sz must be greater than 0
//! \post *ld contains valid logical data handle
//!
//! \note Original data pointer should not be accessed during task execution
//! \note Data will be written back when logical data is destroyed or context finalized
//!
//! \par Example:
//! \code
//! float data[1024];
//! stf_logical_data_handle ld;
//! stf_logical_data(ctx, &ld, data, sizeof(data));
//! // ... use in tasks ...
//! stf_logical_data_destroy(ld);
//! \endcode
//!
//! \see stf_logical_data_empty(), stf_logical_data_destroy()

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz);

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
//! stf_logical_data_handle ld;
//! stf_logical_data(ctx, &ld, data, size);
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
//! stf_logical_data_handle ld;
//! stf_logical_data(ctx, &ld, data, size);
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
//! \param[out] to Pointer to receive logical data handle
//!
//! \pre ctx must be valid context handle
//! \pre length must be greater than 0
//! \pre to must not be NULL
//! \post *to contains valid logical data handle
//!
//! \note First access must be write-only (STF_WRITE)
//! \note No write-back occurs since there's no host backing
//!
//! \par Example:
//! \code
//! stf_logical_data_handle temp;
//! stf_logical_data_empty(ctx, 1024 * sizeof(float), &temp);
//!
//! // First access must be write-only
//! stf_task_add_dep(task, temp, STF_WRITE);
//! \endcode
//!
//! \see stf_logical_data(), stf_logical_data_destroy()

void stf_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle* to);

//!
//! \brief Create synchronization token
//!
//! Creates a logical data handle for synchronization purposes only.
//! Contains no actual data but can be used to enforce execution order.
//!
//! \param ctx Context handle
//! \param[out] ld Pointer to receive token handle
//!
//! \pre ctx must be valid context handle
//! \pre ld must not be NULL
//! \post *ld contains valid token handle
//!
//! \note More efficient than using dummy data for synchronization
//! \note Can be accessed with any access mode
//!
//! \par Example:
//! \code
//! stf_logical_data_handle sync_token;
//! stf_token(ctx, &sync_token);
//!
//! // Task 1 signals completion
//! stf_task_add_dep(task1, sync_token, STF_WRITE);
//!
//! // Task 2 waits for task1
//! stf_task_add_dep(task2, sync_token, STF_READ);
//! \endcode
//!
//! \see stf_logical_data(), stf_logical_data_destroy()

void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld);

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
//! \param[out] t Pointer to receive task handle
//!
//! \pre ctx must be valid context handle
//! \pre t must not be NULL
//! \post *t contains valid task handle
//!
//! \par Example:
//! \code
//! stf_task_handle task;
//! stf_task_create(ctx, &task);
//! // ... configure task ...
//! stf_task_destroy(task);
//! \endcode
//!
//! \see stf_task_destroy(), stf_task_set_exec_place(), stf_task_add_dep()

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t);

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
//! stf_task_handle task;
//! stf_task_create(ctx, &task);
//!
//! // Execute on device 1
//! stf_exec_place place = make_device_place(1);
//! stf_task_set_exec_place(task, &place);
//! \endcode
//!
//! \see make_device_place(), make_host_place()

void stf_task_set_exec_place(stf_task_handle t, stf_exec_place* exec_p);

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
//! stf_task_handle task;
//! stf_task_create(ctx, &task);
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
//! stf_data_place dplace = make_device_data_place(0);
//! stf_task_add_dep_with_dplace(task, ld, STF_READ, &dplace);
//! \endcode
//!
//! \see stf_task_add_dep(), make_device_data_place(), make_host_data_place()

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place* data_p);

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
//! stf_task_handle task;
//! stf_task_create(ctx, &task);
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
//! \param[out] k Pointer to receive kernel handle
//!
//! \pre ctx must be valid context handle
//! \pre k must not be NULL
//! \post *k contains valid kernel handle
//!
//! \par Example:
//! \code
//! stf_cuda_kernel_handle kernel;
//! stf_cuda_kernel_create(ctx, &kernel);
//! // ... configure kernel ...
//! stf_cuda_kernel_destroy(kernel);
//! \endcode
//!
//! \see stf_cuda_kernel_destroy(), stf_task_create()

void stf_cuda_kernel_create(stf_ctx_handle ctx, stf_cuda_kernel_handle* k);

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
//! \see make_device_place(), stf_task_set_exec_place()

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place* exec_p);

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

#ifdef __cplusplus
}
#endif
