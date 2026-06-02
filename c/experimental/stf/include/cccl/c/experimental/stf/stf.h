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
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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

//! \brief Opaque handle to a \c green_context_helper.
typedef struct stf_green_context_helper_opaque_t* stf_green_context_helper_handle;

//! \brief Opaque handle to an active exec_place_scope (RAII context activation).
typedef struct stf_exec_place_scope_opaque_t* stf_exec_place_scope_handle;

//! \brief Opaque handle to an \c exec_place_resources registry.
//!
//! Handles returned by stf_exec_place_resources_create() are owned by the
//! caller and must be released with stf_exec_place_resources_destroy().
//! Handles returned by stf_ctx_get_place_resources() do not own the context
//! resources, but the handle itself should still be released with
//! stf_exec_place_resources_destroy().
typedef struct stf_exec_place_resources_opaque_t* stf_exec_place_resources_handle;

//! \brief Forward declaration of \c stf_ctx_handle (full definition appears
//! below in the context section). Declared here so the
//! stf_ctx_get_place_resources() accessor can refer to it.
typedef struct stf_ctx_handle_t* stf_ctx_handle;

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
//! Writes the result into \p *result. The out-pointer convention is used
//! instead of return-by-value so that the signature is trivially representable
//! in FFI frameworks (ctypes, cffi, Rust) that cannot return C structs.
typedef void (*stf_get_executor_fn)(stf_pos4* result, stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims);

//! \brief Create host execution place (CPU).
stf_exec_place_handle stf_exec_place_host(void);

//! \brief Create device execution place for CUDA device \p dev_id.
stf_exec_place_handle stf_exec_place_device(int dev_id);

//! \brief Create execution place for the current CUDA device.
stf_exec_place_handle stf_exec_place_current_device(void);

//! \brief Create a green-context helper for \p dev_id with \p sm_count SMs per green context.
//! Requires CUDA 12.4+. Returns NULL on failure.
stf_green_context_helper_handle stf_green_context_helper_create(int sm_count, int dev_id);

//! \brief Destroy a green-context helper handle.
void stf_green_context_helper_destroy(stf_green_context_helper_handle h);

//! \brief Number of green contexts created by \p h.
size_t stf_green_context_helper_get_count(stf_green_context_helper_handle h);

//! \brief Device ordinal used by this green-context helper.
int stf_green_context_helper_get_device_id(stf_green_context_helper_handle h);

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

//! \brief Activate the sub-place at linear index \p idx (0 for scalar places).
//! Saves the current CUDA context; call stf_exec_place_scope_exit to restore.
//! \return Opaque scope handle, or NULL on failure.
stf_exec_place_scope_handle stf_exec_place_scope_enter(stf_exec_place_handle place, size_t idx);

//! \brief Restore the CUDA context saved by stf_exec_place_scope_enter and destroy the scope.
//! \p scope may be NULL (no-op).
void stf_exec_place_scope_exit(stf_exec_place_scope_handle scope);

//! \brief Get the affine data_place associated with this exec_place.
//! Caller must stf_data_place_destroy the result.
stf_data_place_handle stf_exec_place_get_affine_data_place(stf_exec_place_handle h);

//! \brief Create a fresh, empty exec_place_resources registry.
//!
//! The registry lazily creates and owns stream pools for places used with
//! stf_exec_place_pick_stream(). Destroying it releases every stream it owns.
stf_exec_place_resources_handle stf_exec_place_resources_create(void);

//! \brief Destroy a registry returned by stf_exec_place_resources_create().
//!
//! For handles returned by stf_ctx_get_place_resources(), this releases only
//! the C handle wrapper and leaves the context-owned resources untouched.
//! \p h may be NULL.
void stf_exec_place_resources_destroy(stf_exec_place_resources_handle h);

//! \brief Pick a CUDA stream for \p h from the pools owned by \p res.
//!
//! \p for_computation is a hint: non-zero requests a compute stream, zero
//! requests a data-transfer stream. The returned stream is owned by \p res and
//! remains valid until \p res is destroyed, or until the owning context is
//! finalized for a borrowed registry.
CUstream stf_exec_place_pick_stream(stf_exec_place_resources_handle res, stf_exec_place_handle h, int for_computation);

//! \brief Get the sub-place at linear index \p idx.
//! For scalar places, \p idx must be 0. Returns NULL if \p idx is out of bounds.
//! Caller must stf_exec_place_destroy the result.
stf_exec_place_handle stf_exec_place_get_place(stf_exec_place_handle h, size_t idx);

//! \brief Create an exec_place from green-context helper \p helper and view index \p idx.
//! If \p use_green_ctx_data_place is non-zero, set the affine data_place to a green-context data place.
//! Returns NULL on failure or if \p idx is out of range.
stf_exec_place_handle
stf_exec_place_green_ctx(stf_green_context_helper_handle helper, size_t idx, int use_green_ctx_data_place);

//! \brief Initialize the machine singleton (P2P access, memory pool setup, topology).
//! Safe to call multiple times; only the first call has effect.
void stf_machine_init(void);

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

//! \brief Create a data_place from green-context helper \p helper and view index \p idx.
//! Returns NULL on failure or if \p idx is out of range.
stf_data_place_handle stf_data_place_green_ctx(stf_green_context_helper_handle helper, size_t idx);

//! \brief Deep copy (caller must stf_data_place_destroy).
stf_data_place_handle stf_data_place_clone(stf_data_place_handle h);

//! \brief Release a data place handle.
void stf_data_place_destroy(stf_data_place_handle h);

//! \brief Device ordinal from \c data_place_interface::get_device_ordinal() (see C++ docs for sentinel values).
int stf_data_place_get_device_ordinal(stf_data_place_handle h);

//! \brief Human-readable description; pointer valid until the next call on this thread.
const char* stf_data_place_to_string(stf_data_place_handle h);

//! \brief Allocate \p size bytes at this data place.
//!
//! For device places the allocation is stream-ordered (cudaMallocAsync).
//! For host/managed places \p stream is ignored.
//! Returns NULL on failure (e.g. unsupported place type or out of memory).
//!
//! \param h     Data place handle (must not be NULL)
//! \param size  Allocation size in bytes
//! \param stream CUDA stream for stream-ordered allocation (may be NULL)
//! \return Pointer to allocated memory, or NULL on failure
void* stf_data_place_allocate(stf_data_place_handle h, ptrdiff_t size, cudaStream_t stream);

//! \brief Deallocate memory previously obtained from stf_data_place_allocate().
//!
//! For device places the deallocation is stream-ordered (cudaFreeAsync).
//! For host/managed places \p stream is ignored.
//!
//! \param h      Data place handle (must not be NULL)
//! \param ptr    Pointer returned by stf_data_place_allocate()
//! \param size   Size of the original allocation in bytes
//! \param stream CUDA stream for stream-ordered deallocation (may be NULL)
void stf_data_place_deallocate(stf_data_place_handle h, void* ptr, size_t size, cudaStream_t stream);

//! \brief Query whether allocations on this place are stream-ordered.
//!
//! \param h Data place handle (must not be NULL)
//! \return 1 if stream-ordered, 0 otherwise
int stf_data_place_allocation_is_stream_ordered(stf_data_place_handle h);

//! \}

//! \defgroup Handles Opaque Handles
//! \brief Opaque handle types for STF objects
//! \{

//!
//! \brief Opaque handle for STF context
//!
//! Context stores the state of the STF library and serves as entry point for all API calls.
//! Must be created with stf_ctx_create() or stf_ctx_create_graph() and destroyed with stf_ctx_finalize().
//! (Forward declared earlier in the place section.)

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
//! \brief Opaque handle to a shared `async_resources_handle`
//!
//! Wraps the C++ `async_resources_handle` so callers can build one up front
//! and share it across many `stf_ctx_create_ex` calls. Reusing a handle lets
//! the graph backend amortize graph-instantiation cost across contexts, and
//! lets every context share the same per-place stream pools.
//!
//! Ownership: the handle is caller-allocated via
//! stf_async_resources_create() and MUST be released with
//! stf_async_resources_destroy().
//!
//! Lifetime: the handle owns CUDA resources (per-place stream pools, an
//! executable-graph cache, etc.) that are torn down synchronously by
//! stf_async_resources_destroy(). The caller must therefore not destroy the
//! handle while any work that could reference those resources is still
//! pending. Concretely:
//!   - Every context constructed with the handle must have been finalized
//!     via stf_ctx_finalize().
//!   - For contexts created with `has_stream != 0`, stf_ctx_finalize() is
//!     non-blocking (see stf_ctx_finalize()): each such caller stream must
//!     reach the point at which finalize enqueued its resource-release work
//!     (e.g. via cudaStreamSynchronize() on every caller stream that was
//!     ever passed in `opts.stream`) before stf_async_resources_destroy()
//!     is called.

typedef struct stf_async_resources_opaque_t* stf_async_resources_handle;

//!
//! \brief Create a shareable `async_resources_handle`
//!
//! \return Handle on success, NULL on allocation failure.
//!
//! \post Caller owns the handle and must release it with
//!       stf_async_resources_destroy() after every context that received it
//!       has been finalized and, for caller-stream contexts, after each such
//!       caller stream has completed the work enqueued by
//!       stf_ctx_finalize().

stf_async_resources_handle stf_async_resources_create(void);

//!
//! \brief Destroy a handle created by stf_async_resources_create()
//!
//! \param h Handle to destroy. NULL is accepted (no-op).
//!
//! \pre Every context created with this handle has already been finalized,
//!      and every caller stream used by such contexts has completed the work
//!      stf_ctx_finalize() enqueued on it (e.g. via cudaStreamSynchronize()).
//!      Destroying the handle while caller-stream work is still pending is
//!      undefined behavior: this call synchronously tears down the
//!      underlying CUDA resources (stream pools, cached executable graphs)
//!      and does not itself synchronize any caller stream.

void stf_async_resources_destroy(stf_async_resources_handle h);

//! \brief Backend selector for stf_ctx_create_ex()
typedef enum stf_backend_kind
{
  STF_BACKEND_STREAM = 0, //!< Default stream-backed backend (eager, same as stf_ctx_create())
  STF_BACKEND_GRAPH  = 1, //!< CUDA-graph-backed backend (same as stf_ctx_create_graph())
} stf_backend_kind;

//!
//! \brief Options for stf_ctx_create_ex()
//!
//! All fields are optional. Zero-initialize and set only what you need; the
//! remaining fields keep their default meaning. Treat this struct as
//! append-only: new knobs may be added at the end in future releases, so
//! always zero the struct before populating it.

typedef struct stf_ctx_options
{
  stf_backend_kind backend; //!< Backend selector (default: STF_BACKEND_STREAM)
  int has_stream; //!< 0: no caller stream; non-zero: inherit `stream`
  cudaStream_t stream; //!< Caller-owned stream (used iff `has_stream != 0`).
                       //!< `cudaStream_t` is a pointer; `nullptr` is the NULL stream,
                       //!< not a sentinel -- use `has_stream` to say "no stream".
  stf_async_resources_handle handle; //!< Shared resources handle, or NULL for "create fresh"
} stf_ctx_options;

//!
//! \brief Create an STF context with optional stream/handle/backend selection
//!
//! Unified factory covering every combination of:
//!   - backend (stream vs CUDA graph),
//!   - caller-provided `cudaStream_t` to inherit,
//!   - caller-provided shared `stf_async_resources_handle`.
//!
//! Passing `opts == NULL` is equivalent to stf_ctx_create() (stream backend,
//! default stream, fresh resources handle).
//!
//! \param opts Zero-initialized options struct, or NULL for all defaults.
//! \return Context handle, or NULL if allocation failed.
//!
//! \post On success, caller must finalize with stf_ctx_finalize().
//!
//! \par Example (reuse one resources handle across many graph contexts):
//! \code
//! cudaStream_t user_stream = ...;          // caller-owned CUDA stream
//! stf_async_resources_handle h = stf_async_resources_create();
//! for (int i = 0; i < N; ++i) {
//!   stf_ctx_options opts = {0};
//!   opts.backend    = STF_BACKEND_GRAPH;
//!   opts.has_stream = 1;           // opt-in to caller stream binding
//!   opts.stream     = user_stream; // may be 0 for the default/NULL stream
//!   opts.handle     = h;
//!   stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
//!   // ... submit tasks ...
//!   stf_ctx_finalize(ctx); // Non-blocking: see stf_ctx_finalize().
//! }
//! // Required before destroying `h`: stf_ctx_finalize() enqueued the
//! // resource-release callbacks of every context on `user_stream`, and
//! // stf_async_resources_destroy() tears down the underlying CUDA resources
//! // synchronously. The same applies to the stream backend.
//! cudaStreamSynchronize(user_stream);
//! stf_async_resources_destroy(h);
//! \endcode

stf_ctx_handle stf_ctx_create_ex(const stf_ctx_options* opts);

//!
//! \brief Finalize STF context
//!
//! Performs write-back of modified data to host and releases all resources
//! associated with the context. Whether the call blocks until the underlying
//! CUDA work has actually completed depends on how the context was created:
//!
//!   - Contexts created with stf_ctx_create(), stf_ctx_create_graph(), or
//!     stf_ctx_create_ex() with `has_stream == 0` block until all pending
//!     operations have completed before returning.
//!   - Contexts created with stf_ctx_create_ex() and `has_stream != 0`
//!     enqueue the remaining work and the context's resource-release callback
//!     onto the caller-provided `opts.stream`, then return without
//!     synchronizing that stream. The context handle itself is invalid as
//!     soon as the call returns, but the queued CUDA work, and any
//!     resources kept alive by it (including a shared
//!     stf_async_resources_handle), only become idle once
//!     `opts.stream` reaches that completion point. Use
//!     cudaStreamSynchronize() (or an event/dependency on `opts.stream`)
//!     before observing results on the host or releasing the shared handle.
//!
//! \param ctx Context handle to finalize
//!
//! \pre ctx must be valid context handle
//! \post All pending operations are either completed (blocking case) or
//!       enqueued on the caller stream (non-blocking case); resources are
//!       released; ctx becomes invalid.
//!
//! \par Example (blocking, default-created context):
//! \code
//! stf_ctx_handle ctx = stf_ctx_create();
//! // ... submit tasks ...
//! stf_ctx_finalize(ctx);  // Blocks until completion
//! \endcode
//!
//! \par Example (non-blocking, caller-provided stream):
//! \code
//! stf_ctx_options opts = {0};
//! opts.has_stream = 1;
//! opts.stream     = user_stream;
//! stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
//! // ... submit tasks ...
//! stf_ctx_finalize(ctx); // Returns before user_stream drains.
//! cudaStreamSynchronize(user_stream); // Required before observing results.
//! \endcode
//!
//! \see stf_ctx_create(), stf_ctx_create_graph(), stf_ctx_create_ex(),
//!      stf_fence()

void stf_ctx_finalize(stf_ctx_handle ctx);

//! \brief Borrow the per-place stream-pool registry embedded in \p ctx.
//!
//! The returned handle refers to resources that remain valid until
//! stf_ctx_finalize(ctx). Release the handle with
//! stf_exec_place_resources_destroy(); doing so does not destroy the
//! context-owned resources.
stf_exec_place_resources_handle stf_ctx_get_place_resources(stf_ctx_handle ctx);

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

//!
//! \brief Synchronize and copy logical data contents to a host buffer
//!
//! Schedules a host callback that reads the logical data, synchronizes
//! to ensure the callback completes, and copies the data into the
//! caller-provided buffer. Unlike stf_ctx_finalize(), the context
//! remains usable after this call, enabling iterative patterns such as
//! convergence checks.
//!
//! \param ctx   Context handle
//! \param ld    Logical data handle to read
//! \param out   Destination host buffer
//! \param size  Size of the destination buffer in bytes
//! \return 0 on success, non-zero on error
//!
//! \pre  ctx and ld must be valid handles; out must not be NULL
//! \pre  The first min(size, data_size) bytes of out must not overlap the
//!       logical data range associated with ld.
//! \post The first min(size, data_size) bytes of the logical data are
//!       written to out.
//!
//! \par Example:
//! \code
//! int h_sum = 0;
//! stf_ctx_wait(ctx, lSum, &h_sum, sizeof(h_sum));
//! // h_sum now contains the result; context is still active
//! \endcode
//!
//! \see stf_fence(), stf_ctx_finalize()

int stf_ctx_wait(stf_ctx_handle ctx, stf_logical_data_handle ld, void* out, size_t size);

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

//!
//! \brief Get grid dimensions of a task's exec place
//!
//! When the task's execution place is a grid (size > 1), writes its
//! shape to \p out_dims. Returns 0 on success, non-zero if the task's
//! exec place is not a grid or \p out_dims is NULL.
//!
//! \param t Task handle
//! \param[out] out_dims On success, the grid shape (x, y, z, t) is written here. Must not be NULL.
//! \return 0 on success; non-zero if task exec place is not a grid or \p out_dims is NULL
//!
//! \pre t must be valid task handle
//! \pre stf_task_start() must have been called
//!
//! \note Total number of grid entries is out_dims->x * out_dims->y * out_dims->z * out_dims->t.
//!
//! \par Example:
//! \code
//! stf_task_start(task);
//! stf_dim4 dims;
//! if (stf_task_get_grid_dims(task, &dims) == 0) {
//!     printf("Grid: %lu x %lu\n", dims.x, dims.y);
//! }
//! \endcode
//!
//! \see stf_task_get_custream_at_index()

int stf_task_get_grid_dims(stf_task_handle t, stf_dim4* out_dims);

//!
//! \brief Get the CUDA stream for a specific grid index
//!
//! When the task's exec place is a grid, returns the CUstream for the
//! given linear index (0 to product of grid dims - 1).
//!
//! \param t Task handle (must have been started; exec place must be a grid)
//! \param place_index Linear index in the grid (0-based; use stf_task_get_grid_dims to get shape)
//! \param[out] out_stream On success, the stream for that index is written here. Must not be NULL.
//! \return 0 on success; non-zero if task is not a grid, index out of range, or no per-index streams
//!
//! \pre t must be valid task handle
//! \pre stf_task_start() must have been called
//!
//! \par Example:
//! \code
//! stf_dim4 dims;
//! stf_task_get_grid_dims(task, &dims);
//! for (size_t i = 0; i < dims.x; ++i) {
//!     CUstream s;
//!     stf_task_get_custream_at_index(task, i, &s);
//!     // launch work on stream s
//! }
//! \endcode
//!
//! \see stf_task_get_grid_dims()

int stf_task_get_custream_at_index(stf_task_handle t, size_t place_index, CUstream* out_stream);

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

//! \defgroup StackableContext Stackable Context
//! \brief Hierarchical context with nested graph scopes, while loops, and repeat loops
//!
//! \details
//! A stackable context exposes the same task / logical-data programming model
//! as the regular STF context, but allows pushing nested scopes that capture
//! work into CUDA child graphs. Three flavours of scope are supported:
//!  - \c stf_stackable_push_graph / \c stf_stackable_pop : a plain nested graph,
//!  - \c stf_stackable_push_while / \c stf_stackable_pop_while : a CUDA 12.4+
//!    conditional while loop (the loop body re-executes while a CUDA conditional
//!    handle is set to non-zero),
//!  - \c stf_stackable_push_repeat / \c stf_stackable_pop_repeat : a fixed
//!    iteration counter built on top of the while-scope primitive.
//!
//! A stackable context handle is interchangeable with a regular
//! \c stf_ctx_handle for typing purposes (it points at a different C++ object
//! internally). The user must always pair a \c stf_stackable_ctx_create
//! with \c stf_stackable_ctx_finalize, and pair every \c push with the matching
//! \c pop in LIFO order.
//!
//! Stackable logical data is allocated with \c stf_stackable_logical_data*
//! and consumed by \c stf_stackable_task_create / \c stf_stackable_task_add_dep
//! and \c stf_stackable_host_launch_create / \c stf_stackable_host_launch_add_dep.
//! Crossing a scope boundary is handled implicitly: when a logical data is
//! first accessed inside a deeper scope STF auto-pushes the value through the
//! intermediate contexts.
//!
//! \par Stackable Usage Pattern:
//! \code
//! stf_ctx_handle sctx = stf_stackable_ctx_create();
//!
//! float buf[N];
//! stf_logical_data_handle lA = stf_stackable_logical_data(sctx, buf, sizeof(buf));
//!
//! stf_stackable_push_graph(sctx);                        // Begin nested graph scope
//! {
//!   stf_task_handle t = stf_stackable_task_create(sctx);
//!   stf_stackable_task_add_dep(sctx, t, lA, STF_RW);
//!   stf_task_start(t);
//!   /* launch CUDA work on stf_task_get_custream(t) */
//!   stf_task_end(t);
//!   stf_task_destroy(t);
//! }
//! stf_stackable_pop(sctx);                               // Instantiate child graph
//!
//! stf_stackable_logical_data_destroy(lA);
//! stf_stackable_ctx_finalize(sctx);
//! \endcode
//!
//! \warning This API is experimental and subject to change.
//! \{

//! \brief Create a stackable context (root is the stream backend)
//!
//! \return Stackable context handle (typed as \c stf_ctx_handle), or NULL on
//!         allocation failure.
//! \post On success, caller must release with \c stf_stackable_ctx_finalize().
//!
//! \note A handle returned by \c stf_stackable_ctx_create() must \b only be
//!       passed to \c stf_stackable_* entry points; mixing it with the
//!       regular \c stf_ctx_* surface is undefined behaviour.
//!
//! \see stf_stackable_ctx_finalize()
stf_ctx_handle stf_stackable_ctx_create(void);

//! \brief Finalize a stackable context and release all associated resources.
//!
//! Blocks until every pending task in every still-open scope completes.
//! \param ctx Stackable context handle (must have been popped back to the root).
//!
//! \see stf_stackable_ctx_create()
void stf_stackable_ctx_finalize(stf_ctx_handle ctx);

//! \brief Get a fence stream for a stackable context (must be at root level).
//!
//! \param ctx Stackable context handle
//! \return CUDA stream that becomes ready when all pending root-level work has
//!         been issued.
//!
//! \warning Calling \c stf_stackable_ctx_fence() inside a nested scope is not
//!          supported and will fail.
cudaStream_t stf_stackable_ctx_fence(stf_ctx_handle ctx);

//! \brief Push a plain nested graph scope onto the context stack.
//!
//! Subsequent tasks/host_launches submitted on \p ctx are captured into a
//! CUDA child graph. The graph is instantiated and launched on the parent
//! scope when \c stf_stackable_pop() is called.
//!
//! \param ctx Stackable context handle
//!
//! \see stf_stackable_pop()
void stf_stackable_push_graph(stf_ctx_handle ctx);

//! \brief Pop the innermost graph scope (must match \c stf_stackable_push_graph()).
//!
//! Use \c stf_stackable_pop_while() / \c stf_stackable_pop_repeat() to close
//! while/repeat scopes instead.
//!
//! \param ctx Stackable context handle
void stf_stackable_pop(stf_ctx_handle ctx);

//! \brief Opaque handle for a re-launchable graph produced by
//!        \c stf_stackable_pop_prologue().
//!
//! The handle remains valid between a matching \c stf_stackable_pop_prologue()
//! and \c stf_stackable_pop_epilogue() pair. Calling \c stf_launchable_graph_launch(),
//! \c stf_launchable_graph_exec() or \c stf_launchable_graph_stream() after the
//! epilogue aborts with a clear message (the underlying C++ layer invalidates
//! every outstanding copy of the handle in one shot).
//!
//! The handle wrapper itself must be released with \c stf_launchable_graph_destroy()
//! to reclaim the small heap allocation made by \c stf_stackable_pop_prologue().
typedef struct stf_launchable_graph_handle_t* stf_launchable_graph_handle;

//! \brief First phase of a two-phase pop of a top-level graph scope.
//!
//! Runs the same prologue as \c stf_stackable_pop() (pops any pushed data,
//! finalises the child graph, instantiates or fetches a \c cudaGraphExec_t
//! from the cache) but does not launch the graph and does not release
//! resources. Returns a handle the caller can use to launch the graph one
//! or more times (via \c stf_launchable_graph_launch()) before finishing
//! the pop with \c stf_stackable_pop_epilogue().
//!
//! Only legal when the innermost scope is a top-level graph (its parent is
//! the stream-backed root). Aborts otherwise.
//!
//! \param ctx Stackable context handle (must not be NULL).
//! \return Launchable graph handle (non-NULL on success; NULL only on
//!         heap-allocation failure, in which case an explanatory message
//!         is printed to stderr). Release with \c stf_launchable_graph_destroy().
//!
//! \see stf_stackable_pop_epilogue()
//! \see stf_launchable_graph_launch()
stf_launchable_graph_handle stf_stackable_pop_prologue(stf_ctx_handle ctx);

//! \brief Second phase of a two-phase pop: release resources and unfreeze data.
//!
//! Runs the deferred portion of \c stf_stackable_pop() that was skipped by
//! \c stf_stackable_pop_prologue(). Invalidates every outstanding
//! \c stf_launchable_graph_handle produced by the matching prologue; the
//! handle wrapper itself is not freed and must still be released with
//! \c stf_launchable_graph_destroy().
//!
//! \param ctx Stackable context handle (must not be NULL).
//!
//! \see stf_stackable_pop_prologue()
void stf_stackable_pop_epilogue(stf_ctx_handle ctx);

//! \brief Launch the instantiated graph once.
//!
//! On the first call, syncs the context's prerequisite events into the
//! support stream. Subsequent calls skip the sync and issue the launch
//! directly. Aborts if the handle has been invalidated by
//! \c stf_stackable_pop_epilogue().
//!
//! \param h Launchable graph handle (must not be NULL).
void stf_launchable_graph_launch(stf_launchable_graph_handle h);

//! \brief Return the underlying \c cudaGraphExec_t for advanced use
//!        (e.g. launching on a user-supplied stream).
//!
//! Aborts if \p h has been invalidated by \c stf_stackable_pop_epilogue().
//!
//! \param h Launchable graph handle (must not be NULL).
//! \return \c cudaGraphExec_t owned by the STF graph cache.
cudaGraphExec_t stf_launchable_graph_exec(stf_launchable_graph_handle h);

//! \brief Return the internal support stream the graph was prepared against.
//!
//! Aborts if \p h has been invalidated by \c stf_stackable_pop_epilogue().
//!
//! \param h Launchable graph handle (must not be NULL).
//! \return \c cudaStream_t used by the default \c stf_launchable_graph_launch().
cudaStream_t stf_launchable_graph_stream(stf_launchable_graph_handle h);

//! \brief Return the underlying (non-executable) CUDA graph topology.
//!
//! Intended for callers who want to embed the graph as a child node into
//! another graph (via \c cudaGraphAddChildGraphNode) rather than launching
//! the pre-instantiated executable graph returned by
//! \c stf_launchable_graph_exec(). Unlike that function, this accessor does
//! NOT trigger \c cudaGraphInstantiate and performs no synchronization.
//!
//! The graph stays valid only until \c stf_stackable_pop_epilogue() is
//! called. Clone it with \c cudaGraphClone if you need it to outlive the
//! epilogue.
//!
//! Aborts if \p h has been invalidated by \c stf_stackable_pop_epilogue().
//!
//! \param h Launchable graph handle (must not be NULL).
//! \return \c cudaGraph_t owned by the nested stackable context.
cudaGraph_t stf_launchable_graph_graph(stf_launchable_graph_handle h);

//! \brief Release the heap-allocated handle wrapper.
//!
//! Does not affect graph validity (that is driven by
//! \c stf_stackable_pop_epilogue()). NULL is a no-op.
//!
//! \param h Launchable graph handle (or NULL).
void stf_launchable_graph_destroy(stf_launchable_graph_handle h);

//! \brief Opaque handle for a shared-ownership, storable launchable graph.
//!
//! Returned by \c stf_stackable_pop_prologue_shared(). Every call to
//! \c stf_launchable_graph_shared_dup() produces a new opaque handle that
//! shares ownership of the same underlying CUDA graph; each of these
//! handles must be released independently with
//! \c stf_launchable_graph_shared_free(). The final \c _free call runs
//! \c stf_stackable_pop_epilogue() automatically, so users do not call the
//! epilogue manually.
//!
//! Typical use:
//! \code
//!   stf_stackable_push_graph(ctx);
//!   // ... submit tasks ...
//!   stf_launchable_graph_shared h;
//!   stf_stackable_pop_prologue_shared(ctx, &h);
//!
//!   for (int i = 0; i < 1000; ++i) {
//!     stf_launchable_graph_shared_launch(h);
//!   }
//!
//!   stf_launchable_graph_shared_free(h);   // last ref -> pop_epilogue
//! \endcode
typedef struct stf_launchable_graph_shared_t* stf_launchable_graph_shared;

//! \brief Shared-ownership flavor of \c stf_stackable_pop_prologue().
//!
//! Runs the prologue and wraps the resulting handle into a shared-ownership
//! opaque. Copies made via \c stf_launchable_graph_shared_dup() share the
//! same underlying graph; the epilogue runs when the last copy is freed.
//!
//! \param ctx Stackable context handle (must not be NULL).
//! \param out Receives the new shared handle on success (non-NULL).
//! \return Zero on success, non-zero on allocation failure.
int stf_stackable_pop_prologue_shared(stf_ctx_handle ctx, stf_launchable_graph_shared* out);

//! \brief Duplicate a shared launchable-graph handle (bumps the shared count).
//!
//! The new handle must be released separately with
//! \c stf_launchable_graph_shared_free(). Aborts if \p h is NULL.
//!
//! \param h Shared handle to duplicate (must not be NULL).
//! \param out Receives the duplicated handle on success (non-NULL).
//! \return Zero on success, non-zero on allocation failure.
int stf_launchable_graph_shared_dup(stf_launchable_graph_shared h, stf_launchable_graph_shared* out);

//! \brief Release one shared reference. When this was the last one,
//!        runs \c stf_stackable_pop_epilogue() automatically.
//!
//! NULL is a no-op, matching the pattern used by other destroy entry points.
//!
//! \param h Shared handle (or NULL).
void stf_launchable_graph_shared_free(stf_launchable_graph_shared h);

//! \brief Query whether the shared handle still refers to a live graph.
//!
//! Returns 0 for a NULL handle or after some other code path
//! (for example a manual \c stf_stackable_pop_epilogue()) has released the
//! underlying state.
int stf_launchable_graph_shared_valid(stf_launchable_graph_shared h);

//! \brief Launch the graph once. Aborts if \p h is NULL or invalid.
void stf_launchable_graph_shared_launch(stf_launchable_graph_shared h);

//! \brief Return the executable graph. Triggers lazy instantiation + dep-A
//!        sync on the first call. Aborts if \p h is NULL or invalid.
cudaGraphExec_t stf_launchable_graph_shared_exec(stf_launchable_graph_shared h);

//! \brief Return the support stream. Purely observational. Aborts if \p h
//!        is NULL or invalid.
cudaStream_t stf_launchable_graph_shared_stream(stf_launchable_graph_shared h);

//! \brief Return the underlying \c cudaGraph_t topology (for embedding as a
//!        child graph). Aborts if \p h is NULL or invalid.
cudaGraph_t stf_launchable_graph_shared_graph(stf_launchable_graph_shared h);

//! \brief Opaque handle for a while-loop scope (CUDA 12.4+).
typedef struct stf_while_scope_handle_t* stf_while_scope_handle;

//! \brief Opaque handle for a repeat-loop scope (CUDA 12.4+).
typedef struct stf_repeat_scope_handle_t* stf_repeat_scope_handle;

// Public C header: avoid libcudacxx-only macros so Python bindings (and other
// pure-C users) can include this without pulling in <cuda/std/__cccl/...>.
// CUDART_VERSION is provided by <cuda_runtime.h> already included above.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

//! \brief Push a while-loop scope (CUDA conditional graph node).
//!
//! The loop body executes at least once. Use \c stf_stackable_while_cond_scalar()
//! or fetch the raw conditional handle via \c stf_while_scope_get_cond_handle()
//! to set the per-iteration continuation flag.
//!
//! \param ctx Stackable context handle
//! \return While-scope handle, or NULL on allocation failure (caller must
//!         release with \c stf_stackable_pop_while()).
stf_while_scope_handle stf_stackable_push_while(stf_ctx_handle ctx);

//! \brief Pop (destroy) a while-loop scope opened by \c stf_stackable_push_while().
//!
//! \param scope While scope handle (NULL is a no-op).
void stf_stackable_pop_while(stf_while_scope_handle scope);

//! \brief Get the underlying \c cudaGraphConditionalHandle as a 64-bit integer.
//!
//! Useful when launching a custom kernel that calls \c cudaGraphSetConditional()
//! directly. For the common case of "continue while a scalar satisfies a
//! comparison" use \c stf_stackable_while_cond_scalar() instead.
//!
//! \param scope While scope handle
//! \return The conditional handle bit-cast to \c uint64_t.
uint64_t stf_while_scope_get_cond_handle(stf_while_scope_handle scope);

//! \brief Push a repeat scope that runs the body \p count times.
//!
//! Internally creates a counter logical data, decrements it on every
//! iteration, and feeds the result into the underlying while-scope condition.
//! The user only has to fill the body between push and pop.
//!
//! \param ctx   Stackable context handle
//! \param count Number of iterations (must be > 0)
//! \return Repeat-scope handle, or NULL on allocation failure (release with
//!         \c stf_stackable_pop_repeat()).
stf_repeat_scope_handle stf_stackable_push_repeat(stf_ctx_handle ctx, size_t count);

//! \brief Pop (destroy) a repeat scope opened by \c stf_stackable_push_repeat().
//!
//! \param scope Repeat scope handle (NULL is a no-op).
void stf_stackable_pop_repeat(stf_repeat_scope_handle scope);

//! \brief Comparison operator for built-in while conditions.
typedef enum stf_compare_op
{
  STF_CMP_GT = 0, //!< Greater than (>)
  STF_CMP_LT = 1, //!< Less than (<)
  STF_CMP_GE = 2, //!< Greater than or equal (>=)
  STF_CMP_LE = 3, //!< Less than or equal (<=)
} stf_compare_op;

//! \brief Scalar element type for \c stf_stackable_while_cond_scalar().
typedef enum stf_dtype
{
  STF_DTYPE_FLOAT32 = 0,
  STF_DTYPE_FLOAT64 = 1,
  STF_DTYPE_INT32   = 2,
  STF_DTYPE_INT64   = 3,
} stf_dtype;

//! \brief Set a built-in while-loop condition: continue while \p ld <op> \p threshold.
//!
//! Schedules an internal task that reads the scalar logical data, evaluates
//! the comparison, and updates the conditional handle accordingly. Call
//! exactly once per iteration after the loop body tasks of the current scope.
//!
//! \param ctx       Stackable context handle
//! \param scope     While scope handle
//! \param ld        Logical data handle for the scalar (1 element of \p dtype)
//! \param op        Comparison operator
//! \param threshold Right-hand side compared against the scalar
//! \param dtype     Element type of \p ld
void stf_stackable_while_cond_scalar(
  stf_ctx_handle ctx,
  stf_while_scope_handle scope,
  stf_logical_data_handle ld,
  stf_compare_op op,
  double threshold,
  stf_dtype dtype);

#endif // CUDART_VERSION >= 12040

//! \brief Create stackable logical data from existing memory and a data place.
//!
//! \param ctx    Stackable context handle
//! \param addr   Pointer to existing buffer
//! \param sz     Size of the buffer in bytes
//! \param dplace Data place describing where the buffer lives
//! \return Stackable logical data handle (typed as \c stf_logical_data_handle),
//!         or NULL on allocation failure (release with
//!         \c stf_stackable_logical_data_destroy()).
stf_logical_data_handle
stf_stackable_logical_data_with_place(stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace);

//! \brief Convenience: \c stf_stackable_logical_data_with_place() with host placement.
stf_logical_data_handle stf_stackable_logical_data(stf_ctx_handle ctx, void* addr, size_t sz);

//! \brief Create empty stackable logical data of \p length bytes (no host backing).
stf_logical_data_handle stf_stackable_logical_data_empty(stf_ctx_handle ctx, size_t length);

//! \brief Create empty stackable logical data that is local to the current stackable scope.
//!
//! The underlying logical data is created at the current head context and is
//! not exported to parent scopes. Intended for temporaries whose lifetime is
//! bounded by the enclosing stackable scope (e.g. bodies of while/repeat).
stf_logical_data_handle stf_stackable_logical_data_no_export_empty(stf_ctx_handle ctx, size_t length);

//! \brief Create a stackable synchronization token (no payload).
stf_logical_data_handle stf_stackable_token(stf_ctx_handle ctx);

//! \brief Set the symbolic name of stackable logical data (debug / DOT output).
void stf_stackable_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol);

//! \brief Mark stackable logical data as read-only (enables concurrent reads across scopes).
void stf_stackable_logical_data_set_read_only(stf_logical_data_handle ld);

//! \brief Explicitly import (push) a stackable logical data into the current
//!        (innermost) scope with the given access mode and, optionally, data
//!        place.
//!
//! By default, the very first task that touches a piece of data inside a
//! nested scope auto-pushes it through the intermediate contexts with a
//! conservative mode (typically \c STF_RW), which in turn serialises sibling
//! scopes that only need to read it. \c stf_stackable_logical_data_push() gives
//! the caller control over that import: the data is made visible in the
//! current scope with exactly \c m, so e.g. calling it with \c STF_READ from
//! inside each of several sibling graph scopes lets those scopes execute
//! concurrently without having to mark the data globally read-only via
//! \c stf_stackable_logical_data_set_read_only().
//!
//! Must be called while a stackable scope is open on \c ctx (i.e. after
//! \c stf_stackable_push_graph() / \c stf_stackable_push_while() /
//! \c stf_stackable_push_repeat() and before the matching pop).
//!
//! \param ld     Stackable logical data handle.
//! \param m      Desired access mode in the current scope.
//! \param dplace Optional data place; pass \c NULL for the default placement.
void stf_stackable_logical_data_push(stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle dplace);

//! \brief Destroy stackable logical data created by \c stf_stackable_logical_data*().
void stf_stackable_logical_data_destroy(stf_logical_data_handle ld);

//! \brief Destroy a stackable token created by \c stf_stackable_token().
//!
//! Tokens use a \c void_interface internally, so they require a dedicated
//! destroyer that knows the right C++ pointee type.
void stf_stackable_token_destroy(stf_logical_data_handle ld);

//! \brief Create a task on the head (innermost) scope of a stackable context.
//!
//! After creation, configure with \c stf_stackable_task_add_dep() (use the
//! stackable variant — \c stf_task_add_dep() will not auto-push data across
//! scopes), then call \c stf_task_start() / \c stf_task_end() and use
//! \c stf_task_get_custream() / \c stf_task_get() as usual.
//!
//! \param ctx Stackable context handle
//! \return Task handle, or NULL on allocation failure (release with
//!         \c stf_task_destroy()).
stf_task_handle stf_stackable_task_create(stf_ctx_handle ctx);

//! \brief Add a dependency to a stackable task (validates and auto-pushes data).
//!
//! \param ctx Stackable context handle (needed for auto-push validation).
//! \param t   Task handle returned by \c stf_stackable_task_create().
//! \param ld  Stackable logical data handle.
//! \param m   Access mode.
void stf_stackable_task_add_dep(stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m);

//! \brief Variant of \c stf_stackable_task_add_dep() with an explicit data place.
void stf_stackable_task_add_dep_with_dplace(
  stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p);

//! \brief Create a host launch scope on the head (innermost) scope of a stackable context.
//!
//! Configure with \c stf_stackable_host_launch_add_dep() and submit with
//! \c stf_stackable_host_launch_submit(). Other configuration (symbol,
//! user data) goes through the regular \c stf_host_launch_set_* functions.
//!
//! \param ctx Stackable context handle
//! \return Host launch handle, or NULL on allocation failure.
stf_host_launch_handle stf_stackable_host_launch_create(stf_ctx_handle ctx);

//! \brief Add a dependency to a stackable host launch scope (auto-pushes data).
void stf_stackable_host_launch_add_dep(
  stf_ctx_handle ctx, stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m);

//! \brief Submit the host callback on a stackable host launch scope.
//!
//! Equivalent to \c stf_host_launch_submit() but matched to
//! \c stf_stackable_host_launch_create() / \c stf_stackable_host_launch_destroy().
void stf_stackable_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback);

//! \brief Destroy a stackable host launch handle.
void stf_stackable_host_launch_destroy(stf_host_launch_handle h);

//! \}

#ifdef __cplusplus
}
#endif // __cplusplus
// NOLINTEND(modernize-use-using)
