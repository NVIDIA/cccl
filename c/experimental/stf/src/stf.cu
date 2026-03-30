//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/__stf/stackable/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

#include <cstddef>
#include <vector>

#include <cccl/c/experimental/stf/stf.h>

using namespace cuda::experimental::stf;

namespace
{
// C++ pos4/dim4 and C stf_pos4/stf_dim4 are layout-compatible (see stf.h: "Layout matches C++ pos4/dim4").
// We pass the C mapper directly to data_place::composite() via reinterpret_cast so no thunk or global is needed.
static_assert(sizeof(pos4) == sizeof(stf_pos4), "pos4 and stf_pos4 must have identical layout for C/C++ interop");
static_assert(sizeof(dim4) == sizeof(stf_dim4), "dim4 and stf_dim4 must have identical layout for C/C++ interop");
static_assert(alignof(pos4) == alignof(stf_pos4), "pos4 and stf_pos4 must have identical alignment");
static_assert(alignof(dim4) == alignof(stf_dim4), "dim4 and stf_dim4 must have identical alignment");
} // namespace

extern "C" {

/* Convert the C-API stf_data_place to a C++ data_place object */
static data_place to_data_place(const stf_data_place* data_p)
{
  _CCCL_ASSERT(data_p != nullptr, "data_place pointer must not be null");

  switch (data_p->kind)
  {
    case STF_DATA_PLACE_HOST:
      return data_place::host();

    case STF_DATA_PLACE_MANAGED:
      return data_place::managed();

    case STF_DATA_PLACE_AFFINE:
      return data_place::affine();

    case STF_DATA_PLACE_DEVICE:
      return data_place::device(data_p->u.device.dev_id);

    case STF_DATA_PLACE_COMPOSITE: {
      stf_exec_place_grid_handle grid_handle = data_p->u.composite.grid;
      stf_get_executor_fn mapper             = data_p->u.composite.mapper;
      _CCCL_ASSERT(grid_handle != nullptr, "Invalid composite data place: grid handle is null.");
      _CCCL_ASSERT(mapper != nullptr, "Invalid composite data place: partitioner function (mapper) is null.");
      if (!grid_handle || !mapper)
      {
        return data_place::invalid();
      }
      auto* grid_ptr = reinterpret_cast<exec_place*>(grid_handle);
      // Layout-compatible: pass C mapper directly so the runtime calls it
      partition_fn_t cpp_mapper = reinterpret_cast<partition_fn_t>(mapper);
      return data_place::composite(cpp_mapper, *grid_ptr);
    }

    default:
      _CCCL_ASSERT(false, "Invalid data place kind");
      return data_place::invalid(); // invalid data_place
  }
}

void stf_ctx_create(stf_ctx_handle* ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle pointer must not be null");
  *ctx = reinterpret_cast<stf_ctx_handle>(new context{});
}

void stf_ctx_create_graph(stf_ctx_handle* ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle pointer must not be null");
  *ctx = reinterpret_cast<stf_ctx_handle>(new context{graph_ctx()});
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = reinterpret_cast<context*>(ctx);
  context_ptr->finalize();
  delete context_ptr;
}

cudaStream_t stf_fence(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = reinterpret_cast<context*>(ctx);
  return context_ptr->fence();
}

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz)
{
  // Convenience wrapper: assume host memory
  stf_logical_data_with_place(ctx, ld, addr, sz, make_host_data_place());
}

void stf_logical_data_with_place(
  stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz, stf_data_place dplace)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle pointer must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);

  // Convert C data_place to C++ data_place
  cuda::experimental::stf::data_place cpp_dplace;
  switch (dplace.kind)
  {
    case STF_DATA_PLACE_HOST:
      cpp_dplace = cuda::experimental::stf::data_place::host();
      break;
    case STF_DATA_PLACE_DEVICE:
      cpp_dplace = cuda::experimental::stf::data_place::device(dplace.u.device.dev_id);
      break;
    case STF_DATA_PLACE_MANAGED:
      cpp_dplace = cuda::experimental::stf::data_place::managed();
      break;
    case STF_DATA_PLACE_AFFINE:
      cpp_dplace = cuda::experimental::stf::data_place::affine();
      break;
    case STF_DATA_PLACE_COMPOSITE:
      cpp_dplace = to_data_place(&dplace);
      break;
    default:
      // Invalid data place - this should not happen with valid input
      _CCCL_ASSERT(false, "Invalid data_place kind");
      cpp_dplace = cuda::experimental::stf::data_place::host(); // fallback
      break;
  }

  // Create logical data with the specified data place
  auto ld_typed = context_ptr->logical_data(make_slice((char*) addr, sz), cpp_dplace);

  // Store the logical_data_untyped directly as opaque pointer
  *ld = reinterpret_cast<stf_logical_data_handle>(new logical_data_untyped{ld_typed});
}

void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* ld_ptr = reinterpret_cast<logical_data_untyped*>(ld);
  ld_ptr->set_symbol(symbol);
}

void stf_logical_data_destroy(stf_logical_data_handle ld)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* ld_ptr = reinterpret_cast<logical_data_untyped*>(ld);
  delete ld_ptr;
}

void stf_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle* to)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(to != nullptr, "logical data output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  auto ld_typed     = context_ptr->logical_data(shape_of<slice<char>>(length));
  *to               = reinterpret_cast<stf_logical_data_handle>(new logical_data_untyped{ld_typed});
}

void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  *ld               = reinterpret_cast<stf_logical_data_handle>(new logical_data_untyped{context_ptr->token()});
}

/* Convert the C-API stf_exec_place to a C++ exec_place object */
exec_place to_exec_place(stf_exec_place* exec_p)
{
  _CCCL_ASSERT(exec_p != nullptr, "exec_place pointer must not be null");

  switch (exec_p->kind)
  {
    case STF_EXEC_PLACE_HOST:
      return exec_place::host();

    case STF_EXEC_PLACE_DEVICE:
      return exec_place::device(exec_p->u.device.dev_id);

    default:
      _CCCL_ASSERT(false, "Invalid execution place kind");
      return exec_place{}; // invalid exec_place
  }
}

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  *t                = reinterpret_cast<stf_task_handle>(new context::unified_task<>{context_ptr->task()});
}

void stf_task_set_exec_place(stf_task_handle t, stf_exec_place* exec_p)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(exec_p != nullptr, "exec_place pointer must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  task_ptr->set_exec_place(to_exec_place(exec_p));
}

void stf_task_set_symbol(stf_task_handle t, const char* symbol)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  task_ptr->set_symbol(symbol);
}

void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = reinterpret_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place* data_p)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(data_p != nullptr, "data_place pointer must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = reinterpret_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m), to_data_place(data_p)));
}

void* stf_task_get(stf_task_handle t, int index)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  auto s         = task_ptr->template get<slice<const char>>(index);
  return (void*) s.data_handle();
}

void stf_task_start(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  task_ptr->start();
}

void stf_task_end(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  task_ptr->end();
}

void stf_task_enable_capture(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  task_ptr->enable_capture();
}

CUstream stf_task_get_custream(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  return static_cast<CUstream>(task_ptr->get_stream());
}

void stf_task_destroy(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  delete task_ptr;
}

/**
 * Low level example of cuda_kernel(_chain)
 *   auto t = ctx.cuda_kernel_chain();
     t.add_deps(lX.read());
     t.add_deps(lY.rw());
     t->*[&]() {
     auto dX = t.template get<slice<double>>(0);
     auto dY = t.template get<slice<double>>(1);
     return std::vector<cuda_kernel_desc> {
         { axpy, 16, 128, 0, alpha, dX, dY },
         { axpy, 16, 128, 0, beta, dX, dY },
         { axpy, 16, 128, 0, gamma, dX, dY }
     };
  };

 *
 */
void stf_cuda_kernel_create(stf_ctx_handle ctx, stf_cuda_kernel_handle* k)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  using kernel_type = decltype(context_ptr->cuda_kernel());
  *k                = reinterpret_cast<stf_cuda_kernel_handle>(new kernel_type{context_ptr->cuda_kernel()});
}

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place* exec_p)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(exec_p != nullptr, "exec_place pointer must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);
  kernel_ptr->set_exec_place(to_exec_place(exec_p));
}

void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);
  kernel_ptr->set_symbol(symbol);
}

void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);
  auto* ld_ptr      = reinterpret_cast<logical_data_untyped*>(ld);
  kernel_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);
  kernel_ptr->start();
}

void stf_cuda_kernel_add_desc_cufunc(
  stf_cuda_kernel_handle k,
  CUfunction cufunc,
  dim3 grid_dim_,
  dim3 block_dim_,
  size_t shared_mem_,
  int arg_cnt,
  const void** args)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);

  cuda_kernel_desc desc;
  desc.configure_raw(cufunc, grid_dim_, block_dim_, shared_mem_, arg_cnt, args);
  kernel_ptr->add_kernel_desc(mv(desc));
}

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);
  auto s            = kernel_ptr->template get<slice<const char>>(index);
  return (void*) (s.data_handle());
}

void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto kernel_ptr   = reinterpret_cast<kernel_type*>(k);
  kernel_ptr->end();
}

void stf_cuda_kernel_destroy(stf_cuda_kernel_handle t)
{
  _CCCL_ASSERT(t != nullptr, "cuda kernel handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(t);
  delete kernel_ptr;
}

// -----------------------------------------------------------------------------
// Host launch
// -----------------------------------------------------------------------------

using host_launch_type = decltype(::std::declval<context>().host_launch());

void stf_host_launch_create(stf_ctx_handle ctx, stf_host_launch_handle* h)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(h != nullptr, "host launch handle output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  *h                = reinterpret_cast<stf_host_launch_handle>(new host_launch_type{context_ptr->host_launch()});
}

void stf_host_launch_add_dep(stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  auto* ld_ptr    = reinterpret_cast<logical_data_untyped*>(ld);
  scope_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_host_launch_set_symbol(stf_host_launch_handle h, const char* symbol)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  scope_ptr->set_symbol(symbol);
}

void stf_host_launch_set_user_data(stf_host_launch_handle h, const void* data, size_t size, void (*dtor)(void*))
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  scope_ptr->set_user_data(data, size, dtor);
}

void stf_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(callback != nullptr, "callback must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  (*scope_ptr)->*[callback](reserved::host_launch_deps& deps) {
    callback(reinterpret_cast<stf_host_launch_deps_handle>(&deps));
  };
}

void stf_host_launch_destroy(stf_host_launch_handle h)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  delete scope_ptr;
}

void* stf_host_launch_deps_get(stf_host_launch_deps_handle deps, size_t index)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<reserved::host_launch_deps*>(deps);
  return d->get<slice<char>>(index).data_handle();
}

size_t stf_host_launch_deps_get_size(stf_host_launch_deps_handle deps, size_t index)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<reserved::host_launch_deps*>(deps);
  return d->get<slice<char>>(index).extent(0);
}

size_t stf_host_launch_deps_size(stf_host_launch_deps_handle deps)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<reserved::host_launch_deps*>(deps);
  return d->size();
}

void* stf_host_launch_deps_get_user_data(stf_host_launch_deps_handle deps)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<reserved::host_launch_deps*>(deps);
  return d->user_data();
}

// -----------------------------------------------------------------------------
// Composite data place and execution place grid (for Python/cuTile multi-stream)
// -----------------------------------------------------------------------------

stf_exec_place_grid_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count)
{
  _CCCL_ASSERT(device_ids != nullptr || count == 0, "device_ids must not be null unless count is 0");
  // count must be >= 1: C++ make_grid() requires non-empty places.
  ::std::vector<exec_place> places;
  places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    places.push_back(exec_place::device(device_ids[i]));
  }
  return reinterpret_cast<stf_exec_place_grid_handle>(new exec_place(make_grid(::std::move(places))));
}

stf_exec_place_grid_handle
stf_exec_place_grid_create(const stf_exec_place* places, size_t count, const stf_dim4* grid_dims)
{
  _CCCL_ASSERT(places != nullptr || count == 0, "places must not be null unless count is 0");
  ::std::vector<exec_place> cpp_places;
  cpp_places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    cpp_places.push_back(to_exec_place(const_cast<stf_exec_place*>(&places[i])));
  }
  exec_place grid = (grid_dims != nullptr)
                    ? make_grid(::std::move(cpp_places), dim4(grid_dims->x, grid_dims->y, grid_dims->z, grid_dims->t))
                    : make_grid(::std::move(cpp_places));
  return reinterpret_cast<stf_exec_place_grid_handle>(new exec_place(::std::move(grid)));
}

void stf_exec_place_grid_destroy(stf_exec_place_grid_handle grid)
{
  if (grid != nullptr)
  {
    delete reinterpret_cast<exec_place*>(grid);
  }
}

void stf_make_composite_data_place(stf_data_place* out, stf_exec_place_grid_handle grid, stf_get_executor_fn mapper)
{
  _CCCL_ASSERT(out != nullptr, "output data_place pointer must not be null");
  _CCCL_ASSERT(grid != nullptr, "exec place grid handle must not be null");
  _CCCL_ASSERT(mapper != nullptr, "partitioner function (mapper) must not be null");
  out->kind               = STF_DATA_PLACE_COMPOSITE;
  out->u.composite.grid   = grid;
  out->u.composite.mapper = mapper;
}

// =============================================================================
// Stackable Context API
// =============================================================================

using stackable_ld_t = stackable_logical_data<slice<char>>;

void stf_stackable_ctx_create(stf_ctx_handle* ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle pointer must not be null");
  *ctx = reinterpret_cast<stf_ctx_handle>(new stackable_ctx{});
}

void stf_stackable_ctx_finalize(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  sctx->finalize();
  delete sctx;
}

cudaStream_t stf_stackable_ctx_fence(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  return sctx->fence();
}

void stf_stackable_push_graph(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  sctx->push();
}

void stf_stackable_pop(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  sctx->pop();
}

#if !_CCCL_CTK_BELOW(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION)

void stf_stackable_push_while(stf_ctx_handle ctx, stf_while_scope_handle* scope)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(scope != nullptr, "scope handle pointer must not be null");
  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  // default_launch_value=1 so the loop body executes at least once (matches the factory method)
  *scope = reinterpret_cast<stf_while_scope_handle>(
    new stackable_ctx::while_graph_scope_guard(*sctx, /*default_launch_value=*/1));
}

void stf_stackable_pop_while(stf_while_scope_handle scope)
{
  _CCCL_ASSERT(scope != nullptr, "while scope handle must not be null");
  delete reinterpret_cast<stackable_ctx::while_graph_scope_guard*>(scope);
}

uint64_t stf_while_scope_get_cond_handle(stf_while_scope_handle scope)
{
  _CCCL_ASSERT(scope != nullptr, "while scope handle must not be null");
  auto* guard = reinterpret_cast<stackable_ctx::while_graph_scope_guard*>(scope);
  return static_cast<uint64_t>(guard->cond_handle());
}

void stf_stackable_push_repeat(stf_ctx_handle ctx, size_t count, stf_repeat_scope_handle* scope)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(scope != nullptr, "scope handle pointer must not be null");
  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  *scope     = reinterpret_cast<stf_repeat_scope_handle>(new repeat_graph_scope_guard(*sctx, count));
}

void stf_stackable_pop_repeat(stf_repeat_scope_handle scope)
{
  _CCCL_ASSERT(scope != nullptr, "repeat scope handle must not be null");
  delete reinterpret_cast<repeat_graph_scope_guard*>(scope);
}

} // extern "C" — close to define C++ template kernel

// Built-in condition kernels for while loops (must be outside extern "C")
namespace
{
template <typename T>
__global__ void while_cond_kernel(const T* value, cudaGraphConditionalHandle handle, double threshold, int op)
{
  double v = static_cast<double>(*value);
  bool result;
  switch (op)
  {
    case STF_CMP_GT:
      result = v > threshold;
      break;
    case STF_CMP_LT:
      result = v < threshold;
      break;
    case STF_CMP_GE:
      result = v >= threshold;
      break;
    case STF_CMP_LE:
      result = v <= threshold;
      break;
    default:
      result = false;
      break;
  }
  cudaGraphSetConditional(handle, result ? 1 : 0);
}
} // anonymous namespace

extern "C" {

void stf_stackable_while_cond_scalar(
  stf_ctx_handle ctx,
  stf_while_scope_handle scope,
  stf_logical_data_handle ld,
  stf_compare_op op,
  double threshold,
  stf_dtype dtype)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(scope != nullptr, "while scope handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* sctx  = reinterpret_cast<stackable_ctx*>(ctx);
  auto* sld   = reinterpret_cast<stackable_ld_t*>(ld);
  auto* guard = reinterpret_cast<stackable_ctx::while_graph_scope_guard*>(scope);

  cudaGraphConditionalHandle cond_handle = guard->cond_handle();

  int offset = sctx->get_head_offset();

  // Validate access and auto-push if needed
  sld->validate_access(offset, *sctx, access_mode::read);

  // Get the underlying logical data at the current offset
  auto& underlying_ld = sld->get_ld(offset);
  logical_data_untyped ld_ut{underlying_ld};

  // Create a task on the underlying context
  auto& underlying_ctx = sctx->get_ctx(offset);
  auto task            = underlying_ctx.task();
  task.add_deps(task_dep_untyped(ld_ut, access_mode::read));
  task.set_symbol("while_condition");
  task.enable_capture();
  task.start();

  auto stream   = task.get_stream();
  auto s        = task.template get<slice<const char>>(0);
  const void* p = s.data_handle();

  switch (dtype)
  {
    case STF_DTYPE_FLOAT32:
      while_cond_kernel<float><<<1, 1, 0, stream>>>(static_cast<const float*>(p), cond_handle, threshold, op);
      break;
    case STF_DTYPE_FLOAT64:
      while_cond_kernel<double><<<1, 1, 0, stream>>>(static_cast<const double*>(p), cond_handle, threshold, op);
      break;
    case STF_DTYPE_INT32:
      while_cond_kernel<int><<<1, 1, 0, stream>>>(static_cast<const int*>(p), cond_handle, threshold, op);
      break;
    case STF_DTYPE_INT64:
      while_cond_kernel<long long><<<1, 1, 0, stream>>>(static_cast<const long long*>(p), cond_handle, threshold, op);
      break;
    default:
      _CCCL_ASSERT(false, "Unsupported dtype for while condition");
      break;
  }

  task.end();
}

#endif // !_CCCL_CTK_BELOW(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION)

// Stackable logical data operations

void stf_stackable_logical_data_with_place(
  stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz, stf_data_place dplace)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle pointer must not be null");

  auto* sctx                             = reinterpret_cast<stackable_ctx*>(ctx);
  cuda::experimental::stf::data_place dp = to_data_place(&dplace);
  auto sld                               = sctx->logical_data(make_slice(static_cast<char*>(addr), sz), dp);
  *ld                                    = reinterpret_cast<stf_logical_data_handle>(new stackable_ld_t{mv(sld)});
}

void stf_stackable_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz)
{
  stf_stackable_logical_data_with_place(ctx, ld, addr, sz, make_host_data_place());
}

void stf_stackable_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle* to)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(to != nullptr, "logical data output pointer must not be null");

  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  auto sld   = sctx->logical_data(shape_of<slice<char>>(length));
  *to        = reinterpret_cast<stf_logical_data_handle>(new stackable_ld_t{mv(sld)});
}

void stf_stackable_token(stf_ctx_handle ctx, stf_logical_data_handle* ld)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle pointer must not be null");

  auto* sctx = reinterpret_cast<stackable_ctx*>(ctx);
  // Tokens use void_interface, store as a separate type
  auto token_ld = sctx->token();
  *ld           = reinterpret_cast<stf_logical_data_handle>(new stackable_logical_data<void_interface>{mv(token_ld)});
}

void stf_stackable_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol must not be null");
  auto* sld = reinterpret_cast<stackable_ld_t*>(ld);
  sld->set_symbol(symbol);
}

void stf_stackable_logical_data_set_read_only(stf_logical_data_handle ld)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  auto* sld = reinterpret_cast<stackable_ld_t*>(ld);
  sld->set_read_only();
}

void stf_stackable_logical_data_destroy(stf_logical_data_handle ld)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  auto* sld = reinterpret_cast<stackable_ld_t*>(ld);
  delete sld;
}

void stf_stackable_token_destroy(stf_logical_data_handle ld)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  auto* token = reinterpret_cast<stackable_logical_data<void_interface>*>(ld);
  delete token;
}

// Stackable task operations

void stf_stackable_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle output pointer must not be null");

  auto* sctx           = reinterpret_cast<stackable_ctx*>(ctx);
  int offset           = sctx->get_head_offset();
  auto& underlying_ctx = sctx->get_ctx(offset);
  *t                   = reinterpret_cast<stf_task_handle>(new context::unified_task<>{underlying_ctx.task()});
}

void stf_stackable_task_add_dep(stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* sctx     = reinterpret_cast<stackable_ctx*>(ctx);
  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  auto* sld      = reinterpret_cast<stackable_ld_t*>(ld);

  int offset = sctx->get_head_offset();

  // Validate access and auto-push data across scope boundaries
  sld->validate_access(offset, *sctx, access_mode(m));

  // Get the underlying logical data at the current offset and add as untyped dep
  auto& underlying_ld = sld->get_ld(offset);
  logical_data_untyped ld_ut{underlying_ld};
  task_ptr->add_deps(task_dep_untyped(ld_ut, access_mode(m)));
}

void stf_stackable_task_add_dep_with_dplace(
  stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place* data_p)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(data_p != nullptr, "data_place pointer must not be null");

  auto* sctx     = reinterpret_cast<stackable_ctx*>(ctx);
  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  auto* sld      = reinterpret_cast<stackable_ld_t*>(ld);

  int offset = sctx->get_head_offset();

  sld->validate_access(offset, *sctx, access_mode(m));

  auto& underlying_ld = sld->get_ld(offset);
  logical_data_untyped ld_ut{underlying_ld};
  task_ptr->add_deps(task_dep_untyped(ld_ut, access_mode(m), to_data_place(data_p)));
}

// -----------------------------------------------------------------------------
// Stackable host launch (must be after stackable_ld_t typedef)
// -----------------------------------------------------------------------------

void stf_stackable_host_launch_create(stf_ctx_handle ctx, stf_host_launch_handle* h)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(h != nullptr, "host launch handle output pointer must not be null");

  auto* sctx           = reinterpret_cast<stackable_ctx*>(ctx);
  int offset           = sctx->get_head_offset();
  auto& underlying_ctx = sctx->get_ctx(offset);
  *h                   = reinterpret_cast<stf_host_launch_handle>(new host_launch_type{underlying_ctx.host_launch()});
}

void stf_stackable_host_launch_add_dep(
  stf_ctx_handle ctx, stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* sctx      = reinterpret_cast<stackable_ctx*>(ctx);
  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  auto* sld       = reinterpret_cast<stackable_ld_t*>(ld);

  int offset = sctx->get_head_offset();

  sld->validate_access(offset, *sctx, access_mode(m));

  auto& underlying_ld = sld->get_ld(offset);
  logical_data_untyped ld_ut{underlying_ld};
  scope_ptr->add_deps(task_dep_untyped(ld_ut, access_mode(m)));
}

void stf_stackable_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(callback != nullptr, "callback must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  (*scope_ptr)->*[callback](reserved::host_launch_deps& deps) {
    callback(reinterpret_cast<stf_host_launch_deps_handle>(&deps));
  };
}

void stf_stackable_host_launch_destroy(stf_host_launch_handle h)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  delete scope_ptr;
}

} // extern "C"
