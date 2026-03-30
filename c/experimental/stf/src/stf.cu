//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cccl/c/experimental/stf/stf.h>
// #include <cccl/c/parallel/include/cccl/c/extern_c.h>
#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/stf.cuh>

#include <cstddef>
#include <vector>

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

} // extern "C"
