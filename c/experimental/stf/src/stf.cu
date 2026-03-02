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

#include <vector>

using namespace cuda::experimental::stf;

namespace
{
// Thread-local C partition function used when converting composite data_place to C++.
// Set in to_data_place() when handling STF_DATA_PLACE_COMPOSITE so the thunk can call it.
thread_local stf_get_executor_fn g_c_mapper = nullptr;

// C++ thunk: converts pos4/dim4 to C types, calls the user's C (or Python) mapper, converts result back.
pos4 call_c_mapper(pos4 data_coords, dim4 data_dims, dim4 grid_dims)
{
  stf_get_executor_fn fn = g_c_mapper;
  _CCCL_ASSERT(fn != nullptr, "composite mapper not set");
  stf_pos4 c_coords{data_coords.x, data_coords.y, data_coords.z, data_coords.t};
  stf_dim4 c_data_dims{data_dims.x, data_dims.y, data_dims.z, data_dims.t};
  stf_dim4 c_grid_dims{grid_dims.x, grid_dims.y, grid_dims.z, grid_dims.t};
  stf_pos4 c_result = fn(c_coords, c_data_dims, c_grid_dims);
  return pos4(c_result.x, c_result.y, c_result.z, c_result.t);
}
} // namespace

extern "C" {

/* Convert the C-API stf_data_place to a C++ data_place object */
static data_place to_data_place(stf_data_place* data_p)
{
  assert(data_p);

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
      _CCCL_ASSERT(grid_handle != nullptr && mapper != nullptr, "Invalid composite data place (null grid or mapper)");
      if (!grid_handle || !mapper)
      {
        return data_place::invalid();
      }
      exec_place_grid* grid_ptr = static_cast<exec_place_grid*>(grid_handle);
      g_c_mapper                = mapper;
      data_place result         = data_place::composite(&call_c_mapper, *grid_ptr);
      g_c_mapper                = nullptr;
      return result;
    }

    default:
      assert(false && "Invalid data place kind");
      return data_place::invalid(); // invalid data_place
  }
}

void stf_ctx_create(stf_ctx_handle* ctx)
{
  assert(ctx);
  *ctx = new context{};
}

void stf_ctx_create_graph(stf_ctx_handle* ctx)
{
  assert(ctx);
  *ctx = new context{graph_ctx()};
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  assert(ctx);
  auto* context_ptr = static_cast<context*>(ctx);
  context_ptr->finalize();
  delete context_ptr;
}

cudaStream_t stf_fence(stf_ctx_handle ctx)
{
  assert(ctx);
  auto* context_ptr = static_cast<context*>(ctx);
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
  assert(ctx);
  assert(ld);

  auto* context_ptr = static_cast<context*>(ctx);

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
      assert(false && "Invalid data_place kind");
      cpp_dplace = cuda::experimental::stf::data_place::host(); // fallback
      break;
  }

  // Create logical data with the specified data place
  auto ld_typed = context_ptr->logical_data(make_slice((char*) addr, sz), cpp_dplace);

  // Store the logical_data_untyped directly as opaque pointer
  *ld = new logical_data_untyped{ld_typed};
}

void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  assert(ld);
  assert(symbol);

  auto* ld_ptr = static_cast<logical_data_untyped*>(ld);
  ld_ptr->set_symbol(symbol);
}

void stf_logical_data_destroy(stf_logical_data_handle ld)
{
  assert(ld);

  auto* ld_ptr = static_cast<logical_data_untyped*>(ld);
  delete ld_ptr;
}

void stf_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle* to)
{
  assert(ctx);
  assert(to);

  auto* context_ptr = static_cast<context*>(ctx);
  auto ld_typed     = context_ptr->logical_data(shape_of<slice<char>>(length));
  *to               = new logical_data_untyped{ld_typed};
}

void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld)
{
  assert(ctx);
  assert(ld);

  auto* context_ptr = static_cast<context*>(ctx);
  *ld               = new logical_data_untyped{context_ptr->token()};
}

/* Convert the C-API stf_exec_place to a C++ exec_place object */
exec_place to_exec_place(stf_exec_place* exec_p)
{
  assert(exec_p);

  switch (exec_p->kind)
  {
    case STF_EXEC_PLACE_HOST:
      return exec_place::host();

    case STF_EXEC_PLACE_DEVICE:
      return exec_place::device(exec_p->u.device.dev_id);

    default:
      assert(false && "Invalid execution place kind");
      return exec_place{}; // invalid exec_place
  }
}

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  assert(ctx);
  assert(t);

  auto* context_ptr = static_cast<context*>(ctx);
  *t                = new context::unified_task<>{context_ptr->task()};
}

void stf_task_set_exec_place(stf_task_handle t, stf_exec_place* exec_p)
{
  assert(t);
  assert(exec_p);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->set_exec_place(to_exec_place(exec_p));
}

void stf_task_set_symbol(stf_task_handle t, const char* symbol)
{
  assert(t);
  assert(symbol);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->set_symbol(symbol);
}

void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  assert(t);
  assert(ld);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = static_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place* data_p)
{
  assert(t);
  assert(ld);
  assert(data_p);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = static_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m), to_data_place(data_p)));
}

void* stf_task_get(stf_task_handle t, int index)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  auto s         = task_ptr->template get<slice<const char>>(index);
  return (void*) s.data_handle();
}

void stf_task_start(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->start();
}

void stf_task_end(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->end();
}

void stf_task_enable_capture(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->enable_capture();
}

CUstream stf_task_get_custream(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  return static_cast<CUstream>(task_ptr->get_stream());
}

void stf_task_destroy(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
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
  assert(ctx);
  assert(k);

  auto* context_ptr = static_cast<context*>(ctx);
  using kernel_type = decltype(context_ptr->cuda_kernel());
  *k                = new kernel_type{context_ptr->cuda_kernel()};
}

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place* exec_p)
{
  assert(k);
  assert(exec_p);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);
  kernel_ptr->set_exec_place(to_exec_place(exec_p));
}

void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
{
  assert(k);
  assert(symbol);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);
  kernel_ptr->set_symbol(symbol);
}

void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
{
  assert(k);
  assert(ld);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);
  auto* ld_ptr      = static_cast<logical_data_untyped*>(ld);
  kernel_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
{
  assert(k);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);
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
  assert(k);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);

  cuda_kernel_desc desc;
  desc.configure_raw(cufunc, grid_dim_, block_dim_, shared_mem_, arg_cnt, args);
  kernel_ptr->add_kernel_desc(mv(desc));
}

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
{
  assert(k);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);
  auto s            = kernel_ptr->template get<slice<const char>>(index);
  return (void*) (s.data_handle());
}

void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
{
  assert(k);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(k);
  kernel_ptr->end();
}

void stf_cuda_kernel_destroy(stf_cuda_kernel_handle t)
{
  assert(t);

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = static_cast<kernel_type*>(t);
  delete kernel_ptr;
}

// -----------------------------------------------------------------------------
// Composite data place and execution place grid (for Python/cuTile multi-stream)
// -----------------------------------------------------------------------------

stf_exec_place_grid_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count)
{
  assert(device_ids != nullptr || count == 0);
  ::std::vector<exec_place> places;
  places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    places.push_back(exec_place::device(device_ids[i]));
  }
  exec_place_grid grid = make_grid(::std::move(places));
  return new exec_place_grid(::std::move(grid));
}

stf_exec_place_grid_handle
stf_exec_place_grid_create(const stf_exec_place* places, size_t count, const stf_dim4* grid_dims)
{
  assert(places != nullptr || count == 0);
  ::std::vector<exec_place> cpp_places;
  cpp_places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    cpp_places.push_back(to_exec_place(const_cast<stf_exec_place*>(&places[i])));
  }
  exec_place_grid grid =
    (grid_dims != nullptr)
      ? make_grid(::std::move(cpp_places), dim4(grid_dims->x, grid_dims->y, grid_dims->z, grid_dims->t))
      : make_grid(::std::move(cpp_places));
  return new exec_place_grid(::std::move(grid));
}

void stf_exec_place_grid_destroy(stf_exec_place_grid_handle grid)
{
  if (grid != nullptr)
  {
    delete static_cast<exec_place_grid*>(grid);
  }
}

void stf_make_composite_data_place(stf_data_place* out, stf_exec_place_grid_handle grid, stf_get_executor_fn mapper)
{
  assert(out != nullptr);
  assert(grid != nullptr);
  assert(mapper != nullptr);
  out->kind               = STF_DATA_PLACE_COMPOSITE;
  out->u.composite.grid   = grid;
  out->u.composite.mapper = mapper;
}

} // extern "C"
