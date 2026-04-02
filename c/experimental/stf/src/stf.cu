//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/places.cuh>
#include <cuda/experimental/stf.cuh>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <cccl/c/experimental/stf/stf.h>

using namespace cuda::experimental::stf;

namespace
{
static_assert(sizeof(pos4) == sizeof(stf_pos4), "pos4 and stf_pos4 must have identical layout for C/C++ interop");
static_assert(sizeof(dim4) == sizeof(stf_dim4), "dim4 and stf_dim4 must have identical layout for C/C++ interop");
static_assert(alignof(pos4) == alignof(stf_pos4), "pos4 and stf_pos4 must have identical alignment");
static_assert(alignof(dim4) == alignof(stf_dim4), "dim4 and stf_dim4 must have identical alignment");

static data_place deref_data_place(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  return *reinterpret_cast<data_place*>(h);
}

static exec_place deref_exec_place(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return *reinterpret_cast<exec_place*>(h);
}
} // namespace

extern "C" {

stf_exec_place_handle stf_exec_place_host(void)
{
  return reinterpret_cast<stf_exec_place_handle>(new exec_place(exec_place::host()));
}

stf_exec_place_handle stf_exec_place_device(int dev_id)
{
  return reinterpret_cast<stf_exec_place_handle>(new exec_place(exec_place::device(dev_id)));
}

stf_exec_place_handle stf_exec_place_current_device(void)
{
  return reinterpret_cast<stf_exec_place_handle>(new exec_place(exec_place::current_device()));
}

stf_exec_place_handle stf_exec_place_clone(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  const auto* ep = reinterpret_cast<const exec_place*>(h);
  return reinterpret_cast<stf_exec_place_handle>(new exec_place(*ep));
}

void stf_exec_place_destroy(stf_exec_place_handle h)
{
  if (h == nullptr)
  {
    return;
  }
  delete reinterpret_cast<exec_place*>(h);
}

int stf_exec_place_is_host(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return reinterpret_cast<exec_place*>(h)->is_host() ? 1 : 0;
}

int stf_exec_place_is_device(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return reinterpret_cast<exec_place*>(h)->is_device() ? 1 : 0;
}

void stf_exec_place_get_dims(stf_exec_place_handle h, stf_dim4* out_dims)
{
  _CCCL_ASSERT(h != nullptr && out_dims != nullptr, "invalid arguments");
  dim4 d = reinterpret_cast<exec_place*>(h)->get_dims();
  ::std::memcpy(out_dims, &d, sizeof(d));
}

size_t stf_exec_place_size(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return reinterpret_cast<exec_place*>(h)->size();
}

void stf_exec_place_set_affine_data_place(stf_exec_place_handle h, stf_data_place_handle affine_dplace)
{
  _CCCL_ASSERT(h != nullptr && affine_dplace != nullptr, "invalid arguments");
  reinterpret_cast<exec_place*>(h)->set_affine_data_place(deref_data_place(affine_dplace));
}

stf_exec_place_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count)
{
  _CCCL_ASSERT(device_ids != nullptr || count == 0, "device_ids must not be null unless count is 0");
  ::std::vector<exec_place> places;
  places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    places.push_back(exec_place::device(device_ids[i]));
  }
  return reinterpret_cast<stf_exec_place_handle>(new exec_place(make_grid(::std::move(places))));
}

stf_exec_place_handle
stf_exec_place_grid_create(const stf_exec_place_handle* places, size_t count, const stf_dim4* grid_dims)
{
  _CCCL_ASSERT(places != nullptr || count == 0, "places must not be null unless count is 0");
  ::std::vector<exec_place> cpp_places;
  cpp_places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    cpp_places.push_back(*reinterpret_cast<const exec_place*>(places[i]));
  }
  exec_place grid = (grid_dims != nullptr)
                    ? make_grid(::std::move(cpp_places), dim4(grid_dims->x, grid_dims->y, grid_dims->z, grid_dims->t))
                    : make_grid(::std::move(cpp_places));
  return reinterpret_cast<stf_exec_place_handle>(new exec_place(::std::move(grid)));
}

void stf_exec_place_grid_destroy(stf_exec_place_handle grid)
{
  stf_exec_place_destroy(grid);
}

stf_data_place_handle stf_data_place_host(void)
{
  return reinterpret_cast<stf_data_place_handle>(new data_place(data_place::host()));
}

stf_data_place_handle stf_data_place_device(int dev_id)
{
  return reinterpret_cast<stf_data_place_handle>(new data_place(data_place::device(dev_id)));
}

stf_data_place_handle stf_data_place_managed(void)
{
  return reinterpret_cast<stf_data_place_handle>(new data_place(data_place::managed()));
}

stf_data_place_handle stf_data_place_affine(void)
{
  return reinterpret_cast<stf_data_place_handle>(new data_place(data_place::affine()));
}

stf_data_place_handle stf_data_place_current_device(void)
{
  return reinterpret_cast<stf_data_place_handle>(new data_place(data_place::current_device()));
}

stf_data_place_handle stf_data_place_composite(stf_exec_place_handle grid, stf_get_executor_fn mapper)
{
  _CCCL_ASSERT(grid != nullptr, "exec place grid handle must not be null");
  _CCCL_ASSERT(mapper != nullptr, "partitioner function (mapper) must not be null");
  auto* grid_ptr            = reinterpret_cast<exec_place*>(grid);
  partition_fn_t cpp_mapper = reinterpret_cast<partition_fn_t>(mapper);
  auto* dp                  = new data_place(data_place::composite(cpp_mapper, *grid_ptr));
  return reinterpret_cast<stf_data_place_handle>(dp);
}

stf_data_place_handle stf_data_place_clone(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  const auto* dp = reinterpret_cast<const data_place*>(h);
  return reinterpret_cast<stf_data_place_handle>(new data_place(*dp));
}

void stf_data_place_destroy(stf_data_place_handle h)
{
  if (h == nullptr)
  {
    return;
  }
  delete reinterpret_cast<data_place*>(h);
}

int stf_data_place_get_device_ordinal(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  return device_ordinal(*reinterpret_cast<data_place*>(h));
}

const char* stf_data_place_to_string(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  static thread_local ::std::string s;
  s = reinterpret_cast<data_place*>(h)->to_string();
  return s.c_str();
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
  _CCCL_ASSERT(ctx != nullptr, "context handle pointer must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  auto ld_typed     = context_ptr->logical_data(make_slice((char*) addr, sz), data_place::host());
  *ld               = reinterpret_cast<stf_logical_data_handle>(new logical_data_untyped{ld_typed});
}

void stf_logical_data_with_place(
  stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz, stf_data_place_handle dplace)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle pointer must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle pointer must not be null");
  _CCCL_ASSERT(dplace != nullptr, "data_place handle must not be null");

  auto* context_ptr     = reinterpret_cast<context*>(ctx);
  data_place cpp_dplace = deref_data_place(dplace);
  auto ld_typed         = context_ptr->logical_data(make_slice((char*) addr, sz), cpp_dplace);
  *ld                   = reinterpret_cast<stf_logical_data_handle>(new logical_data_untyped{ld_typed});
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

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  *t                = reinterpret_cast<stf_task_handle>(new context::unified_task<>{context_ptr->task()});
}

void stf_task_set_exec_place(stf_task_handle t, stf_exec_place_handle exec_p)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(exec_p != nullptr, "exec_place handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  task_ptr->set_exec_place(deref_exec_place(exec_p));
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
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(data_p != nullptr, "data_place handle must not be null");

  auto* task_ptr = reinterpret_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = reinterpret_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m), deref_data_place(data_p)));
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

void stf_cuda_kernel_create(stf_ctx_handle ctx, stf_cuda_kernel_handle* k)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle output pointer must not be null");

  auto* context_ptr = reinterpret_cast<context*>(ctx);
  using kernel_type = decltype(context_ptr->cuda_kernel());
  *k                = reinterpret_cast<stf_cuda_kernel_handle>(new kernel_type{context_ptr->cuda_kernel()});
}

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place_handle exec_p)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(exec_p != nullptr, "exec_place handle must not be null");

  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  auto* kernel_ptr  = reinterpret_cast<kernel_type*>(k);
  kernel_ptr->set_exec_place(deref_exec_place(exec_p));
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

void stf_host_launch_set_symbol(stf_host_launch_handle h, const char* symbol)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  scope_ptr->set_symbol(symbol);
}

void stf_host_launch_add_dep(stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* scope_ptr = reinterpret_cast<host_launch_type*>(h);
  auto* ld_ptr    = reinterpret_cast<logical_data_untyped*>(ld);
  scope_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
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
  (*scope_ptr)->*[callback](cuda::experimental::stf::reserved::host_launch_deps& deps) {
    callback(reinterpret_cast<stf_host_launch_deps_handle>(&deps));
  };
}

void stf_host_launch_destroy(stf_host_launch_handle h)
{
  if (h == nullptr)
  {
    return;
  }

  delete reinterpret_cast<host_launch_type*>(h);
}

void* stf_host_launch_deps_get(stf_host_launch_deps_handle deps, size_t index)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<cuda::experimental::stf::reserved::host_launch_deps*>(deps);
  return d->get<slice<char>>(index).data_handle();
}

size_t stf_host_launch_deps_get_size(stf_host_launch_deps_handle deps, size_t index)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<cuda::experimental::stf::reserved::host_launch_deps*>(deps);
  return d->get<slice<char>>(index).extent(0);
}

size_t stf_host_launch_deps_size(stf_host_launch_deps_handle deps)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<cuda::experimental::stf::reserved::host_launch_deps*>(deps);
  return d->size();
}

void* stf_host_launch_deps_get_user_data(stf_host_launch_deps_handle deps)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = reinterpret_cast<cuda::experimental::stf::reserved::host_launch_deps*>(deps);
  return d->user_data();
}

} // extern "C"
