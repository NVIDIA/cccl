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
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#include <cccl/c/experimental/stf/stf.h>

using namespace cuda::experimental::stf;

namespace
{
static_assert(sizeof(pos4) == sizeof(stf_pos4), "pos4 and stf_pos4 must have identical layout for C/C++ interop");
static_assert(sizeof(dim4) == sizeof(stf_dim4), "dim4 and stf_dim4 must have identical layout for C/C++ interop");
static_assert(alignof(pos4) == alignof(stf_pos4), "pos4 and stf_pos4 must have identical alignment");
static_assert(alignof(dim4) == alignof(stf_dim4), "dim4 and stf_dim4 must have identical alignment");

template <class T, class = void>
struct is_complete : ::std::false_type
{};

template <class T>
struct is_complete<T, ::std::void_t<decltype(sizeof(T))>> : ::std::true_type
{};

template <class T>
inline constexpr bool is_complete_v = is_complete<T>::value;

// Wrap heap allocations that cross the extern "C" boundary. Parallel C APIs map failures to
// CUresult; this STF surface returns null handles (or leaves out-params unset) instead.
template <class F>
[[nodiscard]] auto stf_try_allocate(F&& f) noexcept -> decltype(f())
{
  try
  {
    return f();
  }
  catch (const ::std::exception& exc)
  {
    ::fflush(stderr);
    ::std::fprintf(stderr, "\nEXCEPTION in STF C API (allocation): %s\n", exc.what());
  }
  catch (...)
  {
    ::fflush(stderr);
    ::std::fprintf(stderr, "\nEXCEPTION in STF C API (allocation): non-standard exception\n");
  }
  ::fflush(stdout);
  return nullptr;
}

// Opaque <-> concrete pairings for this translation unit only (C++17, exhaustive if constexpr).
// Dependent false for static_assert in non-matching else (no std:: helper until later C++).
template <class>
inline constexpr bool stf_dependent_false_v = false;

// Heap object pointer -> matching C handle (no explicit handle / pointee template args).
template <class P>
[[nodiscard]] auto to_opaque(P* p) noexcept
{
  static_assert(!::std::is_const_v<P>, "to_opaque expects a non-const pointee pointer");
  void* const opaque_bits = static_cast<void*>(p);
  if constexpr (::std::is_same_v<P, exec_place>)
  {
    return static_cast<stf_exec_place_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, data_place>)
  {
    return static_cast<stf_data_place_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, context>)
  {
    return static_cast<stf_ctx_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, logical_data_untyped>)
  {
    return static_cast<stf_logical_data_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, context::unified_task<>>)
  {
    return static_cast<stf_task_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, context::cuda_kernel_builder>)
  {
    return static_cast<stf_cuda_kernel_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, context::host_launch_builder>)
  {
    return static_cast<stf_host_launch_handle>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<P, reserved::host_launch_deps>)
  {
    return static_cast<stf_host_launch_deps_handle>(opaque_bits);
  }
  else
  {
    static_assert(stf_dependent_false_v<P>, "to_opaque: missing pointee -> handle pairing");
  }
}

// C handle -> const concrete pointer; `from_opaque` adds a `const_cast` for mutable access.
template <class Opaque>
[[nodiscard]] auto* from_opaque_const(Opaque* h) noexcept
{
  static_assert(!is_complete_v<Opaque>);
  const void* const opaque_bits = static_cast<const void*>(h);

  if constexpr (::std::is_same_v<Opaque*, stf_exec_place_handle>)
  {
    return static_cast<const exec_place*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_data_place_handle>)
  {
    return static_cast<const data_place*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_ctx_handle>)
  {
    return static_cast<const context*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_logical_data_handle>)
  {
    return static_cast<const logical_data_untyped*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_task_handle>)
  {
    return static_cast<const context::unified_task<>*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_host_launch_deps_handle>)
  {
    return static_cast<const reserved::host_launch_deps*>(opaque_bits);
  }
  else
  {
    static_assert(stf_dependent_false_v<Opaque>, "from_opaque_const: missing handle -> pointee pairing");
  }
}

template <class Opaque>
[[nodiscard]] auto* from_opaque(Opaque* h) noexcept
{
  auto* const c = from_opaque_const(h);
  return const_cast<::std::remove_const_t<::std::remove_pointer_t<decltype(c)>>*>(c);
}
} // namespace

extern "C" {

stf_exec_place_handle stf_exec_place_host(void)
{
  return to_opaque(stf_try_allocate([] {
    return new exec_place(exec_place::host());
  }));
}

stf_exec_place_handle stf_exec_place_device(int dev_id)
{
  return to_opaque(stf_try_allocate([dev_id] {
    return new exec_place(exec_place::device(dev_id));
  }));
}

stf_exec_place_handle stf_exec_place_current_device(void)
{
  return to_opaque(stf_try_allocate([] {
    return new exec_place(exec_place::current_device());
  }));
}

stf_exec_place_handle stf_exec_place_clone(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  const auto* ep = from_opaque_const(h);
  return to_opaque(stf_try_allocate([ep] {
    return new exec_place(*ep);
  }));
}

void stf_exec_place_destroy(stf_exec_place_handle h)
{
  delete from_opaque(h);
}

int stf_exec_place_is_host(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return from_opaque(h)->is_host();
}

int stf_exec_place_is_device(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return from_opaque(h)->is_device();
}

void stf_exec_place_get_dims(stf_exec_place_handle h, stf_dim4* out_dims)
{
  _CCCL_ASSERT(h != nullptr && out_dims != nullptr, "invalid arguments");
  dim4 d = from_opaque(h)->get_dims();
  static_assert(sizeof(d) == sizeof(stf_dim4), "dim4 and stf_dim4 must have the same size");
  ::std::memcpy(out_dims, &d, sizeof(d));
}

size_t stf_exec_place_size(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return from_opaque(h)->size();
}

void stf_exec_place_set_affine_data_place(stf_exec_place_handle h, stf_data_place_handle affine_dplace)
{
  _CCCL_ASSERT(h != nullptr && affine_dplace != nullptr, "invalid arguments");
  from_opaque(h)->set_affine_data_place(*from_opaque(affine_dplace));
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
  return to_opaque(stf_try_allocate([&places] {
    return new exec_place(make_grid(::std::move(places)));
  }));
}

stf_exec_place_handle
stf_exec_place_grid_create(const stf_exec_place_handle* places, size_t count, const stf_dim4* grid_dims)
{
  _CCCL_ASSERT(places != nullptr || count == 0, "places must not be null unless count is 0");
  ::std::vector<exec_place> cpp_places;
  cpp_places.reserve(count);
  for (size_t i = 0; i < count; i++)
  {
    cpp_places.push_back(*from_opaque_const(places[i]));
  }
  exec_place grid = (grid_dims != nullptr)
                    ? make_grid(::std::move(cpp_places), dim4(grid_dims->x, grid_dims->y, grid_dims->z, grid_dims->t))
                    : make_grid(::std::move(cpp_places));
  return to_opaque(stf_try_allocate([g = ::std::move(grid)]() mutable {
    return new exec_place(::std::move(g));
  }));
}

void stf_exec_place_grid_destroy(stf_exec_place_handle grid)
{
  stf_exec_place_destroy(grid);
}

stf_data_place_handle stf_data_place_host(void)
{
  return to_opaque(stf_try_allocate([] {
    return new data_place(data_place::host());
  }));
}

stf_data_place_handle stf_data_place_device(int dev_id)
{
  return to_opaque(stf_try_allocate([dev_id] {
    return new data_place(data_place::device(dev_id));
  }));
}

stf_data_place_handle stf_data_place_managed(void)
{
  return to_opaque(stf_try_allocate([] {
    return new data_place(data_place::managed());
  }));
}

stf_data_place_handle stf_data_place_affine(void)
{
  return to_opaque(stf_try_allocate([] {
    return new data_place(data_place::affine());
  }));
}

stf_data_place_handle stf_data_place_current_device(void)
{
  return to_opaque(stf_try_allocate([] {
    return new data_place(data_place::current_device());
  }));
}

stf_data_place_handle stf_data_place_composite(stf_exec_place_handle grid, stf_get_executor_fn mapper)
{
  _CCCL_ASSERT(grid != nullptr, "exec place grid handle must not be null");
  _CCCL_ASSERT(mapper != nullptr, "partitioner function (mapper) must not be null");
  auto* grid_ptr = from_opaque(grid);
  // Distinct function pointer types (C typedef vs C++ alias); not convertible via static_cast under nvcc.
  partition_fn_t cpp_mapper = reinterpret_cast<partition_fn_t>(mapper);
  auto* dp                  = stf_try_allocate([cpp_mapper, grid_ptr] {
    return new data_place(data_place::composite(cpp_mapper, *grid_ptr));
  });
  return to_opaque(dp);
}

stf_data_place_handle stf_data_place_clone(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  const auto* dp = from_opaque_const(h);
  return to_opaque(stf_try_allocate([dp] {
    return new data_place(*dp);
  }));
}

void stf_data_place_destroy(stf_data_place_handle h)
{
  delete from_opaque(h);
}

int stf_data_place_get_device_ordinal(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  return device_ordinal(*from_opaque(h));
}

const char* stf_data_place_to_string(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  static thread_local ::std::string s;
  s = from_opaque(h)->to_string();
  return s.c_str();
}

stf_ctx_handle stf_ctx_create(void)
{
  return to_opaque(stf_try_allocate([] {
    return new context{};
  }));
}

stf_ctx_handle stf_ctx_create_graph(void)
{
  return to_opaque(stf_try_allocate([] {
    return new context{graph_ctx()};
  }));
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = from_opaque(ctx);
  context_ptr->finalize();
  delete context_ptr;
}

cudaStream_t stf_fence(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = from_opaque(ctx);
  return context_ptr->fence();
}

stf_logical_data_handle stf_logical_data(stf_ctx_handle ctx, void* addr, size_t sz)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  auto ld_typed     = context_ptr->logical_data(make_slice((char*) addr, sz), data_place::host());
  return to_opaque(stf_try_allocate([&ld_typed] {
    return new logical_data_untyped{::std::move(ld_typed)};
  }));
}

stf_logical_data_handle
stf_logical_data_with_place(stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  _CCCL_ASSERT(dplace != nullptr, "data_place handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  auto ld_typed     = context_ptr->logical_data(make_slice((char*) addr, sz), *from_opaque(dplace));
  return to_opaque(stf_try_allocate([&ld_typed] {
    return new logical_data_untyped{::std::move(ld_typed)};
  }));
}

void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* ld_ptr = from_opaque(ld);
  ld_ptr->set_symbol(symbol);
}

void stf_logical_data_destroy(stf_logical_data_handle ld)
{
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* ld_ptr = from_opaque(ld);
  delete ld_ptr;
}

stf_logical_data_handle stf_logical_data_empty(stf_ctx_handle ctx, size_t length)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  auto ld_typed     = context_ptr->logical_data(shape_of<slice<char>>(length));
  return to_opaque(stf_try_allocate([&ld_typed] {
    return new logical_data_untyped{::std::move(ld_typed)};
  }));
}

stf_logical_data_handle stf_token(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  return to_opaque(stf_try_allocate([&] {
    return new logical_data_untyped{context_ptr->token()};
  }));
}

stf_task_handle stf_task_create(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  return to_opaque(stf_try_allocate([&] {
    return new context::unified_task<>{context_ptr->task()};
  }));
}

void stf_task_set_exec_place(stf_task_handle t, stf_exec_place_handle exec_p)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(exec_p != nullptr, "exec_place handle must not be null");

  auto* task_ptr = from_opaque(t);
  task_ptr->set_exec_place(*from_opaque(exec_p));
}

void stf_task_set_symbol(stf_task_handle t, const char* symbol)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* task_ptr = from_opaque(t);
  task_ptr->set_symbol(symbol);
}

void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* task_ptr = from_opaque(t);
  auto* ld_ptr   = from_opaque(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");
  _CCCL_ASSERT(data_p != nullptr, "data_place handle must not be null");

  auto* task_ptr = from_opaque(t);
  auto* ld_ptr   = from_opaque(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m), *from_opaque(data_p)));
}

void* stf_task_get(stf_task_handle t, int index)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = from_opaque(t);
  auto s         = task_ptr->template get<slice<const char>>(index);
  return (void*) s.data_handle();
}

void stf_task_start(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = from_opaque(t);
  task_ptr->start();
}

void stf_task_end(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = from_opaque(t);
  task_ptr->end();
}

void stf_task_enable_capture(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = from_opaque(t);
  task_ptr->enable_capture();
}

CUstream stf_task_get_custream(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = from_opaque(t);
  return static_cast<CUstream>(task_ptr->get_stream());
}

void stf_task_destroy(stf_task_handle t)
{
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");

  auto* task_ptr = from_opaque(t);
  delete task_ptr;
}

stf_cuda_kernel_handle stf_cuda_kernel_create(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  return to_opaque(stf_try_allocate([&] {
    return new context::cuda_kernel_builder{context_ptr->cuda_kernel()};
  }));
}

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place_handle exec_p)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(exec_p != nullptr, "exec_place handle must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));
  kernel_ptr->set_exec_place(*from_opaque(exec_p));
}

void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));
  kernel_ptr->set_symbol(symbol);
}

void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));
  auto* ld_ptr     = from_opaque(ld);
  kernel_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));
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

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));

  cuda_kernel_desc desc;
  desc.configure_raw(cufunc, grid_dim_, block_dim_, shared_mem_, arg_cnt, args);
  kernel_ptr->add_kernel_desc(mv(desc));
}

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));
  auto s           = kernel_ptr->template get<slice<const char>>(index);
  return (void*) (s.data_handle());
}

void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
{
  _CCCL_ASSERT(k != nullptr, "cuda kernel handle must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(k));
  kernel_ptr->end();
}

void stf_cuda_kernel_destroy(stf_cuda_kernel_handle t)
{
  _CCCL_ASSERT(t != nullptr, "cuda kernel handle must not be null");

  auto* kernel_ptr = static_cast<context::cuda_kernel_builder*>(static_cast<void*>(t));
  delete kernel_ptr;
}

// -----------------------------------------------------------------------------
// Host launch
// -----------------------------------------------------------------------------

stf_host_launch_handle stf_host_launch_create(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");

  auto* context_ptr = from_opaque(ctx);
  return to_opaque(stf_try_allocate([&] {
    return new context::host_launch_builder{context_ptr->host_launch()};
  }));
}

void stf_host_launch_set_symbol(stf_host_launch_handle h, const char* symbol)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol string must not be null");

  auto* scope_ptr = static_cast<context::host_launch_builder*>(static_cast<void*>(h));
  scope_ptr->set_symbol(symbol);
}

void stf_host_launch_add_dep(stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "logical data handle must not be null");

  auto* scope_ptr = static_cast<context::host_launch_builder*>(static_cast<void*>(h));
  auto* ld_ptr    = from_opaque(ld);
  scope_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_host_launch_set_user_data(stf_host_launch_handle h, const void* data, size_t size, void (*dtor)(void*))
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");

  auto* scope_ptr = static_cast<context::host_launch_builder*>(static_cast<void*>(h));
  scope_ptr->set_user_data(data, size, dtor);
}

void stf_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(callback != nullptr, "callback must not be null");

  auto* scope_ptr = static_cast<context::host_launch_builder*>(static_cast<void*>(h));
  (*scope_ptr)->*[callback](cuda::experimental::stf::reserved::host_launch_deps& deps) {
    callback(to_opaque(&deps));
  };
}

void stf_host_launch_destroy(stf_host_launch_handle h)
{
  if (h == nullptr)
  {
    return;
  }

  delete static_cast<context::host_launch_builder*>(static_cast<void*>(h));
}

void* stf_host_launch_deps_get(stf_host_launch_deps_handle deps, size_t index)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = from_opaque(deps);
  return d->get<slice<char>>(index).data_handle();
}

size_t stf_host_launch_deps_get_size(stf_host_launch_deps_handle deps, size_t index)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = from_opaque(deps);
  return d->get<slice<char>>(index).extent(0);
}

size_t stf_host_launch_deps_size(stf_host_launch_deps_handle deps)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = from_opaque(deps);
  return d->size();
}

void* stf_host_launch_deps_get_user_data(stf_host_launch_deps_handle deps)
{
  _CCCL_ASSERT(deps != nullptr, "deps handle must not be null");

  auto* d = from_opaque(deps);
  return d->user_data();
}

} // extern "C"
