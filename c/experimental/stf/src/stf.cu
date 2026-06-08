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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cccl/c/experimental/stf/stf.h>

using namespace cuda::experimental::stf;

struct stf_exec_place_resources_opaque_t
{
  exec_place_resources* resources;
  bool owns_resources;
  bool owns_handle;
};

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
  else if constexpr (::std::is_same_v<P, exec_place_resources>)
  {
    static_assert(stf_dependent_false_v<P>, "use to_place_resources_opaque for exec_place_resources handles");
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
  else if constexpr (::std::is_same_v<P, exec_place_scope>)
  {
    return static_cast<stf_exec_place_scope_handle>(opaque_bits);
  }
#if _CCCL_CTK_AT_LEAST(12, 4)
  else if constexpr (::std::is_same_v<P, green_context_helper>)
  {
    return static_cast<stf_green_context_helper_handle>(opaque_bits);
  }
#endif
  else
  {
    static_assert(stf_dependent_false_v<P>, "to_opaque: missing pointee -> handle pairing");
  }
}

// C handle -> const concrete pointer; `from_opaque` adds a `const_cast` for mutable access.
template <class Opaque>
[[nodiscard]] auto* from_opaque_const(Opaque* h) noexcept
{
  static_assert(!is_complete_v<Opaque> || ::std::is_same_v<Opaque*, stf_exec_place_resources_handle>);
  const void* const opaque_bits = static_cast<const void*>(h);

  if constexpr (::std::is_same_v<Opaque*, stf_exec_place_handle>)
  {
    return static_cast<const exec_place*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_data_place_handle>)
  {
    return static_cast<const data_place*>(opaque_bits);
  }
  else if constexpr (::std::is_same_v<Opaque*, stf_exec_place_resources_handle>)
  {
    return static_cast<const stf_exec_place_resources_opaque_t*>(opaque_bits)->resources;
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
  else if constexpr (::std::is_same_v<Opaque*, stf_exec_place_scope_handle>)
  {
    return static_cast<const exec_place_scope*>(opaque_bits);
  }
#if _CCCL_CTK_AT_LEAST(12, 4)
  else if constexpr (::std::is_same_v<Opaque*, stf_green_context_helper_handle>)
  {
    return static_cast<const green_context_helper*>(opaque_bits);
  }
#endif
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

stf_green_context_helper_handle stf_green_context_helper_create(int sm_count, int dev_id)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  return to_opaque(stf_try_allocate([sm_count, dev_id] {
    return new green_context_helper(sm_count, dev_id);
  }));
#else
  (void) sm_count;
  (void) dev_id;
  return nullptr;
#endif
}

void stf_green_context_helper_destroy(stf_green_context_helper_handle h)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  delete from_opaque(h);
#else
  (void) h;
#endif
}

size_t stf_green_context_helper_get_count(stf_green_context_helper_handle h)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  _CCCL_ASSERT(h != nullptr, "green_context_helper handle must not be null");
  return from_opaque(h)->get_count();
#else
  (void) h;
  return 0;
#endif
}

int stf_green_context_helper_get_device_id(stf_green_context_helper_handle h)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  _CCCL_ASSERT(h != nullptr, "green_context_helper handle must not be null");
  return static_cast<int>(from_opaque(h)->get_device_id());
#else
  (void) h;
  return -1;
#endif
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

stf_exec_place_scope_handle stf_exec_place_scope_enter(stf_exec_place_handle place, size_t idx)
{
  _CCCL_ASSERT(place != nullptr, "exec_place handle must not be null");
  if (idx >= from_opaque(place)->size())
  {
    return nullptr;
  }
  return to_opaque(stf_try_allocate([&] {
    return new exec_place_scope(*from_opaque(place), idx);
  }));
}

void stf_exec_place_scope_exit(stf_exec_place_scope_handle scope)
{
  delete from_opaque(scope);
}

stf_data_place_handle stf_exec_place_get_affine_data_place(stf_exec_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return to_opaque(stf_try_allocate([h] {
    return new data_place(from_opaque(h)->affine_data_place());
  }));
}

stf_exec_place_resources_handle stf_exec_place_resources_create(void)
{
  return stf_try_allocate([] {
    auto* res = new exec_place_resources{};
    try
    {
      return new stf_exec_place_resources_opaque_t{res, true, true};
    }
    catch (...)
    {
      delete res;
      throw;
    }
  });
}

void stf_exec_place_resources_destroy(stf_exec_place_resources_handle h)
{
  if (h == nullptr)
  {
    return;
  }
  if (h->owns_resources)
  {
    delete h->resources;
  }
  if (h->owns_handle)
  {
    delete h;
  }
}

CUstream stf_exec_place_pick_stream(stf_exec_place_resources_handle res, stf_exec_place_handle h, int for_computation)
{
  _CCCL_ASSERT(res != nullptr, "exec_place_resources handle must not be null");
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  return reinterpret_cast<CUstream>(from_opaque(h)->pick_stream(*from_opaque(res), for_computation != 0));
}

stf_exec_place_handle stf_exec_place_get_place(stf_exec_place_handle h, size_t idx)
{
  _CCCL_ASSERT(h != nullptr, "exec_place handle must not be null");
  if (idx >= from_opaque(h)->size())
  {
    return nullptr;
  }
  return to_opaque(stf_try_allocate([h, idx] {
    return new exec_place(from_opaque(h)->get_place(idx));
  }));
}

stf_exec_place_handle
stf_exec_place_green_ctx(stf_green_context_helper_handle helper, size_t idx, int use_green_ctx_data_place)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  _CCCL_ASSERT(helper != nullptr, "green_context_helper handle must not be null");
  auto* gc_helper = from_opaque(helper);
  if (idx >= gc_helper->get_count())
  {
    return nullptr;
  }
  return to_opaque(stf_try_allocate([gc_helper, idx, use_green_ctx_data_place] {
    return new exec_place(exec_place::green_ctx(gc_helper->get_view(idx), use_green_ctx_data_place != 0));
  }));
#else
  (void) helper;
  (void) idx;
  (void) use_green_ctx_data_place;
  return nullptr;
#endif
}

void stf_machine_init(void)
{
  // machine::instance() does real work on first call (P2P/mempool/topology
  // setup) and can throw. Guard the extern "C" boundary so a C++ exception
  // never unwinds into a C caller (which would be UB / std::terminate).
  try
  {
    cuda::experimental::places::reserved::machine::instance();
  }
  catch (const ::std::exception& exc)
  {
    ::fflush(stdout);
    ::std::fprintf(stderr, "\nEXCEPTION in STF C API (machine init): %s\n", exc.what());
    ::fflush(stderr);
  }
  catch (...)
  {
    ::fflush(stdout);
    ::std::fprintf(stderr, "\nEXCEPTION in STF C API (machine init): non-standard exception\n");
    ::fflush(stderr);
  }
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
  // Distinct function pointer types (C typedef vs C++ alias) are not
  // convertible via static_cast under nvcc.
  const auto cpp_mapper = reinterpret_cast<partition_fn_t>(mapper);
  auto* dp              = stf_try_allocate([cpp_mapper, grid_ptr] {
    return new data_place(data_place::composite(cpp_mapper, *grid_ptr));
  });
  return to_opaque(dp);
}

stf_data_place_handle stf_data_place_green_ctx(stf_green_context_helper_handle helper, size_t idx)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  _CCCL_ASSERT(helper != nullptr, "green_context_helper handle must not be null");
  auto* gc_helper = from_opaque(helper);
  if (idx >= gc_helper->get_count())
  {
    return nullptr;
  }
  return to_opaque(stf_try_allocate([gc_helper, idx] {
    return new data_place(data_place::green_ctx(gc_helper->get_view(idx)));
  }));
#else
  (void) helper;
  (void) idx;
  return nullptr;
#endif
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

void* stf_data_place_allocate(stf_data_place_handle h, ptrdiff_t size, cudaStream_t stream)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  try
  {
    return from_opaque(h)->allocate(static_cast<::std::ptrdiff_t>(size), stream);
  }
  catch (const ::std::exception& e)
  {
    fprintf(stderr, "stf_data_place_allocate failed: %s\n", e.what());
    return nullptr;
  }
  catch (...)
  {
    fprintf(stderr, "stf_data_place_allocate failed: unknown exception\n");
    return nullptr;
  }
}

void stf_data_place_deallocate(stf_data_place_handle h, void* ptr, size_t size, cudaStream_t stream)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  try
  {
    from_opaque(h)->deallocate(ptr, size, stream);
  }
  catch (const ::std::exception& e)
  {
    fprintf(stderr, "stf_data_place_deallocate failed: %s\n", e.what());
  }
  catch (...)
  {
    fprintf(stderr, "stf_data_place_deallocate failed: unknown exception\n");
  }
}

int stf_data_place_allocation_is_stream_ordered(stf_data_place_handle h)
{
  _CCCL_ASSERT(h != nullptr, "data_place handle must not be null");
  return from_opaque(h)->allocation_is_stream_ordered() ? 1 : 0;
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

// Opaque bridge types for the extern-C `async_resources_handle` wrapper.
// `async_resources_handle` is defined in cudax and not listed in the generic
// `to_opaque/from_opaque` registry above, so we just reinterpret the pointer
// directly here.
namespace
{
inline stf_async_resources_handle async_resources_to_opaque(async_resources_handle* p) noexcept
{
  return reinterpret_cast<stf_async_resources_handle>(p);
}

inline async_resources_handle* async_resources_from_opaque(stf_async_resources_handle h) noexcept
{
  return reinterpret_cast<async_resources_handle*>(h);
}
} // namespace

stf_async_resources_handle stf_async_resources_create(void)
{
  return async_resources_to_opaque(stf_try_allocate([] {
    return new async_resources_handle{};
  }));
}

void stf_async_resources_destroy(stf_async_resources_handle h)
{
  delete async_resources_from_opaque(h);
}

stf_ctx_handle stf_ctx_create_ex(const stf_ctx_options* opts)
{
  // NULL opts matches stf_ctx_create().
  const stf_ctx_options defaults{};
  const stf_ctx_options& o = opts ? *opts : defaults;

  const bool has_stream           = (o.has_stream != 0);
  const async_resources_handle ah = o.handle ? *async_resources_from_opaque(o.handle) : async_resources_handle{nullptr};

  // C++ overloads distinguish "caller supplied a stream" from "use the
  // default constructor". `cudaStream_t` is pointer-like, so a separate flag is
  // required to let callers intentionally bind the CUDA default stream.
  return to_opaque(stf_try_allocate([&]() -> context* {
    switch (o.backend)
    {
      case STF_BACKEND_GRAPH:
        if (has_stream)
        {
          return new context{graph_ctx(o.stream, ah)};
        }
        if (o.handle != nullptr)
        {
          return new context{graph_ctx(ah)};
        }
        return new context{graph_ctx()};
      case STF_BACKEND_STREAM:
      default:
        if (has_stream)
        {
          return new context{stream_ctx(o.stream, ah)};
        }
        if (o.handle != nullptr)
        {
          return new context{stream_ctx(ah)};
        }
        return new context{};
    }
  }));
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = from_opaque(ctx);
  context_ptr->finalize();
  delete context_ptr;
}

stf_exec_place_resources_handle stf_ctx_get_place_resources(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = from_opaque(ctx);
  return stf_try_allocate([context_ptr] {
    return new stf_exec_place_resources_opaque_t{&context_ptr->async_resources().get_place_resources(), false, true};
  });
}

cudaStream_t stf_fence(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "context handle must not be null");
  auto* context_ptr = from_opaque(ctx);
  return context_ptr->fence();
}

int stf_ctx_wait(stf_ctx_handle ctx, stf_logical_data_handle ld, void* out, size_t size)
{
  if (ctx == nullptr || ld == nullptr || out == nullptr)
  {
    return 1;
  }

  try
  {
    auto* context_ptr = from_opaque(ctx);
    auto* ld_ptr      = from_opaque(ld);

    void* dst  = out;
    size_t cap = size;

    auto builder = context_ptr->host_launch();
    builder.add_deps(task_dep_untyped(*ld_ptr, access_mode::read));
    builder.set_symbol("wait");
    builder->*[dst, cap](reserved::host_launch_deps& deps) {
      auto data      = deps.get<slice<char>>(0);
      size_t copy_sz = ::std::min(cap, static_cast<size_t>(data.extent(0)));
      // The destination must not overlap the logical data range: in practice the
      // logical data is backed by storage that is allocated independently from the
      // caller's readback buffer, so use uintptr_t comparisons (relational pointer
      // comparison across unrelated allocations is unspecified) to encode that
      // contract.
      const auto src_begin = reinterpret_cast<::std::uintptr_t>(data.data_handle());
      const auto src_end   = src_begin + copy_sz;
      const auto dst_begin = reinterpret_cast<::std::uintptr_t>(dst);
      const auto dst_end   = dst_begin + copy_sz;
      _CCCL_ASSERT(copy_sz == 0 || dst_end <= src_begin || src_end <= dst_begin,
                   "stf_ctx_wait destination buffer must not overlap the logical data range");
      ::std::memcpy(dst, data.data_handle(), copy_sz);
    };

    cudaStream_t fence_stream = context_ptr->fence();
    cuda_safe_call(cudaStreamSynchronize(fence_stream));
    return 0;
  }
  catch (...)
  {
    return 1;
  }
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

int stf_task_get_grid_dims(stf_task_handle t, stf_dim4* out_dims)
{
  if (t == nullptr || out_dims == nullptr)
  {
    return -1;
  }
  auto* task_ptr = from_opaque(t);
  dim4 d;
  if (!task_ptr->get_grid_dims(&d))
  {
    return -1;
  }
  out_dims->x = static_cast<uint64_t>(d.x);
  out_dims->y = static_cast<uint64_t>(d.y);
  out_dims->z = static_cast<uint64_t>(d.z);
  out_dims->t = static_cast<uint64_t>(d.t);
  return 0;
}

int stf_task_get_custream_at_index(stf_task_handle t, size_t place_index, CUstream* out_stream)
{
  if (t == nullptr || out_stream == nullptr)
  {
    return -1;
  }
  auto* task_ptr = from_opaque(t);
  cudaStream_t s = task_ptr->get_stream(place_index);
  if (s == nullptr)
  {
    return -1;
  }
  *out_stream = static_cast<CUstream>(s);
  return 0;
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

// ============================================================================
// Stackable Context API
// ============================================================================
//
// The stackable C API mirrors the modern opaque-handle convention used by the
// regular STF C API, but reuses a few existing handle types (stf_ctx_handle,
// stf_logical_data_handle, stf_task_handle, stf_host_launch_handle) so that
// non-stackable accessors (stf_task_start, stf_logical_data_set_symbol via the
// underlying context, ...) keep working transparently.  Internally the
// pointee types differ from the regular STF objects (stackable_ctx vs context,
// stackable_logical_data<slice<char>> vs logical_data_untyped, ...), so
// stackable handles must be created and destroyed through the matching
// stf_stackable_* entry points only.

namespace
{
using stackable_ld_t    = stackable_logical_data<slice<char>>;
using stackable_token_t = stackable_logical_data<void_interface>;

// Convert the new stf_while_scope_handle / stf_repeat_scope_handle opaque
// types to / from their concrete C++ counterparts.  Kept local to the
// stackable section so the main to_opaque/from_opaque dispatchers stay focused
// on the regular API.
[[nodiscard]] auto to_opaque_while(stackable_ctx::while_graph_scope_guard* p) noexcept
{
  return static_cast<stf_while_scope_handle>(static_cast<void*>(p));
}

[[nodiscard]] auto* from_opaque_while(stf_while_scope_handle h) noexcept
{
  return static_cast<stackable_ctx::while_graph_scope_guard*>(static_cast<void*>(h));
}

[[nodiscard]] auto to_opaque_repeat(repeat_graph_scope_guard* p) noexcept
{
  return static_cast<stf_repeat_scope_handle>(static_cast<void*>(p));
}

[[nodiscard]] auto* from_opaque_repeat(stf_repeat_scope_handle h) noexcept
{
  return static_cast<repeat_graph_scope_guard*>(static_cast<void*>(h));
}

[[nodiscard]] auto to_opaque_launchable(launchable_graph_handle* p) noexcept
{
  return static_cast<stf_launchable_graph_handle>(static_cast<void*>(p));
}

[[nodiscard]] auto* from_opaque_launchable(stf_launchable_graph_handle h) noexcept
{
  return static_cast<launchable_graph_handle*>(static_cast<void*>(h));
}

// Each C shared opaque is one heap-allocated C++ `launchable_graph` by value,
// which itself holds one std::shared_ptr to the shared state. Duplicating the
// C handle therefore amounts to allocating a new launchable_graph that
// copy-constructs from the original (bumping the shared_ptr refcount);
// freeing destroys that one launchable_graph which releases its reference.
[[nodiscard]] auto to_opaque_launchable_shared(stackable_ctx::launchable_graph* p) noexcept
{
  return static_cast<stf_launchable_graph_shared>(static_cast<void*>(p));
}

[[nodiscard]] auto* from_opaque_launchable_shared(stf_launchable_graph_shared h) noexcept
{
  return static_cast<stackable_ctx::launchable_graph*>(static_cast<void*>(h));
}

// Stackable handles are typedef-aliased to existing handle types, so the
// generic to_opaque/from_opaque dispatchers cannot disambiguate.  Use these
// thin local helpers instead.
[[nodiscard]] stf_ctx_handle to_opaque_sctx(stackable_ctx* p) noexcept
{
  return static_cast<stf_ctx_handle>(static_cast<void*>(p));
}

[[nodiscard]] stackable_ctx* from_opaque_sctx(stf_ctx_handle h) noexcept
{
  return static_cast<stackable_ctx*>(static_cast<void*>(h));
}

// The C-facade stores stackable logical data behind an opaque handle.  Two
// concrete pointee types exist: stackable_ld_t (for byte-buffer data created
// by stf_stackable_logical_data*()) and stackable_token_t (for tokens created
// by stf_stackable_token()).  They have distinct C++ types (different
// stackable_logical_data<T> instantiations carrying different frozen_ld<T>
// machinery across nested scopes), so we cannot collapse them at the opaque
// boundary.  Instead, every stf_logical_data_handle coming from the stackable
// API points at a tiny wrapper that records which kind of pointee it holds,
// and every entry point dispatches through visit_sld() so the right concrete
// type is used.
struct stackable_ld_opaque
{
  bool is_token;
  void* impl; // stackable_ld_t* if !is_token, stackable_token_t* otherwise
};

[[nodiscard]] stf_logical_data_handle to_opaque_sld(stackable_ld_opaque* w) noexcept
{
  return static_cast<stf_logical_data_handle>(static_cast<void*>(w));
}

[[nodiscard]] stackable_ld_opaque* from_opaque_sld_wrapper(stf_logical_data_handle h) noexcept
{
  return static_cast<stackable_ld_opaque*>(static_cast<void*>(h));
}

// Dispatch on the wrapper kind and forward the concrete stackable_logical_data<T>
// reference to `f`.  Both instantiations expose the same member surface
// (validate_access, get_ld, push, set_symbol, set_read_only, ...), so `f` is
// a generic lambda accepting `auto&`.
template <class F>
decltype(auto) visit_sld(stf_logical_data_handle h, F&& f)
{
  auto* w = from_opaque_sld_wrapper(h);
  return w->is_token ? f(*static_cast<stackable_token_t*>(w->impl)) : f(*static_cast<stackable_ld_t*>(w->impl));
}

#if _CCCL_CTK_AT_LEAST(12, 4)
// Built-in condition kernel for while_cond_scalar.  Reads the head of the
// scalar logical data, applies the requested comparison and updates the
// conditional handle in place.  Lives outside extern "C" because it is a
// device kernel template.
template <typename T>
__global__ void
stf_stackable_while_cond_kernel(const T* value, cudaGraphConditionalHandle handle, double threshold, int op)
{
  const double v = static_cast<double>(*value);
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
#endif // _CCCL_CTK_AT_LEAST(12, 4)
} // namespace

extern "C" {

stf_ctx_handle stf_stackable_ctx_create(void)
{
  return to_opaque_sctx(stf_try_allocate([] {
    return new stackable_ctx{};
  }));
}

void stf_stackable_ctx_finalize(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  auto* sctx = from_opaque_sctx(ctx);
  sctx->finalize();
  delete sctx;
}

cudaStream_t stf_stackable_ctx_fence(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  return from_opaque_sctx(ctx)->fence();
}

void stf_stackable_push_graph(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  from_opaque_sctx(ctx)->push();
}

void stf_stackable_pop(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  from_opaque_sctx(ctx)->pop();
}

stf_launchable_graph_handle stf_stackable_pop_prologue(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  auto* sctx = from_opaque_sctx(ctx);
  return to_opaque_launchable(stf_try_allocate([sctx] {
    return new launchable_graph_handle(sctx->pop_prologue());
  }));
}

void stf_stackable_pop_epilogue(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  from_opaque_sctx(ctx)->pop_epilogue();
}

void stf_launchable_graph_launch(stf_launchable_graph_handle h)
{
  _CCCL_ASSERT(h != nullptr, "launchable graph handle must not be null");
  from_opaque_launchable(h)->launch();
}

cudaGraphExec_t stf_launchable_graph_exec(stf_launchable_graph_handle h)
{
  _CCCL_ASSERT(h != nullptr, "launchable graph handle must not be null");
  return from_opaque_launchable(h)->exec();
}

cudaStream_t stf_launchable_graph_stream(stf_launchable_graph_handle h)
{
  _CCCL_ASSERT(h != nullptr, "launchable graph handle must not be null");
  return from_opaque_launchable(h)->stream();
}

cudaGraph_t stf_launchable_graph_graph(stf_launchable_graph_handle h)
{
  _CCCL_ASSERT(h != nullptr, "launchable graph handle must not be null");
  return from_opaque_launchable(h)->graph();
}

void stf_launchable_graph_destroy(stf_launchable_graph_handle h)
{
  // NULL is a no-op, matching the pattern used by other destroy entry points.
  delete from_opaque_launchable(h);
}

int stf_stackable_pop_prologue_shared(stf_ctx_handle ctx, stf_launchable_graph_shared* out)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  _CCCL_ASSERT(out != nullptr, "output pointer must not be null");
  auto* sctx = from_opaque_sctx(ctx);
  auto* p    = stf_try_allocate([sctx] {
    return new stackable_ctx::launchable_graph(sctx->pop_prologue_shared());
  });
  if (p == nullptr)
  {
    *out = nullptr;
    return 1;
  }
  *out = to_opaque_launchable_shared(p);
  return 0;
}

int stf_launchable_graph_shared_dup(stf_launchable_graph_shared h, stf_launchable_graph_shared* out)
{
  _CCCL_ASSERT(h != nullptr, "shared launchable graph handle must not be null");
  _CCCL_ASSERT(out != nullptr, "output pointer must not be null");
  auto* src = from_opaque_launchable_shared(h);
  auto* p   = stf_try_allocate([src] {
    return new stackable_ctx::launchable_graph(*src); // shared_ptr copy -> bumps refcount
  });
  if (p == nullptr)
  {
    *out = nullptr;
    return 1;
  }
  *out = to_opaque_launchable_shared(p);
  return 0;
}

void stf_launchable_graph_shared_free(stf_launchable_graph_shared h)
{
  // NULL is a no-op, matching the pattern used by other destroy entry points.
  // Destruction drops the shared_ptr held inside the launchable_graph; when
  // the last C-side handle is freed the state destructor runs and triggers
  // ctx.pop_epilogue() automatically.
  delete from_opaque_launchable_shared(h);
}

int stf_launchable_graph_shared_valid(stf_launchable_graph_shared h)
{
  if (h == nullptr)
  {
    return 0;
  }
  return from_opaque_launchable_shared(h)->valid() ? 1 : 0;
}

void stf_launchable_graph_shared_launch(stf_launchable_graph_shared h)
{
  _CCCL_ASSERT(h != nullptr, "shared launchable graph handle must not be null");
  from_opaque_launchable_shared(h)->launch();
}

cudaGraphExec_t stf_launchable_graph_shared_exec(stf_launchable_graph_shared h)
{
  _CCCL_ASSERT(h != nullptr, "shared launchable graph handle must not be null");
  return from_opaque_launchable_shared(h)->exec();
}

cudaStream_t stf_launchable_graph_shared_stream(stf_launchable_graph_shared h)
{
  _CCCL_ASSERT(h != nullptr, "shared launchable graph handle must not be null");
  return from_opaque_launchable_shared(h)->stream();
}

cudaGraph_t stf_launchable_graph_shared_graph(stf_launchable_graph_shared h)
{
  _CCCL_ASSERT(h != nullptr, "shared launchable graph handle must not be null");
  return from_opaque_launchable_shared(h)->graph();
}

#if _CCCL_CTK_AT_LEAST(12, 4)

stf_while_scope_handle stf_stackable_push_while(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  auto* sctx = from_opaque_sctx(ctx);
  // default_launch_value=1 so the loop body executes at least once (matches the C++ factory).
  return to_opaque_while(stf_try_allocate([sctx] {
    return new stackable_ctx::while_graph_scope_guard(*sctx, /*default_launch_value=*/1);
  }));
}

void stf_stackable_pop_while(stf_while_scope_handle scope)
{
  delete from_opaque_while(scope);
}

uint64_t stf_while_scope_get_cond_handle(stf_while_scope_handle scope)
{
  _CCCL_ASSERT(scope != nullptr, "while scope handle must not be null");
  return static_cast<uint64_t>(from_opaque_while(scope)->cond_handle());
}

stf_repeat_scope_handle stf_stackable_push_repeat(stf_ctx_handle ctx, size_t count)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  // The repeat counter is unsigned and decremented on every iteration, so a
  // count of 0 would underflow and produce a huge / non-terminating loop. The
  // public contract requires count > 0; reject 0 instead of forwarding it.
  _CCCL_ASSERT(count > 0, "repeat count must be > 0");
  if (count == 0)
  {
    return nullptr;
  }
  auto* sctx = from_opaque_sctx(ctx);
  return to_opaque_repeat(stf_try_allocate([sctx, count] {
    return new repeat_graph_scope_guard(*sctx, count);
  }));
}

void stf_stackable_pop_repeat(stf_repeat_scope_handle scope)
{
  delete from_opaque_repeat(scope);
}

void stf_stackable_while_cond_scalar(
  stf_ctx_handle ctx,
  stf_while_scope_handle scope,
  stf_logical_data_handle ld,
  stf_compare_op op,
  double threshold,
  stf_dtype dtype)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  _CCCL_ASSERT(scope != nullptr, "while scope handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");

  auto* sctx                             = from_opaque_sctx(ctx);
  auto* guard                            = from_opaque_while(scope);
  cudaGraphConditionalHandle cond_handle = guard->cond_handle();

  const int offset = sctx->get_head_offset();

  // Validate (and auto-push if necessary) the read access on this scope,
  // then materialise the untyped logical_data for the task dep.  The
  // concrete stackable_logical_data<T> is dispatched through visit_sld()
  // so both slice<char>-backed data and void_interface tokens resolve
  // correctly; the while-condition kernel below only makes sense on a
  // scalar-typed slice, but validate_access/get_ld are type-agnostic.
  logical_data_untyped ld_ut = visit_sld(ld, [&](auto& sld) {
    sld.validate_access(offset, *sctx, access_mode::read);
    return logical_data_untyped{sld.get_ld(offset)};
  });

  auto& underlying_ctx = sctx->get_ctx(offset);
  auto task            = underlying_ctx.task();
  task.add_deps(task_dep_untyped(ld_ut, access_mode::read));
  task.set_symbol("while_condition");
  task.enable_capture();
  task.start();

  auto stream     = task.get_stream();
  auto s          = task.template get<slice<const char>>(0);
  const void* ptr = s.data_handle();

  switch (dtype)
  {
    case STF_DTYPE_FLOAT32:
      stf_stackable_while_cond_kernel<float>
        <<<1, 1, 0, stream>>>(static_cast<const float*>(ptr), cond_handle, threshold, op);
      break;
    case STF_DTYPE_FLOAT64:
      stf_stackable_while_cond_kernel<double>
        <<<1, 1, 0, stream>>>(static_cast<const double*>(ptr), cond_handle, threshold, op);
      break;
    case STF_DTYPE_INT32:
      stf_stackable_while_cond_kernel<int>
        <<<1, 1, 0, stream>>>(static_cast<const int*>(ptr), cond_handle, threshold, op);
      break;
    case STF_DTYPE_INT64:
      stf_stackable_while_cond_kernel<long long>
        <<<1, 1, 0, stream>>>(static_cast<const long long*>(ptr), cond_handle, threshold, op);
      break;
    default:
      _CCCL_ASSERT(false, "unsupported dtype for stf_stackable_while_cond_scalar");
      break;
  }

  task.end();
}

#endif // _CCCL_CTK_AT_LEAST(12, 4)

stf_logical_data_handle
stf_stackable_logical_data_with_place(stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  _CCCL_ASSERT(dplace != nullptr, "data_place handle must not be null");

  auto* sctx = from_opaque_sctx(ctx);
  auto sld   = sctx->logical_data(make_slice(static_cast<char*>(addr), sz), *from_opaque(dplace));
  return to_opaque_sld(stf_try_allocate([&sld] {
    ::std::unique_ptr<stackable_ld_t> inner{new stackable_ld_t{::std::move(sld)}};
    auto* w = new stackable_ld_opaque{false, inner.get()};
    inner.release();
    return w;
  }));
}

stf_logical_data_handle stf_stackable_logical_data(stf_ctx_handle ctx, void* addr, size_t sz)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");

  auto* sctx = from_opaque_sctx(ctx);
  auto sld   = sctx->logical_data(make_slice(static_cast<char*>(addr), sz), data_place::host());
  return to_opaque_sld(stf_try_allocate([&sld] {
    ::std::unique_ptr<stackable_ld_t> inner{new stackable_ld_t{::std::move(sld)}};
    auto* w = new stackable_ld_opaque{false, inner.get()};
    inner.release();
    return w;
  }));
}

stf_logical_data_handle stf_stackable_logical_data_empty(stf_ctx_handle ctx, size_t length)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");

  auto* sctx = from_opaque_sctx(ctx);
  auto sld   = sctx->logical_data(shape_of<slice<char>>(length));
  return to_opaque_sld(stf_try_allocate([&sld] {
    ::std::unique_ptr<stackable_ld_t> inner{new stackable_ld_t{::std::move(sld)}};
    auto* w = new stackable_ld_opaque{false, inner.get()};
    inner.release();
    return w;
  }));
}

stf_logical_data_handle stf_stackable_logical_data_no_export_empty(stf_ctx_handle ctx, size_t length)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");

  auto* sctx = from_opaque_sctx(ctx);
  auto sld   = sctx->logical_data_no_export(shape_of<slice<char>>(length));
  return to_opaque_sld(stf_try_allocate([&sld] {
    ::std::unique_ptr<stackable_ld_t> inner{new stackable_ld_t{::std::move(sld)}};
    auto* w = new stackable_ld_opaque{false, inner.get()};
    inner.release();
    return w;
  }));
}

stf_logical_data_handle stf_stackable_token(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");

  auto* sctx = from_opaque_sctx(ctx);
  auto token = sctx->token();
  // Tokens use void_interface internally; the wrapper's is_token flag tells
  // every entry point to dispatch through stackable_token_t (see visit_sld).
  // stf_stackable_token_destroy() is still required for release so callers
  // can match creation / destruction by name.
  return to_opaque_sld(stf_try_allocate([&token] {
    ::std::unique_ptr<stackable_token_t> inner{new stackable_token_t{::std::move(token)}};
    auto* w = new stackable_ld_opaque{true, inner.get()};
    inner.release();
    return w;
  }));
}

void stf_stackable_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");
  _CCCL_ASSERT(symbol != nullptr, "symbol must not be null");
  visit_sld(ld, [symbol](auto& sld) {
    sld.set_symbol(symbol);
  });
}

void stf_stackable_logical_data_set_read_only(stf_logical_data_handle ld)
{
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");
  visit_sld(ld, [](auto& sld) {
    sld.set_read_only();
  });
}

void stf_stackable_logical_data_push(stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle dplace)
{
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");
  visit_sld(ld, [m, dplace](auto& sld) {
    if (dplace != nullptr)
    {
      sld.push(access_mode(m), *from_opaque(dplace));
    }
    else
    {
      sld.push(access_mode(m));
    }
  });
}

void stf_stackable_logical_data_destroy(stf_logical_data_handle ld)
{
  if (ld == nullptr)
  {
    return;
  }
  auto* w = from_opaque_sld_wrapper(ld);
  _CCCL_ASSERT(!w->is_token,
               "stf_stackable_logical_data_destroy called on a token handle; use stf_stackable_token_destroy instead");
  delete static_cast<stackable_ld_t*>(w->impl);
  delete w;
}

void stf_stackable_token_destroy(stf_logical_data_handle ld)
{
  if (ld == nullptr)
  {
    return;
  }
  auto* w = from_opaque_sld_wrapper(ld);
  _CCCL_ASSERT(w->is_token,
               "stf_stackable_token_destroy called on a non-token handle; use stf_stackable_logical_data_destroy");
  delete static_cast<stackable_token_t*>(w->impl);
  delete w;
}

stf_task_handle stf_stackable_task_create(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");

  auto* sctx           = from_opaque_sctx(ctx);
  const int offset     = sctx->get_head_offset();
  auto& underlying_ctx = sctx->get_ctx(offset);
  return to_opaque(stf_try_allocate([&underlying_ctx] {
    return new context::unified_task<>{underlying_ctx.task()};
  }));
}

void stf_stackable_task_add_dep(stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");

  auto* sctx     = from_opaque_sctx(ctx);
  auto* task_ptr = from_opaque(t);

  const int offset = sctx->get_head_offset();
  // Validate access and auto-push data across scope boundaries before
  // binding, dispatching on the concrete stackable_logical_data<T> kind so
  // that slice<char>-backed data and void_interface tokens both flow through
  // the correct freeze/unfreeze machinery.
  logical_data_untyped ld_ut = visit_sld(ld, [&](auto& sld) {
    sld.validate_access(offset, *sctx, access_mode(m));
    return logical_data_untyped{sld.get_ld(offset)};
  });
  task_ptr->add_deps(task_dep_untyped(ld_ut, access_mode(m)));
}

void stf_stackable_task_add_dep_with_dplace(
  stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  _CCCL_ASSERT(t != nullptr, "task handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");
  _CCCL_ASSERT(data_p != nullptr, "data_place handle must not be null");

  auto* sctx     = from_opaque_sctx(ctx);
  auto* task_ptr = from_opaque(t);

  const int offset           = sctx->get_head_offset();
  logical_data_untyped ld_ut = visit_sld(ld, [&](auto& sld) {
    sld.validate_access(offset, *sctx, access_mode(m));
    return logical_data_untyped{sld.get_ld(offset)};
  });
  task_ptr->add_deps(task_dep_untyped(ld_ut, access_mode(m), *from_opaque(data_p)));
}

stf_host_launch_handle stf_stackable_host_launch_create(stf_ctx_handle ctx)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");

  auto* sctx           = from_opaque_sctx(ctx);
  const int offset     = sctx->get_head_offset();
  auto& underlying_ctx = sctx->get_ctx(offset);
  return to_opaque(stf_try_allocate([&underlying_ctx] {
    return new context::host_launch_builder{underlying_ctx.host_launch()};
  }));
}

void stf_stackable_host_launch_add_dep(
  stf_ctx_handle ctx, stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
{
  _CCCL_ASSERT(ctx != nullptr, "stackable context handle must not be null");
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(ld != nullptr, "stackable logical data handle must not be null");

  auto* sctx      = from_opaque_sctx(ctx);
  auto* scope_ptr = static_cast<context::host_launch_builder*>(static_cast<void*>(h));

  const int offset           = sctx->get_head_offset();
  logical_data_untyped ld_ut = visit_sld(ld, [&](auto& sld) {
    sld.validate_access(offset, *sctx, access_mode(m));
    return logical_data_untyped{sld.get_ld(offset)};
  });
  scope_ptr->add_deps(task_dep_untyped(ld_ut, access_mode(m)));
}

void stf_stackable_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
{
  _CCCL_ASSERT(h != nullptr, "host launch handle must not be null");
  _CCCL_ASSERT(callback != nullptr, "callback must not be null");

  auto* scope_ptr = static_cast<context::host_launch_builder*>(static_cast<void*>(h));
  (*scope_ptr)->*[callback](reserved::host_launch_deps& deps) {
    callback(to_opaque(&deps));
  };
}

void stf_stackable_host_launch_destroy(stf_host_launch_handle h)
{
  if (h == nullptr)
  {
    return;
  }
  delete static_cast<context::host_launch_builder*>(static_cast<void*>(h));
}

} // extern "C"
