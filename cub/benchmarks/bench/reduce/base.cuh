// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_reduce.cuh>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
struct arch_policies
{
  _CCCL_API constexpr auto operator()(int /*arch*/) const -> ::cub::reduce_arch_policy
  {
    const auto policy = cub::agent_reduce_policy{
      .block_threads      = TUNE_THREADS_PER_BLOCK,
      .items_per_thread   = TUNE_ITEMS_PER_THREAD,
      .vector_load_length = 1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      .block_algorithm    = cub::BLOCK_REDUCE_WARP_REDUCTIONS,
      .load_modifier      = cub::LOAD_DEFAULT};
    return {policy, policy, policy, policy};
  }
};
#endif // !TUNE_BASE

#if 0
struct caching_last_alloc_mr
{
  void* last_ptr = nullptr;
  ::cuda::stream_ref last_stream;
  size_t last_size;
  bool in_use = false;

  caching_last_alloc_mr() = default;

  caching_last_alloc_mr(const caching_last_alloc_mr& other)
  {
    if (other.last_ptr)
    {
      throw std::runtime_error("Copying caching_last_alloc_mr after first allocation is not implemented");
    }
  }

  caching_last_alloc_mr& operator=(const caching_last_alloc_mr&) = delete;

  ~caching_last_alloc_mr()
  {
    if (last_ptr)
    {
      _CCCL_TRY_CUDA_API(::cudaFreeAsync, "cudaFreeAsync failed", last_ptr, last_stream.get());
    }
  }

  void* allocate(::cuda::stream_ref stream, size_t size, size_t alignment = 1)
  {
    if (last_ptr && (stream != last_stream || size != last_size))
    {
      throw std::runtime_error("Cannot allocate again with a different stream or size");
    }
    if (in_use)
    {
      throw std::runtime_error("Cannot allocate again before deallocating");
    }
    _CCCL_TRY_CUDA_API(::cudaMallocAsync, "cudaMallocAsync failed", &last_ptr, size, stream.get());
    _CCCL_ASSERT(cuda::is_aligned(last_ptr, alignment), "");
    last_stream = stream;
    last_size   = size;
    in_use      = true;
    return last_ptr;
  }

  void deallocate(::cuda::stream_ref /* stream */, void* /* ptr */, size_t /* size */, size_t /* alignment */ = 1)
  {
    in_use = false;
  }

  void* allocate_sync(size_t /* size */, size_t /* alignment */)
  {
    throw std::runtime_error("Not implemented");
  }

  void deallocate_sync(void* /* ptr */, size_t /* size */, size_t /* alignment */)
  {
    throw std::runtime_error("Not implemented");
  }

  friend auto operator==(const caching_last_alloc_mr& a, const caching_last_alloc_mr& b) -> bool
  {
    return a.last_ptr == b.last_ptr;
  }

  friend auto operator!=(const caching_last_alloc_mr& a, const caching_last_alloc_mr& b) -> bool
  {
    return !(a == b);
  }

  // to support querying when used as environment
  auto get_memory_resource() const -> caching_last_alloc_mr
  {
    return {};
  }
};

static_assert(cuda::mr::resource<caching_last_alloc_mr>);
#endif

template <typename T, typename OffsetT>
void reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using init_t   = T;

  // Retrieve axis parameters
  const auto elements = static_cast<offset_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // FIXME(bgruber): the previous implementation did target cub::DispatchReduce, and provided T as accumulator type.
  // This is not realistic, since a user cannot set the accumulator type the same way at the public API. For example,
  // reducing I8 over cuda::std::plus deduces accumulator type I32 at the public API, but the benchmark forces it to I8.
  // This skews the MemBoundScaling, leading to 20% regression for the same tuning when the public API is called (with
  // accum_t I32) over the benchmark (forced accum_t of I8). See also: https://github.com/NVIDIA/cccl/issues/6576
#if 0
  caching_last_alloc_mr mr;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = ::cuda::std::execution::env{
      ::cuda::stream_ref{launch.get_stream().get_stream()},
      ::cuda::std::execution::prop{::cuda::mr::__get_memory_resource, mr}
#  if !TUNE_BASE
      ,
      ::cuda::std::execution::prop{
        ::cuda::execution::__get_tuning_t,
        ::cuda::std::execution::env{
          ::cuda::std::execution::prop{::cub::detail::reduce::get_tuning_query_t, arch_policies{}}}}
#  endif
    };
    static_assert(::cuda::std::execution::__queryable_with<decltype(env), ::cuda::mr::__get_memory_resource_t>);
    (void) cub::DeviceReduce::Reduce(d_in, d_out, elements, op_t{}, init_t{}, env);
  });
#endif

  // So for now, we have to call into the dispatcher again to override the accumulator type:
  auto transform_op = ::cuda::std::identity{};

  std::size_t temp_size;
  cub::detail::reduce::dispatch</* OverrideAccumT = */ T>(
    nullptr,
    temp_size,
    d_in,
    d_out,
    static_cast<offset_t>(elements),
    op_t{},
    init_t{},
    0 /* stream */,
    transform_op
#if !TUNE_BASE
    ,
    arch_policies{}
#endif
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::reduce::dispatch</* OverrideAccumT = */ T>(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      static_cast<offset_t>(elements),
      op_t{},
      init_t{},
      launch.get_stream(),
      transform_op
#if !TUNE_BASE
      ,
      arch_policies{}
#endif
    );
  });
}

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
