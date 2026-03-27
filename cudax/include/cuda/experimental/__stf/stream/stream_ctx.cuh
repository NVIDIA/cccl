//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implements the stream_ctx backend that uses CUDA streams and CUDA events to synchronize tasks
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/internal/acquire_release.cuh>
#include <cuda/experimental/__stf/internal/backend_allocator_setup.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/internal/cuda_kernel_scope.cuh>
#include <cuda/experimental/__stf/internal/host_launch_scope.cuh>
#include <cuda/experimental/__stf/internal/launch.cuh>
#include <cuda/experimental/__stf/internal/parallel_for_scope.cuh>
#include <cuda/experimental/__stf/places/blocked_partition.cuh> // for unit test!
#include <cuda/experimental/__stf/stream/interfaces/slice.cuh> // For implicit logical_data_untyped constructors
#include <cuda/experimental/__stf/stream/interfaces/void_interface.cuh>
#include <cuda/experimental/__stf/stream/stream_task.cuh>
#include <cuda/experimental/__stf/utility/threads.cuh> // for reserved::counter

namespace cuda::experimental::stf
{
template <typename T>
struct streamed_interface_of;

/**
 * @brief Uncached allocator (used as a basis for other allocators)
 *
 * Any allocation/deallocation results in an actual underlying CUDA API call
 * (e.g. cudaMallocAsync). This allocator should generally not be used
 * directly, but used within an allocator with a more advanced strategy
 * (caching, heap allocation, ...)
 */
class uncached_stream_allocator : public block_allocator_interface
{
public:
  uncached_stream_allocator() = default;

  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    auto dstream = memory_node.getDataStream();

    if (!memory_node.allocation_is_stream_ordered())
    {
      // Blocking allocation (e.g., cudaMallocHost, cudaMallocManaged) - no stream synchronization needed
      return memory_node.allocate(s, dstream.stream);
    }

    // Stream-ordered allocation - synchronize prereqs with the stream
    auto op = stream_async_op(ctx, dstream, prereqs);
    if (ctx.generate_event_symbols())
    {
      op.set_symbol("allocate");
    }
    void* result = memory_node.allocate(s, dstream.stream);
    prereqs      = op.end(ctx);
    return result;
  }

  void deallocate(
    backend_ctx_untyped& ctx, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) override
  {
    auto dstream = memory_node.getDataStream();

    if (!memory_node.allocation_is_stream_ordered())
    {
      // Blocking deallocation - synchronize stream first, then free
      cuda_safe_call(cudaStreamSynchronize(dstream.stream));
      memory_node.deallocate(ptr, sz, dstream.stream);
      return;
    }

    // Stream-ordered deallocation - synchronize prereqs with the stream
    auto op = stream_async_op(ctx, dstream, prereqs);
    if (ctx.generate_event_symbols())
    {
      op.set_symbol("deallocate");
    }
    memory_node.deallocate(ptr, sz, dstream.stream);
    prereqs = op.end(ctx);
  }

  // Nothing is done as all deallocation are done immediately
  event_list deinit(backend_ctx_untyped& /* ctx */) override
  {
    return event_list();
  }

  ::std::string to_string() const override
  {
    return "uncached stream allocator";
  }
};

/** This class describes a CUDASTF execution context where CUDA streams and
 *   CUDA events are used as synchronization primitives.
 *
 * This class is copyable, movable, and can be passed by value
 */
class stream_ctx : public backend_ctx<stream_ctx>
{
  using base = backend_ctx<stream_ctx>;

public:
  using task_type = stream_task<>;

  /**
   * @brief Definition for the underlying implementation of `data_interface<T>`
   *
   * @tparam T
   */
  template <typename T>
  using data_interface = typename streamed_interface_of<T>::type;

  /// @brief This type is copyable, assignable, and movable. However, copies have reference semantics.
  ///@{
  stream_ctx(async_resources_handle handle = async_resources_handle(nullptr))
      : backend_ctx<stream_ctx>(::std::make_shared<impl>(mv(handle)))
  {}
  stream_ctx(cudaStream_t user_stream, async_resources_handle handle = async_resources_handle(nullptr))
      : backend_ctx<stream_ctx>(::std::make_shared<impl>(mv(handle)))
  {
    // We set the user stream as the entry point after the creation of the
    // context so that we can manipulate the object, not its shared_ptr
    // implementation
    set_user_stream(user_stream);
  }

  ///@}

  void set_user_stream(cudaStream_t user_stream)
  {
    // TODO first introduce the user stream in our pool
    auto dstream = decorated_stream(user_stream);

    this->state().user_dstream = dstream;

    // Create an event in the stream
    auto start_e = reserved::record_event_in_stream(dstream);
    get_state().add_start_events(*this, event_list(mv(start_e)));

    // When a stream is attached to the context creation, the finalize()
    // semantic is non-blocking
    this->state().blocking_finalize = false;
  }

  // Indicate if the finalize() call should be blocking or not
  bool blocking_finalize = true;

  using backend_ctx<stream_ctx>::task;

  /**
   * @brief Creates a task on the specified execution place
   */
  template <typename... Deps>
  stream_task<Deps...> task(exec_place e_place, task_dep<Deps>... deps)
  {
    return stream_task<Deps...>(*this, mv(e_place), mv(deps)...);
  }

  cudaStream_t fence()
  {
    const auto& user_dstream = state().user_dstream;
    // We either use the user-provided stream, or we get one stream from the pool
    decorated_stream dstream =
      (user_dstream.has_value())
        ? user_dstream.value()
        : exec_place::current_device().getStream(true /* stream for computation */);

    auto prereqs = get_state().insert_fence(*get_dot());

    prereqs.optimize(*this);

    // The output event is used for the tools in practice so we can ignore it
    /* auto before_e = */ reserved::join_with_stream(*this, dstream, prereqs, "fence", false);

    return dstream.stream;
  }

  // no-op for stream_ctx, needed so that context (variant of stream_ctx/graph_ctx) can dispatch submit()
  void submit() {}

  void finalize()
  {
    _CCCL_ASSERT(get_phase() < backend_ctx_untyped::phase::finalized, "");
    auto& state = this->state();

    cudaStream_t submitted_stream = fence();

    // Write-back data and erase automatically created data instances
    state.erase_all_logical_data();
    state.detach_allocators(*this);

    // Make sure we release resources attached to this context
    state.release_ctx_resources(submitted_stream);

    if (state.blocking_finalize)
    {
      cuda_safe_call(cudaStreamSynchronize(submitted_stream));
    }
    state.cleanup();
    set_phase(backend_ctx_untyped::phase::finalized);
  }

  // no-op : so that we can use the same code with stream_ctx and graph_ctx
  void change_stage() {}

  template <typename T>
  auto wait(cuda::experimental::stf::logical_data<T>& ldata)
  {
    typename owning_container_of<T>::type out;

    task(exec_place::host(), ldata.read()).set_symbol("wait")->*[&](cudaStream_t stream, auto data) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      out = owning_container_of<T>::get_value(data);
    };

    return out;
  }

private:
  /* This class contains all the state associated to a stream_ctx, and all states associated to every contexts (in
   * `impl`) */
  class impl : public base::impl
  {
  public:
    impl(async_resources_handle _async_resources = async_resources_handle(nullptr))
        : base::impl(mv(_async_resources))
    {
      reserved::backend_ctx_setup_allocators<impl, uncached_stream_allocator>(*this);
    }

    void cleanup()
    {
      base::impl::cleanup();
    }

    // Due to circular dependencies, we need to define it here, and not in backend_ctx_untyped
    void update_uncached_allocator(block_allocator_untyped custom) override
    {
      reserved::backend_ctx_update_uncached_allocator(*this, mv(custom));
    }

    event_list stream_to_event_list(cudaStream_t stream, ::std::string symbol) const override
    {
      auto e = reserved::record_event_in_stream(decorated_stream(stream), *get_dot(), mv(symbol));
      return event_list(mv(e));
    }

    ::std::string to_string() const override
    {
      return "stream backend context";
    }

    // We need to ensure all dangling events have been completed (eg. by having
    // the CUDA stream used in the finalization wait on these events)
    bool track_dangling_events() const override
    {
      return true;
    }

    // If the context is attached to a user stream, we should use it for
    // finalize() or fence()
    ::std::optional<decorated_stream> user_dstream;

    /* By default, the finalize operation is blocking, unless user provided
     * a stream when creating the context */
    bool blocking_finalize = true;
  };

  impl& state()
  {
    return dynamic_cast<impl&>(get_state());
  }
  const impl& state() const
  {
    return dynamic_cast<const impl&>(get_state());
  }
};

#ifdef UNITTESTED_FILE
UNITTEST("copyable stream_task")
{
  stream_ctx ctx;
  stream_task<> t     = ctx.task();
  stream_task<> t_cpy = t;
  ctx.finalize();
};

UNITTEST("movable stream_task")
{
  stream_ctx ctx;
  stream_task<> t     = ctx.task();
  stream_task<> t_cpy = mv(t);
  ctx.finalize();
};

// FIXME : This test is causing some compiler errors with MSVC, so we disable
// it on MSVC for now
#  if !_CCCL_COMPILER(MSVC)
UNITTEST("logical_data_untyped moveable")
{
  using namespace cuda::experimental::stf;

  class scalar
  {
  public:
    scalar(stream_ctx& ctx)
    {
      size_t s       = sizeof(double);
      double* h_addr = (double*) malloc(s);
      cuda_safe_call(cudaHostRegister(h_addr, s, cudaHostRegisterPortable));
      handle = ctx.logical_data(h_addr, 1);
    }

    scalar& operator=(scalar&& rhs)
    {
      handle = mv(rhs.handle);
      return *this;
    }

    logical_data_untyped handle;
  };

  stream_ctx ctx;
  scalar global_res(ctx);

  for (int bid = 0; bid < 2; bid++)
  {
    // This variable is likely to have the same address across loop
    // iterations, which can stress the logical data management
    scalar res(ctx);
    auto t = ctx.task();
    t.add_deps(res.handle.write());
    t.start();
    t.end();

    global_res = mv(res);
  }

  ctx.finalize();
};
#  endif // !_CCCL_COMPILER(MSVC)

#  if _CCCL_CUDA_COMPILATION()
namespace reserved
{
static __global__ void dummy() {}
} // namespace reserved

UNITTEST("copyable stream_ctx")
{
  stream_ctx ctx;
  stream_ctx ctx2 = ctx;
  auto t          = ctx.task();
  auto t2         = ctx2.task();

  ctx2.finalize();
};

UNITTEST("movable stream_ctx")
{
  stream_ctx ctx;
  auto t          = ctx.task();
  stream_ctx ctx2 = mv(ctx);
  auto t2         = ctx2.task();

  ctx2.finalize();
};

UNITTEST("stream context basics")
{
  stream_ctx ctx;

  double X[1024], Y[1024];
  auto handle_X = ctx.logical_data(X);
  auto handle_Y = ctx.logical_data(Y);

  for (int k = 0; k < 10; k++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto /*unuased*/) {
      reserved::dummy<<<1, 1, 0, s>>>();
    };
  }

  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t s, auto /*unuased*/, auto /*unuased*/) {
    reserved::dummy<<<1, 1, 0, s>>>();
  };

  ctx.finalize();
};

UNITTEST("stream context after logical data")
{
  stream_ctx ctx;
  logical_data<slice<double>> handle_X, handle_Y;

  double X[1024], Y[1024];
  handle_X = ctx.logical_data(X);
  handle_Y = ctx.logical_data(Y);

  for (int k = 0; k < 10; k++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto /*unused*/) {
      reserved::dummy<<<1, 1, 0, s>>>();
    };
  }

  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t s, auto /*unused*/, auto /*unused*/) {
    reserved::dummy<<<1, 1, 0, s>>>();
  };

  ctx.finalize();
};

UNITTEST("set_symbol on stream_task and stream_task<>")
{
  stream_ctx ctx;

  double X[1024], Y[1024];
  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  stream_task<> t = ctx.task();
  t.add_deps(lX.rw(), lY.rw());
  t.set_symbol("stream_task<>");
  t.start();
  t.end();

  stream_task<slice<double>, slice<double>> t2 = ctx.task(lX.rw(), lY.rw());
  t2.set_symbol("stream_task");
  t2.start();
  t2.end();

  ctx.finalize();
};

UNITTEST("non contiguous slice")
{
  stream_ctx ctx;

  int X[32 * 32];

  // Pinning non contiguous memory is extremely expensive, so we do it now
  cuda_safe_call(cudaHostRegister(&X[0], 32 * 32 * sizeof(int), cudaHostRegisterPortable));

  for (size_t i = 0; i < 32 * 32; i++)
  {
    X[i] = 1;
  }

  // Create a non-contiguous slice
  auto lX = ctx.logical_data(make_slice(&X[0], ::std::tuple{24, 32}, 32));

  ctx.host_launch(lX.rw())->*[](auto sX) {
    for (size_t i = 0; i < sX.extent(0); i++)
    {
      for (size_t j = 0; j < sX.extent(1); j++)
      {
        sX(i, j) = 2;
      }
    }
  };

  ctx.finalize();

  for (size_t j = 0; j < 32; j++)
  {
    for (size_t i = 0; i < 32; i++)
    {
      size_t ind   = i + 32 * j;
      int expected = ((i < 24) ? 2 : 1);
      EXPECT(X[ind] == expected);
    }
  }

  cuda_safe_call(cudaHostUnregister(&X[0]));
};

UNITTEST("logical data from a shape of 2D slice")
{
  stream_ctx ctx;

  // Create a non-contiguous slice
  auto lX = ctx.logical_data(shape_of<slice<char, 2>>(24, 32));

  // We cannot use the parallel_for construct in a UNITTEST so we here rely
  // on the fact that X is likely going to be allocated in a contiguous
  // fashion on devices, and we use cudaMemsetAsync.
  ctx.task(lX.write())->*[](cudaStream_t stream, auto sX) {
    assert(contiguous_dims(sX) == 2);
    cudaMemsetAsync(sX.data_handle(), 42, sX.size() * sizeof(char), stream);
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < sX.extent(0); i++)
    {
      for (size_t j = 0; j < sX.extent(1); j++)
      {
        EXPECT(sX(i, j) == 42);
      }
    }
  };

  ctx.finalize();
};

UNITTEST("logical data from a shape of 3D slice")
{
  stream_ctx ctx;

  // Create a non-contiguous slice
  auto lX = ctx.logical_data(shape_of<slice<char, 3>>(24, 32, 16));

  // We cannot use the parallel_for construct in a UNITTEST so we here rely
  // on the fact that X is likely going to be allocated in a contiguous
  // fashion on devices, and we use cudaMemsetAsync.
  ctx.task(lX.write())->*[](cudaStream_t stream, auto sX) {
    EXPECT(contiguous_dims(sX) == 3);
    cudaMemsetAsync(sX.data_handle(), 42, sX.size() * sizeof(char), stream);
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < sX.extent(0); i++)
    {
      for (size_t j = 0; j < sX.extent(1); j++)
      {
        for (size_t k = 0; k < sX.extent(2); k++)
        {
          EXPECT(sX(i, j, k) == 42);
        }
      }
    }
  };

  ctx.finalize();
};

UNITTEST("const class containing a stream_ctx")
{
  struct foo
  {
    foo() = default;
    stream_ctx& get_ctx() const
    {
      return ctx;
    }

    mutable stream_ctx ctx;
  };

  const foo f;

  // Create a non-contiguous slice
  auto lX = f.get_ctx().logical_data(shape_of<slice<char>>(24));

  f.get_ctx().finalize();
};

#  endif

UNITTEST("stream ctx loop")
{
  for (size_t iter = 0; iter < 10; iter++)
  {
    stream_ctx ctx;
    auto lA = ctx.logical_data(shape_of<slice<char>>(64));
    ctx.task(lA.write())->*[](cudaStream_t, auto) {};
    ctx.finalize();
  }
};

UNITTEST("stream ctx loop custom allocator")
{
  for (size_t iter = 0; iter < 10; iter++)
  {
    stream_ctx ctx;
    ctx.set_allocator(block_allocator<pooled_allocator>(ctx));
    auto lA = ctx.logical_data(shape_of<slice<char>>(64));
    ctx.task(lA.write())->*[](cudaStream_t, auto) {};
    ctx.finalize();
  }
};

// This test ensures we can convert a logical_data_untyped to a logical_data
// It is located here because we need all other classes to actually test it
UNITTEST("get logical_data from a task_dep")
{
  stream_ctx ctx;

  using T = slice<size_t>;
  auto lA = ctx.logical_data(shape_of<T>(64));

  // Create a task dependency using that logical data
  auto d = lA.read();

  logical_data_untyped ul = d.get_data();
  EXPECT(ul == lA);

  logical_data<T> lB = d.get_data();
  EXPECT(lB == lA);

  auto lC = logical_data<T>(d.get_data());
  EXPECT(lC == lA);

  auto lD = static_cast<logical_data<T>>(d.get_data());
  EXPECT(lD == lA);

  ctx.finalize();
};

#  if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
namespace reserved
{
inline void unit_test_pfor()
{
  stream_ctx ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(lA.shape(), lA.write())->*[] _CCCL_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
}

UNITTEST("basic parallel_for test")
{
  unit_test_pfor();
};

inline void unit_test_pfor_integral_shape()
{
  stream_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));

  // Directly use 64 as a shape here
  ctx.parallel_for(64, lA.write())->*[] _CCCL_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("parallel_for with integral shape")
{
  unit_test_pfor_integral_shape();
};

inline void unit_test_host_pfor()
{
  stream_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(exec_place::host(), lA.shape(), lA.write())->*[](size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on host")
{
  unit_test_host_pfor();
};

inline void unit_test_pfor_mix_host_dev()
{
  stream_ctx ctx;

  const int N = 1024;
  double X[N];

  auto lx = ctx.logical_data(X);

  ctx.parallel_for(lx.shape(), lx.write())->*[=] _CCCL_DEVICE(size_t pos, auto sx) {
    sx(pos) = 17 * pos + 4;
  };

  ctx.parallel_for(exec_place::host(), lx.shape(), lx.rw())->*[=](size_t pos, auto sx) {
    sx(pos) = sx(pos) * sx(pos);
  };

  ctx.parallel_for(lx.shape(), lx.rw())->*[=] _CCCL_DEVICE(size_t pos, auto sx) {
    sx(pos) = sx(pos) - 7;
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    EXPECT(X[i] == (17 * i + 4) * (17 * i + 4) - 7);
  }
}

/* This test ensures that parallel_for on exec_place::host are properly
 * interleaved with those on devices. */
UNITTEST("parallel_for host and device")
{
  unit_test_pfor_mix_host_dev();
};

inline void unit_test_untyped_place_pfor()
{
  stream_ctx ctx;

  exec_place where = exec_place::host();

  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  // We have to put both __host__ __device__ qualifiers as this is resolved
  // dynamically and both host and device codes will be generated
  ctx.parallel_for(where, lA.shape(), lA.write())->*[] _CCCL_HOST_DEVICE(size_t i, slice<size_t> A) {
    // Even if we do have a __device__ qualifier, we are not supposed to call it
    NV_IF_TARGET(NV_IS_DEVICE, (assert(0);))
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on host (untyped execution place)")
{
  unit_test_untyped_place_pfor();
};

inline void unit_test_pfor_grid()
{
  stream_ctx ctx;
  auto where = exec_place::all_devices();
  auto lA    = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(blocked_partition(), where, lA.shape(), lA.write())->*[] _CCCL_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on grid")
{
  unit_test_pfor_grid();
};

inline void unit_test_pfor_untyped_grid()
{
  stream_ctx ctx;

  exec_place where = exec_place::repeat(exec_place::current_device(), 4);

  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(blocked_partition(), where, lA.shape(), lA.write())->*[] _CCCL_HOST_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on grid")
{
  unit_test_pfor_untyped_grid();
};

inline void unit_test_launch()
{
  stream_ctx ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.launch(lA.write())->*[] _CCCL_DEVICE(auto t, slice<size_t> A) {
    for (auto i : t.apply_partition(shape(A)))
    {
      A(i) = 2 * i;
    }
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
}

UNITTEST("basic launch test")
{
  unit_test_launch();
};
} // end namespace reserved
#  endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

#endif // UNITTESTED_FILE
} // namespace cuda::experimental::stf
