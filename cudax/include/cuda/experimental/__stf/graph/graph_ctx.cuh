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
 * @brief Implements the graph_ctx backend that executes tasks using the CUDA graph API
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

#include <cuda/experimental/__stf/graph/graph_task.cuh>
#include <cuda/experimental/__stf/graph/interfaces/slice.cuh>
#include <cuda/experimental/__stf/internal/acquire_release.cuh>
#include <cuda/experimental/__stf/internal/backend_allocator_setup.cuh>
#include <cuda/experimental/__stf/internal/launch.cuh>
#include <cuda/experimental/__stf/internal/parallel_for_scope.cuh>
#include <cuda/experimental/__stf/places/blocked_partition.cuh> // for unit test!

namespace cuda::experimental::stf
{

namespace reserved
{

// For counters
class graph_tag
{
public:
  class launch
  {};
  class instantiate
  {};
  class update
  {
  public:
    class success
    {};
    class failure
    {};
  };
};

} // end namespace reserved

/**
 * @brief Uncached allocator (used as a base for other allocators)
 *
 * Any allocation/deallocation results in an actual underlying CUDA API call
 * (e.g. `cudaGraphAddMemAllocNode`). This allocator should generally not be used
 * directly, but used within an allocator with a more advanced strategy
 * (caching, heap allocation, ...)
 */
class uncached_graph_allocator : public block_allocator_interface
{
public:
  uncached_graph_allocator() = default;

  void*
  allocate(backend_ctx_untyped& bctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    // This is not implemented yet
    EXPECT(!memory_node.is_composite(), "Composite data places are not implemented yet.");

    void* result = nullptr;

    const size_t graph_epoch                   = bctx.epoch();
    const cudaGraph_t graph                    = bctx.graph();
    const ::std::vector<cudaGraphNode_t> nodes = reserved::join_with_graph_nodes(prereqs, graph_epoch);
    cudaGraphNode_t out                        = nullptr;

    if (memory_node == data_place::host)
    {
      cuda_try(cudaMallocHost(&result, s));
      SCOPE(fail)
      {
        cudaFreeHost(&result);
      };
      cuda_try(cudaGraphAddEmptyNode(&out, graph, nodes.data(), nodes.size()));
    }
    else
    {
      if (getenv("USE_CUDA_MALLOC"))
      {
        cuda_safe_call(cudaGraphAddEmptyNode(&out, graph, nodes.data(), nodes.size()));
        cuda_try(cudaMalloc(&result, s));
      }
      else
      {
        result = add_mem_alloc_node(memory_node, out, graph, s, nodes);
      }
      assert(s > 0);
    }

    reserved::fork_from_graph_node(bctx, out, graph_epoch, prereqs, "alloc");
    return result;
  }

  void deallocate(
    backend_ctx_untyped& bctx, const data_place& memory_node, event_list& prereqs, void* ptr, size_t /*sz*/) override
  {
    const cudaGraph_t graph                    = bctx.graph();
    const size_t graph_epoch                   = bctx.epoch();
    cudaGraphNode_t out                        = nullptr;
    const ::std::vector<cudaGraphNode_t> nodes = reserved::join_with_graph_nodes(prereqs, graph_epoch);
    if (memory_node == data_place::host)
    {
      // fprintf(stderr, "TODO deallocate host memory (graph_ctx)\n");
      cuda_safe_call(cudaGraphAddEmptyNode(&out, graph, nodes.data(), nodes.size()));
    }
    else
    {
      cuda_safe_call(cudaGraphAddMemFreeNode(&out, graph, nodes.data(), nodes.size(), ptr));
    }
    reserved::fork_from_graph_node(bctx, out, graph_epoch, prereqs, "dealloc");
  }

  ::std::string to_string() const override
  {
    return "uncached graph allocator";
  }

  // Nothing is done as all deallocation are done immediately
  event_list deinit(backend_ctx_untyped& /* ctx */) override
  {
    return event_list();
  }

private:
  void* add_mem_alloc_node(
    const data_place& memory_node,
    cudaGraphNode_t& out,
    cudaGraph_t graph,
    size_t s,
    const ::std::vector<cudaGraphNode_t>& input_nodes)
  {
    static const int ndevices = cuda_try<cudaGetDeviceCount>();
    // We need to declare who may access this buffer
    ::std::vector<cudaMemAccessDesc> desc(ndevices);
    for (int peer : each(0, ndevices))
    {
      desc[peer].location.type = cudaMemLocationTypeDevice;
      desc[peer].location.id   = peer;
      desc[peer].flags         = cudaMemAccessFlagsProtReadWrite;
    }

    // Parameter structure for cudaGraphAddMemAllocNode - most values are constants.
    static cudaMemAllocNodeParams params = [] {
      cudaMemAllocNodeParams result{};
      result.poolProps.allocType               = cudaMemAllocationTypePinned;
      result.poolProps.handleTypes             = cudaMemHandleTypeNone;
      result.poolProps.location                = {.type = cudaMemLocationTypeDevice, .id = 0};
      result.poolProps.win32SecurityAttributes = nullptr;
#if CUDA_VERSION >= 12030
      result.poolProps.maxSize = 0;
#endif
      result.accessDescs     = nullptr;
      result.accessDescCount = 0;
      result.bytesize        = 0;
      result.dptr            = nullptr;
      return result;
    }();

    // Set only the variable parameters
    params.poolProps.location.id = device_ordinal(memory_node);
    params.accessDescs           = desc.data();
    params.accessDescCount       = size_t(ndevices);
    params.bytesize              = size_t(s);

    cuda_safe_call(cudaGraphAddMemAllocNode(&out, graph, input_nodes.data(), input_nodes.size(), &params));

    return params.dptr;
  }
};

/**
 * @brief A graph context, which is a CUDA graph that we can automatically built using tasks.
 *
 */
class graph_ctx : public backend_ctx<graph_ctx>
{
  class impl : public backend_ctx<graph_ctx>::impl
  {
  public:
    impl(async_resources_handle _async_resources = async_resources_handle(nullptr))
        : backend_ctx<graph_ctx>::impl(mv(_async_resources))
        , _graph(shared_cuda_graph())
    {
      reserved::backend_ctx_setup_allocators<impl, uncached_graph_allocator>(*this);
    }

    // Note that graph contexts with an explicit graph passed by the user cannot use epochs
    impl(cudaGraph_t g)
        : _graph(wrap_cuda_graph(g))
        , explicit_graph(true)
    {
      reserved::backend_ctx_setup_allocators<impl, uncached_graph_allocator>(*this);
    }

    ~impl() override {}

    // Due to circular dependencies, we need to define it here, and not in backend_ctx_untyped
    void update_uncached_allocator(block_allocator_untyped custom) override
    {
      reserved::backend_ctx_update_uncached_allocator(*this, mv(custom));
    }

    cudaGraph_t graph() const override
    {
      assert(_graph.get());
      return *_graph;
    }

    size_t epoch() const override
    {
      return graph_epoch;
    }

    /* Store a vector of previously instantiated graphs, with the number of
     * nodes, number of edges, the executable graph, and the corresponding epoch.
     * */
    ::std::vector<::std::tuple<size_t, size_t, ::std::shared_ptr<cudaGraphExec_t>, size_t>> previous_exec_graphs;
    cudaStream_t submitted_stream                 = nullptr; // stream used in submit
    ::std::shared_ptr<cudaGraphExec_t> exec_graph = nullptr;
    ::std::shared_ptr<cudaGraph_t> _graph         = nullptr;
    size_t graph_epoch                            = 0;
    bool submitted                                = false; // did we submit ?
    mutable bool explicit_graph                   = false;

    /* By default, the finalize operation is blocking, unless user provided
     * a stream when creating the context */
    bool blocking_finalize = true;
  };

public:
  using task_type = graph_task<>;

  /**
   * @brief Definition for the underlying implementation of `data_interface<T>`
   *
   * @tparam T
   */
  template <typename T>
  using data_interface = typename graphed_interface_of<T>::type;

  /// @brief This type is copyable, assignable, and movable. Howeever, copies have reference semantics.
  ///@{
  graph_ctx(async_resources_handle handle = async_resources_handle(nullptr))
      : backend_ctx<graph_ctx>(::std::make_shared<impl>(mv(handle)))
  {}

  graph_ctx(cudaStream_t user_stream, async_resources_handle handle = async_resources_handle(nullptr))
      : backend_ctx<graph_ctx>(::std::make_shared<impl>(mv(handle)))
  {
    auto& state = this->state();

    // Ensure that we use this stream to launch the graph(s)
    state.submitted_stream = user_stream;

    state.blocking_finalize = false;
  }

  /// @brief Constructor taking a user-provided graph. User code is not supposed to destroy the graph later.
  graph_ctx(cudaGraph_t g)
      : backend_ctx<graph_ctx>(::std::make_shared<impl>(g))
  {}
  ///@}

  ::std::string to_string() const
  {
    return "graph backend context";
  }

  using backend_ctx<graph_ctx>::task;

  /**
   * @brief Creates a typed task on the specified execution place
   */
  template <typename... Deps>
  auto task(exec_place e_place, task_dep<Deps>... deps)
  {
    auto dump_hooks = reserved::get_dump_hooks(this, deps...);
    auto res        = graph_task<Deps...>(*this, get_graph(), get_graph_epoch(), mv(e_place), mv(deps)...);
    res.add_post_submission_hook(dump_hooks);
    return res;
  }

  // submit a new epoch : this will submit a graph in a stream that we return
  cudaStream_t task_fence()
  {
    change_epoch();
    // Get the stream used to submit the graph of the epoch we have
    // launched when changing the epoch
    return this->state().submitted_stream;
  }

  void finalize()
  {
    assert(get_phase() < backend_ctx_untyped::phase::finalized);
    auto& state = this->state();
    if (!state.submitted)
    {
      // Not submitted yet
      submit();
    }
    assert(state.submitted_stream);
    if (state.blocking_finalize)
    {
      cuda_try(cudaStreamSynchronize(state.submitted_stream));
    }
    state.submitted_stream = nullptr;
    state.cleanup();
    set_phase(backend_ctx_untyped::phase::finalized);

#ifdef CUDASTF_DEBUG
    const char* display_stats_env = getenv("CUDASTF_DISPLAY_STATS");
    if (!display_stats_env || atoi(display_stats_env) == 0)
    {
      return;
    }

    fprintf(
      stderr, "[STATS CUDA GRAPHS] instantiated=%lu\n", reserved::counter<reserved::graph_tag::instantiate>.load());
    fprintf(stderr, "[STATS CUDA GRAPHS] launched=%lu\n", reserved::counter<reserved::graph_tag::launch>.load());
    fprintf(stderr,
            "[STATS CUDA GRAPHS] updated=%lu success=%ld failed=%ld\n",
            reserved::counter<reserved::graph_tag::update>.load(),
            reserved::counter<reserved::graph_tag::update::success>.load(),
            reserved::counter<reserved::graph_tag::update::failure>.load());
#endif
  }

  void submit(cudaStream_t stream = nullptr)
  {
    assert(get_phase() < backend_ctx_untyped::phase::submitted);
    auto& state = this->state();
    if (!state.submitted_stream)
    {
      if (stream)
      {
        // We could submit just the last iteration in that specific stream, and synchronize the user-provided
        // stream with this one
        EXPECT(state.graph_epoch == 0, "Cannot call submit with an explicit stream and change epoch.");
        state.submitted_stream = stream;
      }
      else if (!state.submitted_stream)
      {
        state.submitted_stream = pick_stream();
        assert(state.submitted_stream != nullptr);
      }
    }

    // Only for the latest graph ... Will also do write-back and cleanup
    instantiate();

    // cuda_safe_call(cudaStreamSynchronize(state.submitted_stream));

    cuda_try(cudaGraphLaunch(*state.exec_graph, state.submitted_stream));

#ifdef CUDASTF_DEBUG
    reserved::counter<reserved::graph_tag::launch>.increment();
#endif

    // Note that we comment this out for now, so that it is possible to use
    // the print_to_dot method; but we may perhaps discard this graph to
    // some dedicated member variable.
    // Ensure nobody tries to use that graph again ...
    // state.set_graph(nullptr);

    state.submitted = true;
    set_phase(backend_ctx_untyped::phase::submitted);
  }

  // Start a new epoch !
  void change_epoch()
  {
    auto& state = this->state();

    EXPECT(!state.explicit_graph);

    // Put the graph of the current epoch on the vector of previous graphs
    submit_one_epoch(*state._graph, state.graph_epoch);

    // We now use a new graph for that epoch
    state._graph = shared_cuda_graph();

    state.graph_epoch++;

    auto& dot = *get_dot();
    if (dot.is_tracing())
    {
      dot.change_epoch();
    }
    // fprintf(stderr, "Starting epoch %ld : previous graph %p new graph %p\n", state.graph_epoch, prev_graph,
    //         new_graph);
  }

  /// @brief Get the "support" graph associated with the context
  ::std::shared_ptr<cudaGraph_t> get_shared_graph() const
  {
    return state()._graph;
  }

  cudaGraph_t& get_graph() const
  {
    assert(state()._graph);
    return *(state()._graph);
  }

  size_t get_graph_epoch() const
  {
    return state().graph_epoch;
  }

  auto get_exec_graph() const
  {
    return state().exec_graph;
  }

  /* Make sure all pending operations are finished, and release resources */
  ::std::shared_ptr<cudaGraph_t> finalize_as_graph()
  {
    // Write-back data and erase automatically created data instances
    get_state().erase_all_logical_data();
    get_state().detach_allocators(*this);

    // Make sure all pending async ops are sync'ed with
    task_fence_impl();

    auto& state = this->state();

    if (getenv("CUDASTF_DISPLAY_GRAPH_INFO"))
    {
      display_graph_info(get_graph());
    }

    return state._graph;
  }

  // Execute the CUDA graph in the provided stream.
  ::std::shared_ptr<cudaGraphExec_t> instantiate()
  {
    ::std::shared_ptr<cudaGraph_t> g = finalize_as_graph();

    size_t nedges;
    size_t nnodes;

    cuda_safe_call(cudaGraphGetNodes(*g, nullptr, &nnodes));
    cuda_safe_call(cudaGraphGetEdges(*g, nullptr, nullptr, &nedges));

    auto& state = this->state();

    for (const auto& [nnodes_cached, nedges_cached, prev_e_ptr] : async_resources().get_cached_graphs())
    {
      // Skip the update early if needed
      if (nnodes_cached != nnodes || nedges_cached != nedges)
      {
        continue;
      }
      cudaGraphExec_t& prev_e = *prev_e_ptr;
      if (try_updating_executable_graph(prev_e, *g))
      {
        // Return the updated graph
        state.exec_graph = prev_e_ptr;
        return prev_e_ptr;
      }
    }

    if (getenv("CUDASTF_DUMP_GRAPHS"))
    {
      static int instantiated_graph = 0;
      print_to_dot("instantiated_graph" + ::std::to_string(instantiated_graph++) + ".dot");
    }

    state.exec_graph = graph_instantiate(*g);

    // Save this graph for later use
    async_resources().put_graph_in_cache(nnodes, nedges, state.exec_graph);

    return state.exec_graph;
  }

  void display_graph_info(cudaGraph_t g)
  {
    size_t numNodes;
    cuda_safe_call(cudaGraphGetNodes(g, nullptr, &numNodes));

    size_t numEdges;
    cuda_safe_call(cudaGraphGetEdges(g, nullptr, nullptr, &numEdges));

    cuuint64_t mem_attr;
    cuda_safe_call(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &mem_attr));

    // fprintf(stderr, "INSTANTIATING graph %p with %ld nodes %ld edges - MEM %ld\n", g, numNodes, numEdges,
    // mem_attr);
  }

  // flags = cudaGraphDebugDotFlagsVerbose to display all available debug information
  void print_to_dot(const ::std::string& filename, enum cudaGraphDebugDotFlags flags = cudaGraphDebugDotFlags(0))
  {
    cudaGraphDebugDotPrint(get_graph(), filename.c_str(), flags);
  }

private:
  impl& state()
  {
    return dynamic_cast<impl&>(get_state());
  };
  const impl& state() const
  {
    return dynamic_cast<const impl&>(get_state());
  };

  /// @brief Inserts an empty node that depends on all pending operations
  cudaStream_t task_fence_impl()
  {
    auto& state = this->state();

    auto& cs          = this->get_stack();
    auto prereq_fence = cs.insert_task_fence(*get_dot());

    const size_t graph_epoch             = state.graph_epoch;
    ::std::vector<cudaGraphNode_t> nodes = reserved::join_with_graph_nodes(prereq_fence, graph_epoch);

    // Create an empty graph node
    cudaGraphNode_t n;
    cuda_safe_call(cudaGraphAddEmptyNode(&n, get_graph(), nodes.data(), nodes.size()));

    reserved::fork_from_graph_node(*this, n, graph_epoch, prereq_fence, "fence");

    return nullptr; // for conformity with the stream version
  }

  // Instantiate a CUDA graph
  static ::std::shared_ptr<cudaGraphExec_t> graph_instantiate(cudaGraph_t g)
  {
    // Custom deleter specifically for cudaGraphExec_t
    auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
      cudaGraphExecDestroy(*pGraphExec);
    };

    ::std::shared_ptr<cudaGraphExec_t> res(new cudaGraphExec_t, cudaGraphExecDeleter);

    cuda_try(cudaGraphInstantiateWithFlags(res.get(), g, 0));

#ifdef CUDASTF_DEBUG
    reserved::counter<reserved::graph_tag::instantiate>.increment();
#endif

    return res;
  }

  // Creates a new CUDA graph and wrap it into a shared_ptr
  static ::std::shared_ptr<cudaGraph_t> shared_cuda_graph()
  {
    auto cudaGraphDeleter = [](cudaGraph_t* pGraph) {
      cudaGraphDestroy(*pGraph);
    };

    ::std::shared_ptr<cudaGraph_t> res(new cudaGraph_t, cudaGraphDeleter);

    cuda_try(cudaGraphCreate(res.get(), 0));

    return res;
  }

  // Wrap an existing CUDA graph into a shared_ptr, the destruction of the graph is let to the application
  static ::std::shared_ptr<cudaGraph_t> wrap_cuda_graph(cudaGraph_t g)
  {
    // Allocate memory for a new cudaGraph_t and copy the existing graph to it
    cudaGraph_t* pGraph = new cudaGraph_t;
    *pGraph             = g;

    // There is no custom deleter : only the pointer itself will be destroyed
    ::std::shared_ptr<cudaGraph_t> res(pGraph);

    return res;
  }

  cudaStream_t submit_one_epoch(cudaGraph_t g, size_t epoch)
  {
    auto& state = this->state();
    // TODO rework to make sure we don't push a specific stream later ?
    if (!state.submitted_stream)
    {
      // fprintf(stderr, "Initializing stream support for ctx ...\n");
      state.submitted_stream = pick_stream();
      assert(state.submitted_stream != nullptr);
    }
    //        cuda_safe_call(cudaStreamSynchronize(state.submitted_stream));

    size_t nedges;
    size_t nnodes;

    cuda_safe_call(cudaGraphGetNodes(g, nullptr, &nnodes));
    cuda_safe_call(cudaGraphGetEdges(g, nullptr, nullptr, &nedges));

    cudaGraphExec_t local_exec_graph = nullptr;

    // fprintf(stderr, "Instantiate epoch %ld\n", epoch);
    display_graph_info(g);

#if 0
        ::std::string name = "epoch" + ::std::to_string(epoch) + ".dot";
        cudaGraphDebugDotPrint(g, name.c_str(), cudaGraphDebugDotFlagsVerbose);
#endif

    bool found = false;

    // Try to reuse previous instantiated graphs
    for (auto& [prev_nnodes, prev_nedges, prev_e_ptr, last_epoch] : state.previous_exec_graphs)
    {
      // Early exit if the topology cannot match
      if (prev_nnodes != nnodes || prev_nedges != nedges)
      {
        continue;
      }
      cudaGraphExec_t& prev_e = *prev_e_ptr;
      if (try_updating_executable_graph(prev_e, g))
      {
        local_exec_graph = prev_e;

        // Update epoch in the vector of pairs
        last_epoch = epoch;

        found = true;
        break;
      }
    }

    if (!found)
    {
      // We need to instantiate a new exec_graph
      auto e_graph_ptr = graph_instantiate(g);

      // Save for future use
      state.previous_exec_graphs.push_back(::std::make_tuple(nnodes, nedges, e_graph_ptr, epoch));

      local_exec_graph = *e_graph_ptr;
    }

    cuda_try(cudaGraphLaunch(local_exec_graph, state.submitted_stream));

#ifdef CUDASTF_DEBUG
    reserved::counter<reserved::graph_tag::launch>.increment();
#endif

    return state.submitted_stream;
  }

public:
  // This tries to instantiate the graph by updating an existing executable graph
  // the returned value indicates whether the update was successful or not
  static bool try_updating_executable_graph(cudaGraphExec_t exec_graph, cudaGraph_t graph)
  {
#if CUDA_VERSION < 12000
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphExecUpdate(exec_graph, graph, &errorNode, &updateResult);
#else
    cudaGraphExecUpdateResultInfo resultInfo;
    cudaGraphExecUpdate(exec_graph, graph, &resultInfo);
#endif

    // Be sure to "erase" the last error
    cudaError_t res = cudaGetLastError();

#ifdef CUDASTF_DEBUG
    reserved::counter<reserved::graph_tag::update>.increment();
    if (res == cudaSuccess)
    {
      reserved::counter<reserved::graph_tag::update::success>.increment();
    }
    else
    {
      reserved::counter<reserved::graph_tag::update::failure>.increment();
    }
#endif

    return (res == cudaSuccess);
  }

  friend inline cudaGraph_t ctx_to_graph(backend_ctx_untyped& ctx)
  {
    return ctx.graph();
  }

  friend inline size_t ctx_to_graph_epoch(backend_ctx_untyped& ctx)
  {
    return ctx.epoch();
  }
};

#ifdef UNITTESTED_FILE

UNITTEST("movable graph_ctx")
{
  graph_ctx ctx;
  graph_ctx ctx2 = mv(ctx);
};

UNITTEST("copyable graph_task<>")
{
  graph_ctx ctx;
  graph_task<> t     = ctx.task();
  graph_task<> t_cpy = t;
};

UNITTEST("copyable graph_ctx")
{
  graph_ctx ctx;
  graph_ctx ctx2  = ctx;
  graph_task<> t  = ctx.task();
  graph_task<> t2 = ctx2.task();
};

UNITTEST("movable graph_task<>")
{
  graph_ctx ctx;
  graph_task<> t     = ctx.task();
  graph_task<> t_cpy = mv(t);
};

UNITTEST("set_symbol on graph_task and graph_task<>")
{
  graph_ctx ctx;

  double X[1024], Y[1024];
  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  graph_task<> t = ctx.task();
  t.add_deps(lX.rw(), lY.rw());
  t.set_symbol("graph_task<>");
  t.start();
  cudaGraphNode_t n;
  cuda_safe_call(cudaGraphAddEmptyNode(&n, t.get_graph(), nullptr, 0));
  t.end();

  graph_task<slice<double>, slice<double>> t2 = ctx.task(lX.rw(), lY.rw());
  t2.set_symbol("graph_task");
  t2.start();
  cudaGraphNode_t n2;
  cuda_safe_call(cudaGraphAddEmptyNode(&n2, t2.get_graph(), nullptr, 0));
  t2.end();

  ctx.finalize();
};

#  if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
namespace reserved
{

inline void unit_test_graph_epoch()
{
  graph_ctx ctx;

  const size_t N     = 8;
  const size_t NITER = 10;

  double A[N];
  for (size_t i = 0; i < N; i++)
  {
    A[i] = 1.0 * i;
  }

  auto lA = ctx.logical_data(A);

  for (size_t k = 0; k < NITER; k++)
  {
    ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.rw())
        ->*[] _CCCL_HOST_DEVICE(size_t i, slice<double> A) {
              A(i) = cos(A(i));
            };

    ctx.change_epoch();
  }

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    double Ai_ref = 1.0 * i;
    for (size_t k = 0; k < NITER; k++)
    {
      Ai_ref = cos(Ai_ref);
    }

    EXPECT(fabs(A[i] - Ai_ref) < 0.01);
  }
}

UNITTEST("graph with epoch")
{
  unit_test_graph_epoch();
};

inline void unit_test_graph_empty_epoch()
{
  graph_ctx ctx;

  const size_t N     = 8;
  const size_t NITER = 10;

  double A[N];
  for (size_t i = 0; i < N; i++)
  {
    A[i] = 1.0 * i;
  }

  auto lA = ctx.logical_data(A);

  for (size_t k = 0; k < NITER; k++)
  {
    ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.rw())
        ->*[] _CCCL_HOST_DEVICE(size_t i, slice<double> A) {
              A(i) = cos(A(i));
            };

    ctx.change_epoch();
    // Nothing in between
    ctx.change_epoch();
  }

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    double Ai_ref = 1.0 * i;
    for (size_t k = 0; k < NITER; k++)
    {
      Ai_ref = cos(Ai_ref);
    }

    EXPECT(fabs(A[i] - Ai_ref) < 0.01);
  }
}

UNITTEST("graph with empty epoch")
{
  unit_test_graph_empty_epoch();
};

inline void unit_test_graph_epoch_2()
{
  graph_ctx ctx;

  const size_t N     = 8;
  const size_t NITER = 10;

  double A[N];
  for (size_t i = 0; i < N; i++)
  {
    A[i] = 1.0 * i;
  }

  auto lA = ctx.logical_data(A);

  for (size_t k = 0; k < NITER; k++)
  {
    if ((k % 2) == 0)
    {
      ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.rw())
          ->*[] _CCCL_HOST_DEVICE(size_t i, slice<double> A) {
                A(i) = cos(A(i));
              };
    }
    else
    {
      ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.rw())
          ->*[] _CCCL_HOST_DEVICE(size_t i, slice<double> A) {
                A(i) = sin(A(i));
              };
    }

    ctx.change_epoch();
  }

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    double Ai_ref = 1.0 * i;
    for (size_t k = 0; k < NITER; k++)
    {
      Ai_ref = ((k % 2) == 0) ? cos(Ai_ref) : sin(Ai_ref);
    }

    EXPECT(fabs(A[i] - Ai_ref) < 0.01);
  }
}

UNITTEST("graph with epoch 2")
{
  unit_test_graph_epoch_2();
};

inline void unit_test_graph_epoch_3()
{
  graph_ctx ctx;

  const size_t N     = 8;
  const size_t NITER = 10;

  double A[N];
  double B[N];
  for (size_t i = 0; i < N; i++)
  {
    A[i] = 1.0 * i;
    B[i] = -1.0 * i;
  }

  auto lA = ctx.logical_data(A);
  auto lB = ctx.logical_data(B);

  for (size_t k = 0; k < NITER; k++)
  {
    if ((k % 2) == 0)
    {
      ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.rw(), lB.read())
          ->*[] _CCCL_HOST_DEVICE(size_t i, slice<double> A, slice<double> B) {
                A(i) = cos(B(i));
              };
    }
    else
    {
      ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.read(), lB.rw())
          ->*[] _CCCL_HOST_DEVICE(size_t i, slice<double> A, slice<double> B) {
                B(i) = sin(A(i));
              };
    }

    ctx.change_epoch();
  }

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    double Ai_ref = 1.0 * i;
    double Bi_ref = -1.0 * i;
    for (size_t k = 0; k < NITER; k++)
    {
      if ((k % 2) == 0)
      {
        Ai_ref = cos(Bi_ref);
      }
      else
      {
        Bi_ref = sin(Ai_ref);
      }
    }

    EXPECT(fabs(A[i] - Ai_ref) < 0.01);
    EXPECT(fabs(B[i] - Bi_ref) < 0.01);
  }
}

UNITTEST("graph with epoch 3")
{
  unit_test_graph_epoch_3();
};

inline void unit_test_launch_graph()
{
  graph_ctx ctx;
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

UNITTEST("basic launch test (graph_ctx)")
{
  unit_test_launch_graph();
};

inline void unit_test_launch_many_graph_ctx()
{
  // Stress the allocators and all ressources !
  for (size_t i = 0; i < 256; i++)
  {
    graph_ctx ctx;
    auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
    ctx.launch(lA.write())->*[] _CCCL_DEVICE(auto t, slice<size_t> A) {
      for (auto i : t.apply_partition(shape(A)))
      {
        A(i) = 2 * i;
      }
    };
    ctx.finalize();
  }
}

UNITTEST("create many graph ctxs")
{
  unit_test_launch_many_graph_ctx();
};

} // end namespace reserved

#  endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf
