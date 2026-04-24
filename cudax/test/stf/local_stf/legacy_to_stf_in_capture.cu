//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Exercise building an STF context on a caller-provided stream that is
 *        currently in CUDA graph capture mode.
 *
 * The flow is:
 *
 *   1. Create an explicit CUDA stream.
 *   2. Start a capture on it with ``cudaStreamCaptureModeRelaxed``.
 *   3. Construct a ``stream_ctx`` bound to that captured stream and submit a
 *      small diamond DAG of tasks:
 *
 *          ctx.task(lA.write())          // initA on one pool stream
 *          ctx.task(lB.write())          // initB on another pool stream
 *          ctx.task(lA.read(), lB.rw())  // axpy joining the two branches
 *          ctx.task()                    // empty epilogue
 *
 *   4. ``ctx.finalize()`` and ``cudaStreamEndCapture`` produce a CUDA graph.
 *   5. The graph is instantiated, launched several times on a clean replay
 *      stream, and the resulting arrays are validated on the host.
 *
 * The captured graph is also walked with the CUDA runtime API to confirm the
 * two ``init`` kernel nodes are mutually independent, i.e. STF's fork/join
 * through its internal stream pool actually produced multi-stream concurrency
 * rather than a serialized single-stream chain.
 *
 * A final negative case verifies that constructing ``stream_ctx(user_stream,
 * handle)`` with a caller-provided ``async_resources_handle`` while
 * ``user_stream`` is capturing is rejected early with a precise diagnostic.
 */

#include <cuda/experimental/stf.cuh>

#include <cmath>
#include <unordered_set>
#include <vector>

using namespace cuda::experimental::stf;

__global__ void initA(double* d_ptrA, size_t N)
{
  size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < N; i += nthreads)
  {
    d_ptrA[i] = sin((double) i);
  }
}

__global__ void initB(double* d_ptrB, size_t N)
{
  size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < N; i += nthreads)
  {
    d_ptrB[i] = cos((double) i);
  }
}

// B += alpha * A
__global__ void axpy(double alpha, const double* d_ptrA, double* d_ptrB, size_t N)
{
  size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < N; i += nthreads)
  {
    d_ptrB[i] += alpha * d_ptrA[i];
  }
}

__global__ void empty_kernel()
{
  // no-op: acts as the "epilogue" task in the diamond DAG.
}

/**
 * @brief Control path: same diamond DAG, built with plain CUDA API calls --
 *        two side streams forked from the captured main stream via events,
 *        then joined back on the main stream.
 *
 * Establishes a baseline that the diamond pattern itself (fork + join via
 * events, with non-blocking side streams) is legal inside a Relaxed-mode
 * capture, independently of STF.
 */
void submit_diamond_manual(cudaStream_t main, double* d_ptrA, double* d_ptrB, size_t N)
{
  // Fresh side streams created inside the capture so they have no prior
  // (uncaptured) work of their own. Use the non-blocking flag to mirror how
  // STF creates its own pool streams.
  cudaStream_t sA;
  cudaStream_t sB;
  cuda_safe_call(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
  cuda_safe_call(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));

  // Start event on the captured main stream.
  cudaEvent_t e_start;
  cuda_safe_call(cudaEventCreateWithFlags(&e_start, cudaEventDisableTiming));
  cuda_safe_call(cudaEventRecord(e_start, main));

  // Fork: bring sA and sB into the capture.
  cuda_safe_call(cudaStreamWaitEvent(sA, e_start, 0));
  cuda_safe_call(cudaStreamWaitEvent(sB, e_start, 0));

  // Run the two independent init branches on sA / sB.
  initA<<<128, 32, 0, sA>>>(d_ptrA, N);
  initB<<<128, 32, 0, sB>>>(d_ptrB, N);

  // Join both side streams back on ``main`` for the axpy combine step.
  cudaEvent_t e_A;
  cudaEvent_t e_B;
  cuda_safe_call(cudaEventCreateWithFlags(&e_A, cudaEventDisableTiming));
  cuda_safe_call(cudaEventCreateWithFlags(&e_B, cudaEventDisableTiming));
  cuda_safe_call(cudaEventRecord(e_A, sA));
  cuda_safe_call(cudaEventRecord(e_B, sB));
  cuda_safe_call(cudaStreamWaitEvent(main, e_A, 0));
  cuda_safe_call(cudaStreamWaitEvent(main, e_B, 0));

  axpy<<<128, 32, 0, main>>>(3.0, d_ptrA, d_ptrB, N);
  empty_kernel<<<16, 8, 0, main>>>();

  // Destroy events (safe at any point -- cleanup).
  cuda_safe_call(cudaEventDestroy(e_start));
  cuda_safe_call(cudaEventDestroy(e_A));
  cuda_safe_call(cudaEventDestroy(e_B));
  // Side-stream destruction is deferred: they are now part of the capture
  // and must outlive it.
  cuda_safe_call(cudaStreamDestroy(sA));
  cuda_safe_call(cudaStreamDestroy(sB));
}

/**
 * @brief Submit the token-based diamond DAG inside an already-started capture.
 *
 * Must be called while ``stream`` is in ``StreamCaptureStatusActive``.
 */
void submit_diamond_token(cudaStream_t stream, double* d_ptrA, double* d_ptrB, size_t N)
{
  stream_ctx ctx(stream);

  auto lA = ctx.token();
  auto lB = ctx.token();

  ctx.task(lA.write())->*[=](cudaStream_t s) {
    initA<<<128, 32, 0, s>>>(d_ptrA, N);
  };

  ctx.task(lB.write())->*[=](cudaStream_t s) {
    initB<<<128, 32, 0, s>>>(d_ptrB, N);
  };

  ctx.task(lA.read(), lB.rw())->*[=](cudaStream_t s) {
    axpy<<<128, 32, 0, s>>>(3.0, d_ptrA, d_ptrB, N);
  };

  ctx.task()->*[](cudaStream_t s) {
    empty_kernel<<<16, 8, 0, s>>>();
  };

  // ``finalize()`` is non-blocking when the context is bound to a
  // user-provided stream: all work is now enqueued on ``stream`` (or on
  // pool streams that have been folded into ``stream``'s capture).
  ctx.finalize();
}

/**
 * @brief Compute the transitive-dependency closure of ``root`` in a CUDA graph.
 *
 * Walks predecessors via ``cudaGraphNodeGetDependencies`` and returns every
 * node reachable from ``root`` (excluding ``root`` itself). Used to assert
 * that two kernel nodes are mutually independent in the captured graph.
 */
static std::unordered_set<cudaGraphNode_t> transitive_dependencies(cudaGraphNode_t root)
{
  std::unordered_set<cudaGraphNode_t> visited;
  std::vector<cudaGraphNode_t> stack;
  stack.push_back(root);
  while (!stack.empty())
  {
    cudaGraphNode_t n = stack.back();
    stack.pop_back();
    size_t ndeps = 0;
    cuda_safe_call(cudaGraphNodeGetDependencies(n, nullptr, nullptr, &ndeps));
    if (ndeps == 0)
    {
      continue;
    }
    std::vector<cudaGraphNode_t> deps(ndeps);
    cuda_safe_call(cudaGraphNodeGetDependencies(n, deps.data(), nullptr, &ndeps));
    for (cudaGraphNode_t d : deps)
    {
      if (visited.insert(d).second)
      {
        stack.push_back(d);
      }
    }
  }
  visited.erase(root);
  return visited;
}

/**
 * @brief Walk the captured graph and assert that the ``initA``/``initB``
 *        branches are mutually independent (i.e. really parallel, not
 *        serialized by a hidden edge STF inserted).
 *
 * We identify the three kernel nodes by launch-dimension signature: the two
 * ``init`` kernels and the ``axpy`` kernel all use ``<<<128, 32>>>`` while the
 * epilogue uses ``<<<16, 8>>>``. The combiner (``axpy``) is the kernel node
 * that transitively depends on both of the others; the remaining two are the
 * independent ``init`` branches.
 */
static void assert_inits_are_parallel(cudaGraph_t graph)
{
  size_t nnodes = 0;
  cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &nnodes));
  std::vector<cudaGraphNode_t> nodes(nnodes);
  cuda_safe_call(cudaGraphGetNodes(graph, nodes.data(), &nnodes));

  // Collect kernel nodes whose grid/block signature matches the three
  // diamond kernels (initA / initB / axpy all launch <<<128, 32>>>).
  std::vector<cudaGraphNode_t> diamond_kernels;
  for (cudaGraphNode_t n : nodes)
  {
    cudaGraphNodeType t;
    cuda_safe_call(cudaGraphNodeGetType(n, &t));
    if (t != cudaGraphNodeTypeKernel)
    {
      continue;
    }
    cudaKernelNodeParams p = {};
    cuda_safe_call(cudaGraphKernelNodeGetParams(n, &p));
    if (p.gridDim.x == 128 && p.blockDim.x == 32)
    {
      diamond_kernels.push_back(n);
    }
  }

  EXPECT(diamond_kernels.size() == 3,
         "Expected exactly 3 diamond-shape kernel nodes (initA, initB, axpy), got ",
         diamond_kernels.size());

  // Find the combiner: the single kernel whose transitive predecessors
  // contain the other two diamond kernels.
  int combiner_idx = -1;
  for (int i = 0; i < 3; ++i)
  {
    auto deps = transitive_dependencies(diamond_kernels[i]);
    int hits  = 0;
    for (int j = 0; j < 3; ++j)
    {
      if (j != i && deps.count(diamond_kernels[j]) > 0)
      {
        ++hits;
      }
    }
    if (hits == 2)
    {
      EXPECT(combiner_idx == -1, "More than one combiner kernel found; DAG is not a diamond");
      combiner_idx = i;
    }
  }
  EXPECT(combiner_idx != -1, "No combiner kernel found; init A and init B must both flow into axpy");

  // The two remaining kernels are the init branches. They must be mutually
  // independent -- neither reachable from the other via predecessor edges.
  std::vector<cudaGraphNode_t> inits;
  for (int i = 0; i < 3; ++i)
  {
    if (i != combiner_idx)
    {
      inits.push_back(diamond_kernels[i]);
    }
  }
  EXPECT(inits.size() == 2);

  auto deps_a = transitive_dependencies(inits[0]);
  auto deps_b = transitive_dependencies(inits[1]);

  EXPECT(deps_a.count(inits[1]) == 0,
         "init branch A transitively depends on init branch B -- STF serialized the diamond");
  EXPECT(deps_b.count(inits[0]) == 0,
         "init branch B transitively depends on init branch A -- STF serialized the diamond");
}

/**
 * @brief Run ``submit``, under Relaxed-mode capture on a caller-owned stream.
 *        Harvest the captured graph, replay it ``NREPLAYS`` times on a clean
 *        replay stream, and validate the resulting arrays on the host.
 */
template <typename Submit>
void run_diamond_under_capture(const char* label, Submit&& submit)
{
  const size_t N = 128 * 1024;

  double* d_ptrA = nullptr;
  double* d_ptrB = nullptr;
  cuda_safe_call(cudaMalloc(&d_ptrA, N * sizeof(double)));
  cuda_safe_call(cudaMalloc(&d_ptrB, N * sizeof(double)));

  // Zero the buffers so a failure to run the captured graph would produce a
  // host-side mismatch rather than coincidentally-correct data.
  cuda_safe_call(cudaMemset(d_ptrA, 0, N * sizeof(double)));
  cuda_safe_call(cudaMemset(d_ptrB, 0, N * sizeof(double)));

  // Caller-owned stream, started in Relaxed capture mode.
  cudaStream_t capture_stream;
  cuda_safe_call(cudaStreamCreate(&capture_stream));
  cuda_safe_call(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeRelaxed));

  submit(capture_stream, d_ptrA, d_ptrB, N);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(capture_stream, &graph));
  EXPECT(graph != nullptr, "cudaStreamEndCapture returned a null graph");

  // Captured DAG check: initA and initB must be mutually independent.
  assert_inits_are_parallel(graph);

  // Instantiate and replay the captured graph several times on a clean replay
  // stream.
  cudaGraphExec_t graph_exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  cudaStream_t replay_stream;
  cuda_safe_call(cudaStreamCreate(&replay_stream));
  const int NREPLAYS = 4;
  for (int r = 0; r < NREPLAYS; ++r)
  {
    cuda_safe_call(cudaGraphLaunch(graph_exec, replay_stream));
  }
  cuda_safe_call(cudaStreamSynchronize(replay_stream));

  // Validate results on the host.
  //   A[i] = sin(i)
  //   B[i] = cos(i) + 3*sin(i)
  std::vector<double> h_A(N, 0.0);
  std::vector<double> h_B(N, 0.0);
  cuda_safe_call(cudaMemcpy(h_A.data(), d_ptrA, N * sizeof(double), cudaMemcpyDeviceToHost));
  cuda_safe_call(cudaMemcpy(h_B.data(), d_ptrB, N * sizeof(double), cudaMemcpyDeviceToHost));

  double max_err_A = 0.0;
  double max_err_B = 0.0;
  for (size_t i = 0; i < N; ++i)
  {
    double ref_A = sin((double) i);
    double ref_B = cos((double) i) + 3.0 * sin((double) i);
    max_err_A    = std::fmax(max_err_A, std::fabs(h_A[i] - ref_A));
    max_err_B    = std::fmax(max_err_B, std::fabs(h_B[i] - ref_B));
  }
  EXPECT(max_err_A < 1e-10, "[", label, "] A mismatch: max|A - sin(i)| = ", max_err_A);
  EXPECT(max_err_B < 1e-10, "[", label, "] B mismatch: max|B - (cos(i) + 3 sin(i))| = ", max_err_B);

  cuda_safe_call(cudaGraphExecDestroy(graph_exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(replay_stream));
  cuda_safe_call(cudaStreamDestroy(capture_stream));
  cuda_safe_call(cudaFree(d_ptrA));
  cuda_safe_call(cudaFree(d_ptrB));
}

int main()
{
  // Control path: plain CUDA API diamond inside a Relaxed-mode capture.
  run_diamond_under_capture("manual", submit_diamond_manual);

  // STF path: token-based diamond submitted through ``stream_ctx(user_stream)``.
  // The context is constructed without an explicit ``async_resources_handle``
  // so it gets a fresh, empty stream pool that STF is free to fold into the
  // on-going capture. This is the supported in-capture configuration.
  run_diamond_under_capture("stf_token", submit_diamond_token);

  // Negative case: ``stream_ctx(user_stream, handle)`` with a user-provided
  // handle while ``user_stream`` is capturing must be rejected early, before
  // any pool streams are touched.
  {
    cudaStream_t capture_stream = nullptr;
    cuda_safe_call(cudaStreamCreate(&capture_stream));
    cuda_safe_call(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeRelaxed));

    bool threw = false;
    try
    {
      async_resources_handle h; // user-provided handle (non-null)
      stream_ctx ctx(capture_stream, mv(h));
    }
    catch (const ::std::exception&)
    {
      threw = true;
    }
    EXPECT(threw,
           "stream_ctx(user_stream, handle) should have aborted because "
           "user_stream is in a capture and a user-provided handle was passed.");

    // Discard the capture we started -- if we reached here, no work was
    // actually enqueued on ``capture_stream`` after the failed construction.
    cudaGraph_t unused = nullptr;
    cuda_safe_call(cudaStreamEndCapture(capture_stream, &unused));
    if (unused)
    {
      cuda_safe_call(cudaGraphDestroy(unused));
    }
    cuda_safe_call(cudaStreamDestroy(capture_stream));
  }

  return 0;
}
