/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This file demonstrates the usage of conditional graph nodes with
 * a series of *simple* example graphs.
 *
 * For more information on conditional nodes, see the programming guide:
 *
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-graph-nodes
 *
 */

#include <cuda/experimental/stf.cuh>

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)

// This kernel will only be executed if the condition is true
__global__ void doWhileEmptyKernel(void)
{
  printf("GPU: doWhileEmptyKernel()\n");
  return;
}

__global__ void doWhileLoopKernel(char* dPtr, cudaGraphConditionalHandle handle)
{
  if (--(*dPtr) == 0)
  {
    cudaGraphSetConditional(handle, 0);
  }
  printf("GPU: counter = %d\n", *dPtr);
}

void simpleDoWhileGraph(void)
{
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  cudaGraphNode_t conditionalNode;

  // Allocate a byte of device memory to use as input
  char* dPtr;
  cuda_safe_call(cudaMalloc((void**) &dPtr, 1));

  printf("simpleDoWhileGraph: Building graph...\n");
  cuda_safe_call(cudaGraphCreate(&graph, 0));

  cudaGraphConditionalHandle handle;
  cuda_safe_call(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));

  cudaGraphNodeParams cParams{};
  cParams.type               = cudaGraphNodeTypeConditional;
  cParams.conditional.handle = handle;
  cParams.conditional.type   = cudaGraphCondTypeWhile;
  cParams.conditional.size   = 1;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cuda_safe_call(cudaGraphAddNode(&conditionalNode, graph, NULL, NULL, 0, &cParams));
#  else
  cuda_safe_call(cudaGraphAddNode(&conditionalNode, graph, NULL, 0, &cParams));
#  endif

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  cudaStream_t captureStream;
  cuda_safe_call(cudaStreamCreate(&captureStream));

  cuda_safe_call(
    cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
  doWhileEmptyKernel<<<1, 1, 0, captureStream>>>();
  doWhileEmptyKernel<<<1, 1, 0, captureStream>>>();
  doWhileLoopKernel<<<1, 1, 0, captureStream>>>(dPtr, handle);
  cuda_safe_call(cudaStreamEndCapture(captureStream, nullptr));
  cuda_safe_call(cudaStreamDestroy(captureStream));

  cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  // Initialize device memory and launch the graph
  cuda_safe_call(cudaMemset(dPtr, 10, 1)); // Set dPtr to 10
  printf("Host: Launching graph with loop counter set to 10\n");
  cuda_safe_call(cudaGraphLaunch(graphExec, 0));
  cuda_safe_call(cudaDeviceSynchronize());

  // Cleanup
  cuda_safe_call(cudaGraphExecDestroy(graphExec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaFree(dPtr));

  printf("simpleDoWhileGraph: Complete\n\n");
}

void stf_dowhile()
{
  stackable_ctx ctx;

  {
    auto repeat_guard = ctx.repeat_graph_scope(10);

    ctx.task()->*[](cudaStream_t stream) {
      doWhileEmptyKernel<<<1, 1, 0, stream>>>();
      doWhileEmptyKernel<<<1, 1, 0, stream>>>();
    };
  }

  ctx.finalize();
  printf("STF do while complete\n\n");
}

/*
 * Create a graph containing a conditional while loop using stream capture.
 * This demonstrates how to insert a conditional node into a stream which is
 * being captured. The graph consists of a kernel node, A, followed by a
 * conditional while node, B, followed by a kernel node, D. The conditional
 * body is populated by a single kernel node, C:
 *
 * A -> B [ C ] -> D
 *
 * The same kernel will be used for both nodes A and C. This kernel will test
 * a device memory location and set the condition when the location is non-zero.
 * We must run the kernel before the loop as well as inside the loop in order
 * to behave like a while loop as opposed to a do-while loop. We need to evaluate
 * the device memory location before the conditional node is evaluated in order
 * to set the condition variable properly. Because we're using a kernel upstream
 * of the conditional node, there is no need to use the handle default value to
 * initialize the conditional value.
 */

__global__ void capturedWhileKernel(char* dPtr, cudaGraphConditionalHandle handle)
{
  printf("GPU: counter = %d\n", *dPtr);
  if (*dPtr)
  {
    (*dPtr)--;
  }
  cudaGraphSetConditional(handle, *dPtr);
}

__global__ void capturedWhileEmptyKernel(void)
{
  printf("GPU: capturedWhileEmptyKernel()\n");
  return;
}

void capturedWhileGraph(void)
{
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  cudaStreamCaptureStatus status;
  const cudaGraphNode_t* dependencies;
  size_t numDependencies;

  // Allocate a byte of device memory to use as input
  char* dPtr;
  cuda_safe_call(cudaMalloc((void**) &dPtr, 1));

  printf("capturedWhileGraph: Building graph...\n");
  cudaStream_t captureStream;
  cuda_safe_call(cudaStreamCreate(&captureStream));

  cuda_safe_call(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

  // Obtain the handle of the graph
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cuda_safe_call(cudaStreamGetCaptureInfo(captureStream, &status, NULL, &graph, &dependencies, NULL, &numDependencies));
#  else
  cuda_safe_call(cudaStreamGetCaptureInfo(captureStream, &status, NULL, &graph, &dependencies, &numDependencies));
#  endif

  // Create the conditional handle
  cudaGraphConditionalHandle handle;
  cuda_safe_call(cudaGraphConditionalHandleCreate(&handle, graph));

  // Insert kernel node A
  capturedWhileKernel<<<1, 1, 0, captureStream>>>(dPtr, handle);

  // Obtain the handle for node A
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cuda_safe_call(cudaStreamGetCaptureInfo(captureStream, &status, NULL, &graph, &dependencies, NULL, &numDependencies));
#  else
  cuda_safe_call(cudaStreamGetCaptureInfo(captureStream, &status, NULL, &graph, &dependencies, &numDependencies));
#  endif

  // Insert conditional node B
  cudaGraphNode_t conditionalNode;
  cudaGraphNodeParams cParams{};
  cParams.type               = cudaGraphNodeTypeConditional;
  cParams.conditional.handle = handle;
  cParams.conditional.type   = cudaGraphCondTypeWhile;
  cParams.conditional.size   = 1;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cuda_safe_call(cudaGraphAddNode(&conditionalNode, graph, dependencies, NULL, numDependencies, &cParams));
#  else
  cuda_safe_call(cudaGraphAddNode(&conditionalNode, graph, dependencies, numDependencies, &cParams));
#  endif

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  // Update stream capture dependencies to account for the node we manually added
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cuda_safe_call(
    cudaStreamUpdateCaptureDependencies(captureStream, &conditionalNode, NULL, 1, cudaStreamSetCaptureDependencies));
#  else
  cuda_safe_call(
    cudaStreamUpdateCaptureDependencies(captureStream, &conditionalNode, 1, cudaStreamSetCaptureDependencies));
#  endif

  // Insert kernel node D
  capturedWhileEmptyKernel<<<1, 1, 0, captureStream>>>();

  cuda_safe_call(cudaStreamEndCapture(captureStream, &graph));
  cuda_safe_call(cudaStreamDestroy(captureStream));

  // Populate conditional body graph using stream capture
  cudaStream_t bodyStream;
  cuda_safe_call(cudaStreamCreate(&bodyStream));

  cuda_safe_call(
    cudaStreamBeginCaptureToGraph(bodyStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));

  // Insert kernel node C
  capturedWhileKernel<<<1, 1, 0, bodyStream>>>(dPtr, handle);
  cuda_safe_call(cudaStreamEndCapture(bodyStream, nullptr));
  cuda_safe_call(cudaStreamDestroy(bodyStream));

  cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  // Initialize device memory and launch the graph
  // Device memory is zero, so the conditional node will not execute
  cuda_safe_call(cudaMemset(dPtr, 0, 1)); // Set dPtr to 0
  printf("Host: Launching graph with loop counter set to 0\n");
  cuda_safe_call(cudaGraphLaunch(graphExec, 0));
  cuda_safe_call(cudaDeviceSynchronize());

  // Initialize device memory and launch the graph
  cuda_safe_call(cudaMemset(dPtr, 10, 1)); // Set dPtr to 10
  printf("Host: Launching graph with loop counter set to 10\n");
  cuda_safe_call(cudaGraphLaunch(graphExec, 0));
  cuda_safe_call(cudaDeviceSynchronize());

  // Cleanup
  cuda_safe_call(cudaGraphExecDestroy(graphExec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaFree(dPtr));

  printf("capturedWhileGraph: Complete\n\n");
}

void stf_dowhile_2()
{
  stackable_ctx ctx;

  {
    // We force everything to be a CUDA graph
    auto scope = ctx.graph_scope();

    // We use a token to ensure that A, B(C) and D are serialized
    auto t = ctx.token();

    // A
    ctx.task(t.rw())->*[](cudaStream_t stream) {
      doWhileEmptyKernel<<<1, 1, 0, stream>>>();
    };

    // B
    {
      auto repeat_guard = ctx.repeat_graph_scope(10);

      // C
      ctx.task(t.rw())->*[](cudaStream_t stream) {
        doWhileEmptyKernel<<<1, 1, 0, stream>>>();
      };
    }

    // D
    ctx.task(t.rw())->*[](cudaStream_t stream) {
      doWhileEmptyKernel<<<1, 1, 0, stream>>>();
    };
  }

  ctx.finalize();
}

void stf_dowhile_2_cuda_kernel()
{
  stackable_ctx ctx;

  {
    // We force everything to be a CUDA graph
    auto scope = ctx.graph_scope();

    // We use a token to ensure that A, B(C) and D are serialized
    auto t = ctx.token();

    // A
    ctx.cuda_kernel(t.rw())->*[]() {
      return cuda_kernel_desc{doWhileEmptyKernel, 1, 1, 0};
    };

    // B
    {
      auto repeat_guard = ctx.repeat_graph_scope(10);

      // C
      ctx.cuda_kernel(t.rw())->*[]() {
        return cuda_kernel_desc{doWhileEmptyKernel, 1, 1, 0};
      };
    }

    // D
    ctx.cuda_kernel(t.rw())->*[]() {
      return cuda_kernel_desc{doWhileEmptyKernel, 1, 1, 0};
    };
  }

  ctx.finalize();
}

#endif // _CCCL_CTK_AT_LEAST(12, 4)

int main(int, char**)
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  simpleDoWhileGraph();
  stf_dowhile();

  capturedWhileGraph();
  stf_dowhile_2();

  // same as stf_dowhile_2 but uses cuda_kernel
  stf_dowhile_2_cuda_kernel();

  return 0;
#endif // _CCCL_CTK_AT_LEAST(12, 4)
}
