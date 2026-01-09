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
 * @brief An experimental mechanism to launch callback on specific CPU threads
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

#include <cuda/experimental/__stf/utility/traits.cuh>

#include <cstdio>
#include <stack>

#ifndef _CCCL_DOXYGEN_INVOKED // do not document

#  if !_CCCL_COMPILER(MSVC)
#    define STATEFUL_CALLBACKS

namespace cuda::experimental::stf
{
class cb;

#    ifdef STATEFUL_CALLBACKS
class cudaCallbackStateCtxKeys : public reserved::meyers_singleton<cudaCallbackStateCtxKeys>
{
protected:
  cudaCallbackStateCtxKeys()  = default;
  ~cudaCallbackStateCtxKeys() = default;

public:
  // per thread current callback
  pthread_key_t cb_key;

  // Initialize cb_key once only
  pthread_once_t cb_key_once;
};

class cudaCallbackStateCtx : public reserved::meyers_singleton<cudaCallbackStateCtx>
{
protected:
  cudaCallbackStateCtx()
  {
    // Create a pthread_key to store the current callback
    // This is done only once for all, so that all pthread share the key
    pthread_once(&cudaCallbackStateCtxKeys::instance().cb_key_once, &cb_key_init);
  }

  ~cudaCallbackStateCtx()
  {
    pthread_key_delete(cudaCallbackStateCtxKeys::instance().cb_key_once);
  }

public:
  void set_current_cb(cb* cb)
  {
    pthread_setspecific(cudaCallbackStateCtxKeys::instance().cb_key, cb);
  }

  cb* get_current_cb()
  {
    return ((cb*) pthread_getspecific(cudaCallbackStateCtxKeys::instance().cb_key));
  }

private:
  static void cb_key_init()
  {
    // We cannot pass an argument in this function with the
    // pthread_once interface, so we retrieve the static instance
    // instead.
    cudaCallbackStateCtxKeys* ctx = &cudaCallbackStateCtxKeys::instance();
    pthread_key_create(&ctx->cb_key, NULL);
    pthread_setspecific(ctx->cb_key, NULL);
  }
};
#    endif // STATEFUL_CALLBACKS

class callback_queue;

bool cudaCallbackQueueProgress(callback_queue* q, bool flag);
cudaError_t cudaStreamAddCallbackWithQueue(
  cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags, class callback_queue* q);
int* cf_pop(callback_queue* q);

#    define USE_COMPLETION_FLAG_POOL 1

#    ifdef USE_COMPLETION_FLAG_POOL
class completion_flag_pool
{
public:
  void init()
  {
    cnt = 1024;
    cudaMallocManaged((void**) &pool, cnt * sizeof(int));
    memset(pool, 0, cnt * sizeof(int));

    int ii;
    for (ii = 0; ii < cnt; ii++)
    {
      stack.push(&pool[ii]);
    }
  }

  // TODO destructor
  int* pop()
  {
    if (stack.empty())
    {
      return NULL;
    }

    int* ptr = stack.top();
    stack.pop();

    return ptr;
  }

  void push(int* flag_ptr)
  {
    stack.push(flag_ptr);
  }

private:
  int cnt;
  int* pool;
  ::std::stack<int*> stack; // a stack to store available entries in the pool

  // A mutex to protect the pool of completion flags
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
};
#    endif

class cb
{
public:
  // Graph Host nodes slightly differ as they don't have a status
  bool is_graph_host_node;
  cudaHostFn_t graph_callback;
  cudaGraph_t graph;

  // For streams
  cudaError_t status;
  cudaStream_t stream;
  cudaStreamCallback_t callback;

  void* userData;

  class callback_queue* queue;

#    ifdef STATEFUL_CALLBACKS
  // To deal with restartable callbacks
  int step          = 0;
  void* private_ptr = NULL;
#    endif

  // This serves as a spinlock to unlock the stream after the execution
  // of the callback.
  // TODO: we could have a pool of suck spinlocks or an array (per stream ??)?
  int* completion_flag;

  /* Callback from cudaGraph */
  cb(cudaHostFn_t _graph_callback, void* _userData, class callback_queue* _queue)
  {
    is_graph_host_node = true;
    graph_callback     = _graph_callback;
    callback           = NULL;
    userData           = _userData;
    queue              = _queue;

#    ifdef STATEFUL_CALLBACKS
    step        = 0;
    private_ptr = NULL;
#    endif

#    ifdef USE_COMPLETION_FLAG_POOL
    completion_flag = cf_pop(queue);
#    else
    cudaMallocManaged((void**) &completion_flag, sizeof(int));
#    endif
  }

  /* Callback from CUDA streams */
  cb(cudaStream_t _stream, cudaStreamCallback_t _callback, void* _userData, class callback_queue* _queue)
  {
    is_graph_host_node = false;
    stream             = _stream;
    userData           = _userData;
    callback           = _callback;
    graph_callback     = NULL;
    queue              = _queue;

#    ifdef STATEFUL_CALLBACKS
    step        = 0;
    private_ptr = NULL;
#    endif

#    ifdef USE_COMPLETION_FLAG_POOL
    completion_flag = cf_pop(queue);
#    else
    cudaMallocManaged((void**) &completion_flag, sizeof(int));
#    endif
  }
};

class callback_queue : public reserved::meyers_singleton<callback_queue>
{
protected:
  callback_queue()
  {
    init();
    launch_worker();
  }

  ~callback_queue() = default;

public:
  ::std::deque<cb*> dq;

  // 0 = running
  // 1 = destroyed
  int status = 0;

  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t cond   = PTHREAD_COND_INITIALIZER;

  void init()
  {
    // Create a pool of completion flags
    cfp.init();
  }

  void async_destroy(cudaStream_t stream)
  {
    cudaStreamAddCallbackWithQueue(stream, NULL, NULL, 0, this);
  }

  void launch_worker()
  {
    // Avoid C++ whining ...
    typedef void* (*func_ptr)(void*);
    pthread_create(&progress_thread, NULL, (func_ptr) &callback_queue::callback_queue_worker, this);
  }

  void wait_worker()
  {
    pthread_join(progress_thread, NULL);
  }

  /* Helper routines to create a thread dedicated to this queue ! */
  pthread_t progress_thread;

#    ifdef USE_COMPLETION_FLAG_POOL
  class completion_flag_pool cfp;
#    endif

  static void* callback_queue_worker(void* args)
  {
    fprintf(stderr, "CALLBACK RUNNING...\n");
    class callback_queue* cbq = (callback_queue*) args;
    cudaCallbackQueueProgress(cbq, 1);
    fprintf(stderr, "CALLBACK HALTING...\n");

    return NULL;
  }
};

inline callback_queue* default_callback_queue()
{
  class callback_queue* default_cb = &callback_queue::instance();
  return default_cb;
}

inline int* cf_pop(callback_queue* q)
{
  return q->cfp.pop();
}

inline void callback_dispatcher(cudaStream_t, cudaError_t, void* userData)
{
  class cb* cb_               = (cb*) userData;
  class callback_queue* queue = cb_->queue;

  // Protect the queue
  pthread_mutex_lock(&queue->mutex);
  queue->dq.push_back(cb_);
  pthread_cond_broadcast(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
}

inline void cudagraph_callback_dispatcher(void* userData)
{
  cb* cb_ = (cb*) userData;

  class callback_queue* queue = cb_->queue;

  // Protect the queue
  pthread_mutex_lock(&queue->mutex);
  queue->dq.push_back(cb_);
  pthread_cond_broadcast(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
}

// There is likely a more efficient way in the current implementation of callbacks !
template <typename = void>
__global__ void callback_completion_kernel(int* completion_flag)
{
  // Loop until *completion_flag == 1
  while (1 != (atomicCAS(completion_flag, 1, 1)))
  {
  }

  //    printf("notified %p\n", completion_flag);
}

#    ifdef STATEFUL_CALLBACKS
inline void set_current_cb(cb* cb)
{
  cudaCallbackStateCtx::instance().set_current_cb(cb);
}

inline class cb* get_current_cb()
{
  return cudaCallbackStateCtx::instance().get_current_cb();
}

inline cudaError_t cudaCallbackSetStatus(int step, void* private_ptr)
{
  class cb* current_cb = get_current_cb();
  assert(current_cb);
  current_cb->step        = step;
  current_cb->private_ptr = private_ptr;
  return cudaSuccess;
}

inline cudaError_t cudaCallbackGetStatus(int* step, void** private_ptr)
{
  class cb* current_cb = get_current_cb();
  assert(current_cb);

  if (step)
  {
    *step = current_cb->step;
  }

  if (private_ptr)
  {
    *private_ptr = current_cb->private_ptr;
  }

  return cudaSuccess;
}

// Helpers
inline int cudaCallbackGetStep()
{
  int step;
  cudaCallbackGetStatus(&step, NULL);
  return step;
}

inline void* cudaCallbackGetPrivatePtr()
{
  void* private_ptr;
  cudaCallbackGetStatus(NULL, &private_ptr);
  return private_ptr;
}
#    endif // STATEFUL_CALLBACKS

inline void execute_callback(cb* cb)
{
  // Indicates if a stateful callback needs to be resubmitted Note that we do
  // not check for STATEFUL_CALLBACKS to avoid really miserable code ...
  int cb_restart = 0;

  if (cb->callback || cb->graph_callback)
  {
#    ifdef STATEFUL_CALLBACKS
    // This makes it possible to retrieve the state of the callback from
    // the user-provided callback function itself
    set_current_cb(cb);
#    endif
    if (cb->is_graph_host_node)
    {
      //           fprintf(stderr, "EXECUTING cb->graph_callback\n");
      cb->graph_callback(cb->userData);
    }
    else
    {
      //           fprintf(stderr, "EXECUTING cb->callback\n");
      cb->callback(cb->stream, cb->status, cb->userData);
    }

#    ifdef STATEFUL_CALLBACKS
    // Values equal to 0 or strictly negative indicate the callback is over
    cb_restart = (cb->step > 0);
    set_current_cb(NULL);
#    endif
  }

  // If the callback is over
  if (!cb_restart)
  {
    // Notify the GPU-side kernel that we can go-on
    //        fprintf(stderr, "NOTIFY flag %p\n", cb->completion_flag);
    *(cb->completion_flag) = 1;
  }
}

/*
 * Public functions
 */

/**
 * @brief Submit a callback in a specific queue
 *
 * @param stream the CUDA stream which should be used to synchronized with the callback
 * @param callback the description of the callback
 * @param userData a generic pointer passed to the callback
 * @param flags (ignored for now)
 * @param q the queue where to submit the callback
 * @return cudaError_t indicating if the submission was successful
 */
inline _CCCL_HOST cudaError_t cudaStreamAddCallbackWithQueue(
  cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags, callback_queue* q)
{
  if (q)
  {
    // We store the arguments in a structure that will be destroyed later on
    class cb* data  = new cb(stream, callback, userData, q);
    cudaError_t err = cudaStreamAddCallback(stream, callback_dispatcher, data, flags);

    // Submit completion kernel in the stream ...
    callback_completion_kernel<<<1, 1, 0, stream>>>(data->completion_flag);

    return err;
  }
  else
  {
    // If there is no queue, we use the usual callback mechanism
    return cudaStreamAddCallback(stream, callback, userData, flags);
  }
}

/**
 * @brief Submit a callback in a specific queue
 *
 * @param stream the CUDA stream which should be used to synchronized with the callback
 * @param fn the function implementing the callback
 * @param userData a generic pointer passed to the callback
 * @param q the queue where to submit the callback
 * @return cudaError_t indicating if the submission was successful
 */
inline _CCCL_HOST cudaError_t
cudaLaunchHostFuncWithQueue(cudaStream_t stream, cudaHostFn_t fn, void* userData, class callback_queue* q)
{
  assert(q);
  // We store the arguments in a structure that will be destroyed later on
  class cb* data  = new cb(fn, userData, q);
  cudaError_t err = cudaLaunchHostFunc(stream, (cudaHostFn_t) cudagraph_callback_dispatcher, data);

  // Submit completion kernel in the stream ...
  callback_completion_kernel<<<1, 1, 0, stream>>>(data->completion_flag);

  return err;
}

/**
 * @brief Submit a host node in a CUDA graph into a specific queue
 */
inline _CCCL_HOST cudaError_t cudaGraphAddHostNodeWithQueue(
  cudaGraphNode_t* node,
  cudaGraph_t graph,
  cudaGraphNode_t* deps,
  size_t ndeps,
  const cudaHostNodeParams* params,
  class callback_queue* q)
{
  assert(q);

  // We store the arguments in a structure that will be destroyed later on
  class cb* data = new cb(params->fn, params->userData, q);

  // XXX we should expose a child graph ...
  cudaGraphNode_t node0;

  cudaHostNodeParams dispatcher_params;
  dispatcher_params.fn       = (cudaHostFn_t) cudagraph_callback_dispatcher;
  dispatcher_params.userData = data;
  cudaError_t err            = cudaGraphAddHostNode(&node0, graph, deps, ndeps, &dispatcher_params);
  if (err != cudaSuccess)
  {
    return err;
  }

  // Submit completion kernel in the stream ...
  // callback_completion_kernel<<<1,1,0,stream>>>(data->completion_flag);
  cudaKernelNodeParams kernel_node_params;
  kernel_node_params.func            = (void*) callback_completion_kernel<>;
  kernel_node_params.gridDim         = 1;
  kernel_node_params.blockDim        = 1;
  kernel_node_params.kernelParams    = new void*[1];
  kernel_node_params.kernelParams[0] = (void*) &data->completion_flag;
  err                                = cudaGraphAddKernelNode(node, graph, &node0, 1, &kernel_node_params);

  return err;
}

/**
 * @brief Progress method of a callback queue
 *
 * This function processes the callback queue for a CUDA device. It can operate in blocking or non-blocking mode,
 * depending on the `blocking` parameter. In blocking mode, the function processes the queue until it is empty. In
 * non-blocking mode, the function processes the queue for one iteration and then returns.
 *
 * @param q A pointer to the callback_queue to be processed.
 * @param blocking mode of operation (`false` for non-blocking, `true` for blocking).
 * @return `true` if the queue's status is set to 1, indicating that the queue has been processed and is empty, `false`
 * otherwise.
 */
inline bool cudaCallbackQueueProgress(callback_queue* q, bool flag)
{
  int blocking = (flag == 1);

  int stop = 0;

  if (q->status == 1)
  {
    return true;
  }

  while (!stop)
  {
    class cb* cb = NULL;

    // Protect the queue
    pthread_mutex_lock(&q->mutex);

    // If there is something in this queue, handle it
    if (!q->dq.empty())
    {
      // Pop the front element
      assert(q->dq.size() > 0);
      cb = q->dq.front();
      q->dq.pop_front();
    }
    else
    {
      pthread_cond_wait(&q->cond, &q->mutex);
    }

    pthread_mutex_unlock(&q->mutex);

    if (cb)
    {
      execute_callback(cb);

#    ifdef STATEFUL_CALLBACKS
      int cb_restart = (cb->step > 0);
      if (cb_restart)
      {
        pthread_mutex_lock(&q->mutex);
        q->dq.push_back(cb);
        pthread_cond_broadcast(&q->cond);
        pthread_mutex_unlock(&q->mutex);
      }
#    endif

      if (cb->callback == NULL && cb->graph_callback == NULL)
      {
        // destroy
        q->status = 1;
        return true;
      }
    }

    if (blocking)
    {
      stop = 0;
    }
  }

  return false;
}
} // end namespace cuda::experimental::stf

#  endif // !_CCCL_COMPILER(MSVC)
#endif // _CCCL_DOXYGEN_INVOKED do not document
