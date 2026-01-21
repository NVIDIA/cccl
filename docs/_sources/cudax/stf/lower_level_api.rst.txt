.. _stf_lower_level_api:

Lower-level API
===============

In some situations, the use of ``operator->*()`` on the object returned
by ``ctx.task()`` (where ``ctx`` is a stream or graph context) may not
be suitable, for example when the number of parameters is not known
statically. To address such situations, CUDASTF provides a lower-level
interface for creating tasks, which is described below.

.. code:: cpp

   #include "cudastf/stf.h"
   #include "cudastf/__stf/stream/stream_ctx.h"

   using namespace cudastf;

   template <typename T>
   __global__ void axpy(int n, T a, T* x, T* y) {
       int tid = blockIdx.x * blockDim.x + threadIdx.x;
       int nthreads = gridDim.x * blockDim.x;

       for (int ind = tid; ind < n; ind += nthreads) {
           y[ind] += a * x[ind];
       }
   }

   int main(int argc, char** argv) {
       stream_ctx ctx;

       const size_t N = 16;
       double X[N], Y[N];

       for (size_t ind = 0; ind < N; ind++) {
           X[ind] = sin(double(ind));
           Y[ind] = cos(double(ind));
       }

       auto lX = ctx.logical_data(X);
       auto lY = ctx.logical_data(Y);

       double alpha = 3.14;

       /* Compute Y = Y + alpha X */
       auto t = ctx.task(lX.read(), lY.rw());
       t.start();
       slice<double> sX = t.get<0>();
       slice<double> sY = t.get<1>();
       axpy<<<16, 128, 0, t.get_stream()>>>(sX.size(), alpha, sX.data_handle(), sY.data_handle());
       t.end();

       ctx.sync();
   }

The ``ctx.task()`` call returns a task object. This object provides
access to the local description of the data associated with the task and
a CUDA stream that can be used to submit work asynchronously. The
beginning of the task body and its end are delimited by the ``.start()``
and ``.end()`` calls. Failing to call either of these methods or calling
them more than once or in the wrong order results in undefined behavior.

Asynchrony is achieved by using the CUDA stream, which provides a
mechanism to submit work on the execution place (here, implicitly the
current CUDA device). CUDA ensures that all kernels synchronized with
this CUDA stream will only be executed once all prerequisites have been
fulfilled (e.g., preceding tasks, data transfers, etc.). In addition,
CUDASTF performs all the necessary synchronization so that future tasks
will be properly synchronized with the operations enqueued in the CUDA
stream associated with this task after calling ``.start()`` and before
calling ``.end()``.

Compatibility with CUDA graphs
==============================

Similarly to the CUDA stream backend with a context of type
``stream_ctx``, the CUDA graph backend ``graph_ctx`` also provides a
low-level interface.

.. code:: cpp

       graph_ctx ctx;

       double X[1024], Y[1024];
       auto lX = ctx.logical_data(X);
       auto lY = ctx.logical_data(Y);

       for (int k = 0; k < 10; k++) {
           graph_task t = ctx.task();
           t.add_deps(handle_X.rw());
           t.start();
           cudaGraphNode_t n;
           cuda_safe_call(cudaGraphAddEmptyNode(&n, t.get_graph(), nullptr, 0));
           t.end();
       }

       graph_task t2 = ctx.task();
       t2.add_deps(lX.read(), lY.rw());
       t2.start();
       cudaGraphNode_t n2;
       cuda_safe_call(cudaGraphAddEmptyNode(&n2, t2.get_graph(), nullptr, 0));
       t2.end();

       ctx.sync();

A task in the CUDA graph backend corresponds to a *child graph*
automatically inserted into the CUDA graph associated to a ``graph_ctx``
context. The example above creates 10 tasks that modify logical data
``lX``, followed by a task that reads ``lX`` and modifies ``lY``. The
code illustrates how one can add dependencies to a task by using the
``add_deps`` method.

Similarly to the CUDA stream backend, a task is outlined by a pair of
calls to the ``start()``/``end()`` member functions.
