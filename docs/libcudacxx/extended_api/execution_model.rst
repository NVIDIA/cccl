.. _libcudacxx-extended-api-execution-model:

Execution model
===============

CUDA C++ aims to provide `_parallel forward progress_ [intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__ 
for all device threads of execution, making the parallelization of pre-existing C++ applications with CUDA C++ straight-forward.

.. dropdown:: [intro.progress]

    - `[intro.progress.7] <https://eel.is/c++draft/intro.progress#7>`__: For a thread of execution
      providing `concurrent forward progress guarantees <https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees>`__, 
      the implementation ensures that the thread will eventually make progress for as long as it has not terminated.

      [Note 5: This applies regardless of whether or not other threads of execution (if any) have been or are making progress. 
      To eventually fulfill this requirement means that this will happen in an unspecified but finite amount of time. — end note]

    - `[intro.progress.9] <https://eel.is/c++draft/intro.progress>`__: For a thread of execution providing
      `parallel forward progress guarantees <https://eel.is/c++draft/intro.progress#9>`__, the implementation is not required to ensure that
      the thread will eventually make progress if it has not yet executed any execution step; once this thread has executed a step, 
      it provides concurrent forward progress guarantees.

      [Note 6: This does not specify a requirement for when to start this thread of execution, which will typically be specified by the entity
      that creates this thread of execution. For example, a thread of execution that provides concurrent forward progress guarantees and executes
      tasks from a set of tasks in an arbitrary order, one after the other, satisfies the requirements of parallel forward progress for these
      tasks. — end note]

.. _libcudacxx-extended-api-execution-model-host-threads:

Host threads
------------

The forward-progress provided by threads of execution created by the host implementation to
execute `main <https://en.cppreference.com/w/cpp/language/main_function>`__, `std::thread <https://en.cppreference.com/w/cpp/thread/thread>`__,
and `std::jthread https://en.cppreference.com/w/cpp/thread/jthread>`__ is implementation-defined behavior of the host
implementation `[intro.progress] <https://eel.is/c++draft/intro.progress>`__. 
General-purpose host implementations should provide _concurrent forward progress_.

If the host implementation provides `_concurrent forward progress_ [intro.progress.7] <https://eel.is/c++draft/intro.progress#7>`__,
then CUDA C++ provides `_parallel forward progress_ [intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__ for device threads.


.. _libcudacxx-extended-api-execution-model-device-threads:

Device threads
--------------

Once a device-thread makes progress:

- If the device-thread is part of a `Cooperative Grid <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73>`__,
  then all device-threads in its grid shall eventually make progress.
_ Otherwise, all device-threads in its `thread-block cluster <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters>`__ 
  shall eventually make progress.
  
  [Note: Threads in other thread-block clusters are not guaranteed to eventually make progress. - end note.]
  [Note: This implies that all device-threads in its thread block shall eventually make progress. - end note.]

The order in which device-threads eventually get a chance to make progress is _unspecified_.

Modify `[intro.progress.1] <https://eel.is/c++draft/intro.progress>`__ as follows:

The implementation may assume that any **host** thread will eventually do one of the following:

    1. terminate,
    2. invoke the function `std::his_thread::yield <https://en.cppreference.com/w/cpp/thread/yield>`__ (`[thread.thread.this] <http://eel.is/c++draft/thread.thread.this>`__),
    3. make a call to a library I/O function,
    4. perform an access through a volatile glvalue,
    5. perform a synchronization operation or an atomic operation, or
    6. continue execution of a trivial infinite loop (`[stmt.iter.general] <http://eel.is/c++draft/stmt.iter.general>`__).

**The implementation may assume that any device thread will eventually do one of the following:**

    1. **terminate**,
    2. **make a call to a library I/O function**,
    3. **perform an access through a volatile glvalue except if the designated object has automatic storage duration, or**
    4. **perform a synchronization operation or an atomic read operation except if the designated object has automatic storage duration.**

.. dropdown:: Examples of forward progress guarantee differences between host and device threads due to [intro.progress.1].

    The following examples refer to the itemized sub-clauses of the implementation assumptions for host and device threads above
    using "host.threads.<id>" and "device.threads.<id>", respectively.

    .. code:: cuda
        // Example: Execution.Model.Device.0
        // Outcome: grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration.
        __global__ void ii(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
            if (threadIdx.x == 0) {
                while(atom.load(cuda::memory_order_relaxed) == 0);
            } else if (threadIdx.x == 1) {
                atom.store(1, cuda::memory_order_relaxed);
            }
        }

    .. code:: cuda
        // Example: Execution.Model.Device.1
        // Allowed outcome: No thread makes progress because device threads don't support host.threads.2.
        __global__ void ii() {
            while(true) std::this_thread::yield();
        }

    .. code:: cuda
        // Example: Execution.Model.Device.2
        // Allowed outcome: No thread makes progress because device threads don't support host.threads.4
        // for objects with automatic storage duration (see exception in device.threads.3).
        __global__ void iv() {
            volatile bool True = true;
            while(True);
        }

    .. code:: cuda
        // Example: Execution.Model.Device.3
        // Allowed outcome: No thread makes progress because device threads don't support host.threads.5
        // for objects with automatic storage duration (see exception in device.threads.4).
        __global__ void v_atomic_automatic() {
            cuda::atomic<bool, cuda::thread_scope_thread> True = true;
            while(True.load());
        }

    .. code:: cuda
        // Example: Execution.Model.Device.4
        // Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
        __global void vi() {
            while(true) { /* empty */ }
        }

.. _libcudacxx-extended-api-execution-model-cuda-apis:

CUDA APIs
---------

Any CUDA API shall eventually either return or ensure at least one device-thread makes progress.

CUDA query functions (e.g. `cudaStreamQuery <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435>`__,
`cudaEventQuery <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7>`__, etc.) shall not consistently 
return ``cudaErrorNotReady`` without a device-thread making progress.

[Note: The device-thread need not be "related" to the API call, e.g., an API operating on one stream or process may ensure progress of a device-thread on another stream or process. - end note.]

[Note: A simple but not sufficient method to test workloads for CUDA API Forward Progress conformance is to run them with following environment variables set: ``CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=1`` - end note.]

.. dropdown:: Examples of CUDA API forward progress guarantees.

    .. code:: cuda
        // Example: Execution.Model.API.1 
        // Outcome: if device empty, terminates and returns cudaSuccess.
        // Rationale: CUDA guarantees that if the device is empty:
        // - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
        // - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
        // - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
        // - Therefore all device threads eventually exit the `hello_world`` grid.
        // - And `cudaDeviceSynchronize`` eventually unblocks.
        __global__ void hello_world() { __syncthreads(); }
        int main() {
            hello_world<<<1,2>>>();
            return (int)cudaDeviceSynchronize();
        }

    .. code:: cuda
        // Example: Execution.Model.API.2
        // Allowed outcome: eventually, no thread makes progress.
        // Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
        // However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
        // Therefore, the host thread may never be unblocked from the flag spin-loop.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void producer() { flag.store(1); }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            producer<<<1,1>>>();
            while (flag.load() == 0);
            return cudaDeviceSynchronize();
        }

    .. code:: cuda
        // Example: Execution.Model.API.3
        // Allowed outcome: eventually, no thread makes progress.
        // Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
        // the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void producer() { flag.store(1); }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            producer<<<1,1>>>();
            (void)cudaStreamQuery(0);
            while (flag.load() == 0);
            return cudaDeviceSynchronize();
        }

    .. code:: cuda
        // Example: Execution.Model.API.4
        // Outcome: terminates.
        // Rationale: same as Execution.Model.API.3, but this example repeatedly calls
        // a CUDA query API in within the flag spin-loop, which guarantees that the device thread
        // eventually makes progress.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void producer() { flag.store(1); }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            producer<<<1,1>>>();
            while (flag.load() == 0) { 
                (void)cudaStreamQuery(0);
            }
            return cudaDeviceSynchronize();
        }

.. _libcudacxx-extended-api-execution-model-stream-ordering:

Stream and event ordering
-------------------------

A device-thread shall not make progress if it is dependent on termination of one or more unterminated device-threads or tasks via CUDA streams and/or events.

[Note: This excludes dependencies such as Programmatic Dependent Launch or Launch Completion which do not encompass termination of the dependency. - end note.]

[Note: Tasks are also known as `Commands <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams>`__. - end note. ]

.. dropdown:: Examples of CUDA API forward progress guarantees due to Stream and event ordering

    .. code:: cuda
        // Example: Exeuction.Model.Stream.0
        // Allowed outcome: eventually, no thread makes progress.
        // Rationale: while CUDA guarantees that one device thread makes progress, since there
        // is no dependency between `first` and `second`, it does not guarantee which thread,
        // and therefore it could always pick the device thread from `second`, which then never 
        // unblocks from the spin-loop.
        // That is, `second` may starve `first`.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void first() { flag.store(1, rlx); }
        __global__ void second() { while(flag.load(rlx) == 0) {} }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            cudaStream_t s0, s1;
            cudaStreamCreate(&s0); 
            cudaStreamCreate(&s1);
            first<<<1,1,0,s0>>>();
            second<<<1,1,0,s1>>>();
            return cudaDeviceSynchronize();
        }

    .. code:: cuda
        // Example: Exeuction.Model.Stream.1
        // Outcome: terminates.
        // Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
        // between first and second, which requires CUDA to run the grids in order.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void first() { flag.store(1, rlx); }
        __global__ void second() { while(flag.load(rlx) == 0) {} }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            cudaStream_t s0;
            cudaStreamCreate(&s0); 
            first<<<1,1,0,s0>>>();
            second<<<1,1,0,s0>>>();
            return cudaDeviceSynchronize();
        }
