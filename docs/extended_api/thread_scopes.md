---
parent: Extended API
nav_order: 0
---

# Thread Scopes

```cuda
namespace cuda {

// Header `<cuda/atomic>`.

enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};

template <typename T, thread_scope Scope = thread_scope_system>
class atomic;

void atomic_thread_fence(std::memory_order, thread_scope = thread_scope_system);

// Header `<cuda/barrier>`.

template <thread_scope Scope,
          typename Completion = /* implementation-defined */>
class barrier;

// Header `<cuda/latch>`.

template <thread_scope Scope>
class latch;

// Header `<cuda/semaphore>`.

template <thread_scope Scope>
class binary_semaphore;

template <thread_scope Scope = thread_scope_thread,
          ptrdiff_t LeastMaximumValue = /* implementation-defined */>
class counting_semaphore;

}
```

Standard C++ presents a view that the cost to synchronize threads is uniform
  and low.
CUDA C++ is different: the overhead is low among threads within a block, and
  high across arbitrary threads in the system.

To bridge these two realities, libcu++ introduces **thread scopes**
  to the Standard's concurrency facilities in the `cuda::` namespace, while
  retaining the syntax and semantics of Standard C++ by default.
A thread scope specifies the kind of threads that can synchronize with each
  other using a primitive such as an `atomic` or a `barrier`.

## Scope Relationships

Each program thread is related to each other program thread by one or more
  thread scope relations:
- Each thread (CPU or GPU) is related to each other thread in the computer
  system by the *system* thread scope, specified with `thread_scope_system`.
- Each GPU thread is related to each other GPU thread in the same CUDA device
  by the *device* thread scope, specified with `thread_scope_device`.
- Each GPU thread is related to each other GPU thread in the same CUDA block
  by the *block* thread scope, specified with `thread_scope_block`.
- Each thread (CPU or GPU) is related to itself by the `thread` thread scope,
  specified with `thread_scope_thread`.

Objects in namespace `cuda::std::` have the same behavior as corresponding
  objects in namespace `cuda::` when instantiated with a scope of
  `cuda::thread_scope_system`.

Refer to the [CUDA programming guide] for more information on how CUDA launches
  threads into devices and blocks.

## Atomicity

An atomic operation is atomic at the scope it specifies if:
- it specifies a scope other than `thread_scope_system`, or
- it affects an object in unified memory and [`concurrentManagedAccess`] is
  `1`, or
- it affects an object in CPU memory and [`hostNativeAtomicSupported`] is `1`,
  or
- it affects an object in GPU memory and only GPU threads access it.

Refer to the [CUDA programming guide] for more information on
  unified memory, CPU memory, and GPU peer memory.

## Data Races

Modify [intro.races paragraph 21] of ISO/IEC IS 14882 (the C++ Standard) as
  follows:
> The execution of a program contains a data race if it contains two
> potentially concurrent conflicting actions, at least one of which is not
> atomic
> ***at a scope that includes the thread that performed the other operation***,
> and neither happens before the other, except for the special
> case for signal handlers described below. Any such data race results in
> undefined behavior. [...]

Modify [thread.barrier.class paragraph 4] of ISO/IEC IS 14882 (the C++
  Standard) as follows:
> 4. Concurrent invocations of the member functions of `barrier`, other than its
> destructor, do not introduce data races
> ***as if they were atomic operations***.
> [...]

Modify [thread.latch.class paragraph 2] of ISO/IEC IS 14882 (the C++ Standard)
  as follows:
> 2. Concurrent invocations of the member functions of `latch`, other than its
> destructor, do not introduce data races
> ***as if they were atomic operations***.

Modify [thread.sema.cnt paragraph 3] of ISO/IEC IS 14882 (the C++ Standard) as
  follows:
> 3. Concurrent invocations of the member functions of `counting_semaphore`,
> other than its destructor, do not introduce data races
> ***as if they were atomic operations***.

Modify [atomics.fences paragraph 2 through 4] of ISO/IEC IS 14882 (the C++
  Standard) as follows:
> A release fence A synchronizes with an acquire fence B if there exist atomic
> operations X and Y, both operating on some atomic object M, such that A is
> sequenced before X, X modifies M, Y is sequenced before B, and Y reads the
> value written by X or a value written by any side effect in the hypothetical
> release sequence X would head if it were a release operation,
> ***and each operation (A, B, X, and Y) specifies a scope that includes the thread that performed each other operation***.

> A release fence A synchronizes with an atomic operation B that performs an
> acquire operation on an atomic object M if there exists an atomic operation X
> such that A is sequenced before X, X modifies M, and B reads the value
> written by X or a value written by any side effect in the hypothetical
> release sequence X would head if it were a release operation,
> ***and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.

> An atomic operation A that is a release operation on an atomic object M
> synchronizes with an acquire fence B if there exists some atomic operation X
> on M such that X is sequenced before B and reads the value written by A or a
> value written by any side effect in the release sequence headed by A,
> ***and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.


[intro.races paragraph 21]: https://eel.is/c++draft/intro.races#21
[thread.barrier.class paragraph 4]: https://eel.is/c++draft/thread.barrier.class#4
[thread.latch.class paragraph 2]: https://eel.is/c++draft/thread.latch.class#2
[thread.sema.cnt paragraph 3]: https://eel.is/c++draft/thread.sema.cnt#3
[atomics.fences paragraph 2 through 4]: https://eel.is/c++draft/atomics.fences#2

[CUDA programming guide]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
