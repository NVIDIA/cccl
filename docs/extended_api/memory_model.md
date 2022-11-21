---
parent: Extended API
nav_order: 0
---

# Memory model

Standard C++ presents a view that the cost to synchronize threads is uniform and low.

CUDA C++ is different: the cost to synchronize threads grows as threads are further apart.
It is low across threads within a block, but high across arbitrary threads in the system running on multiple GPUs and CPUs.

To account for non-uniform thread synchronization costs that are not always low, CUDA C++ extends the standard C++ memory model and concurrency facilities in the `cuda::` namespace with **thread scopes**, retaining the syntax and semantics of standard C++ by default.

## Thread Scopes

A _thread scope_ specifies the kind of threads that can synchronize with each other using synchronization primitive such as [`atomic`] or [`barrier`].

```cuda
namespace cuda {

enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};

}  // namespace cuda
```

[`atomic`]: synchronization_primitives/atomic.md
[`barrier`]: synchronization_primitives/barrier.md

### Scope Relationships

Each program thread is related to each other program thread by one or more thread scope relations:
- Each thread in the system is related to each other thread in the system by the *system* thread scope: `thread_scope_system`.
- Each GPU thread is related to each other GPU thread in the same CUDA device by the *device* thread scope: `thread_scope_device`.
- Each GPU thread is related to each other GPU thread in the same CUDA thread block by the *block* thread scope: `thread_scope_block`.
- Each thread is related to itself by the `thread` thread scope: `thread_scope_thread`.

## Synchronization primitives

Types in namespaces `std::` and `cuda::std::` have the same behavior as corresponding types in namespace `cuda::` when instantiated with a scope of `cuda::thread_scope_system`.

## Atomicity

An atomic operation is atomic at the scope it specifies if:
- it specifies a scope other than `thread_scope_system`, **or**

the scope is `thread_scope_system` and:

- it affects an object in [unified memory] and [`concurrentManagedAccess`] is `1`, **or**
- it affects an object in CPU memory and [`hostNativeAtomicSupported`] is `1`, **or**
- it is a load or store that affects a naturally-aligned object of sizes `1`, `2`, `4`, or `8` bytes on [mapped memory], **or**
- it affects an object in GPU memory and only GPU threads access it.

[mapped memory]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory

Refer to the [CUDA programming guide] for more information on [unified memory], [mapped memory], CPU memory, and GPU peer memory.

[mapped memory]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory
[unified memory]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
[CUDA programming guide]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
[`concurrentManagedAccess`]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported`]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f

## Data Races

Modify [intro.races paragraph 21] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at least one of which is not atomic ***at a scope that includes the thread that performed the other operation***, and neither happens before the other, except for the special case for signal handlers described below. Any such data race results in undefined behavior. [...]

Modify [thread.barrier.class paragraph 4] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> 4. Concurrent invocations of the member functions of `barrier`, other than its destructor, do not introduce data races ***as if they were atomic operations***. [...]

Modify [thread.latch.class paragraph 2] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> 2. Concurrent invocations of the member functions of `latch`, other than its destructor, do not introduce data races ***as if they were atomic operations***.

Modify [thread.sema.cnt paragraph 3] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> 3. Concurrent invocations of the member functions of `counting_semaphore`, other than its destructor, do not introduce data races ***as if they were atomic operations***.

Modify [thread.stoptoken.intro paragraph 5] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> Calls to the functions request_­stop, stop_­requested, and stop_­possible do not introduce data races ***as if they were atomic operations***. [...]

[thread.stoptoken.intro paragraph 5]: https://eel.is/c++draft/thread#stoptoken.intro-5

Modify [atomics.fences paragraph 2 through 4] of ISO/IEC IS 14882 (the C++ Standard) as follows:
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

## Example: Message Passing

The following example passes a message stored to the `x` variable by a thread in block `0` to a thread in block `1` via the flag `f`:

<table class="display">
<tr class="header"><td colspan="2" markdown="span" align="center">
`int x = 0;`<br>
`int f = 0;`
</td></tr>
<tr class="header">
<td markdown="span" align="center"> 
**Thread 0 Block 0** 
</td><td markdown="span" align="center"> 
**Thread 0 Block 1** 
</td>
</tr>
<tr>
<td markdown="span">
`x = 42;`<br>
`cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);`<br>
`flag.store(1, memory_order_release);`
</td>
<td markdown="span">
`cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);`<br>
`while(flag.load(memory_order_acquire) != 1);`<br>
`assert(x == 42);`
</td>
</tr>
</table>

In the following variation of the previous example, two threads concurrently access the `f` object without synchronization, which leads to a **data race**, and exhibits **undefined behavior**:

<table>
<tr><td colspan="2" markdown="span" align="center">
`int x = 0;`<br>
`int f = 0;`
</td></tr>
<tr>
<td markdown="span" align="center"> 
**Thread 0 Block 0** 
</td><td markdown="span" align="center"> 
**Thread 0 Block 1** 
</td>
</tr>
<tr>
<td markdown="span">
`x = 42;`<br>
`cuda::atomic_ref<int, cuda::thread_scope_block> flag(f);`<br>
`flag.store(1, memory_order_release);  // UB: data race`
</td>
<td markdown="span">
`cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);`<br>
`while(flag.load(memory_order_acquire) != 1); // UB: data race`<br>
`assert(x == 42);`
</td>
</tr>
</table>

While the memory operations on `f` - the store and the loads - are atomic, the scope of the store operation is "block scope". Since the store is performed by Thread 0 of Block 0, it only includes all other threads of Block 0. However, the thread doing the loads is in Block 1, i.e., it is not in a scope included by the store operation performed in Block 0, causing the store and the load to not be "atomic", and introducing a data-race. 

For more examples see the [PTX memory consistency model litmus tests].

[PTX memory consistency model litmus tests]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms 
[intro.races paragraph 21]: https://eel.is/c++draft/intro.races#21
[thread.barrier.class paragraph 4]: https://eel.is/c++draft/thread.barrier.class#4
[thread.latch.class paragraph 2]: https://eel.is/c++draft/thread.latch.class#2
[thread.sema.cnt paragraph 3]: https://eel.is/c++draft/thread.sema.cnt#3
[atomics.fences paragraph 2 through 4]: https://eel.is/c++draft/atomics.fences#2
