.. _libcudacxx-extended-api-synchronization-barrier-barrier-native-handle:

cuda::device::barrier_native_handle
=======================================

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   __device__ cuda::std::uint64_t* cuda::device::barrier_native_handle(
     cuda::barrier<cuda::thread_scope_block>& bar);

Returns a pointer to the native handle of a :ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`
if its scope is ``cuda::thread_scope_block`` and it is allocated in shared memory.
The pointer is suitable for use with PTX instructions.

Notes
-----

If ``bar`` is not in ``__shared__`` memory, the behavior is undefined.

Return Value
------------

A pointer to the PTX “mbarrier” subobject of the :ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`
object.

Example
-------

.. code:: cuda

   #include <cuda/barrier>

   __global__ void example_kernel(cuda::barrier<cuda::thread_scope_block>& bar) {
     auto ptr = cuda::device::barrier_native_handle(bar);

     asm volatile (
         "mbarrier.arrive.b64 _, [%0];"
         :
         : "l" (ptr)
         : "memory");
     // Equivalent to: `(void)b.arrive()`.
   }

`See it on Godbolt <https://godbolt.org/z/dr4798Y76>`_
