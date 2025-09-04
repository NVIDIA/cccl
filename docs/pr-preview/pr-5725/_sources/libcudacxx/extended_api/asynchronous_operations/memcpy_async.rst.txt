.. _libcudacxx-extended-api-asynchronous-operations-memcpy-async:

``cuda::memcpy_async``
======================

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   // (1)
   template <typename Shape, cuda::thread_scope Scope, typename CompletionFunction>
   __host__ __device__
   void cuda::memcpy_async(void* destination, void const* source, Shape size,
                           cuda::barrier<Scope, CompletionFunction>& barrier);

   // (2)
   template <typename Group,
             typename Shape, cuda::thread_scope Scope, typename CompletionFunction>
   __host__ __device__
   void cuda::memcpy_async(Group const& group,
                           void* destination, void const* source, Shape size,
                           cuda::barrier<Scope, CompletionFunction>& barrier);

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   // (3)
   template <typename Shape, cuda::thread_scope Scope>
   __host__ __device__
   void cuda::memcpy_async(void* destination, void const* source, Shape size,
                           cuda::pipeline<Scope>& pipeline);

   // (4)
   template <typename Group, typename Shape, cuda::thread_scope Scope>
   __host__ __device__
   void cuda::memcpy_async(Group const& group,
                           void* destination, void const* source, Shape size,
                           cuda::pipeline<Scope>& pipeline);

Defined in header ``<cuda/annotated_ptr>``:

.. code:: cuda

   // (5)
   template <typename Dst, typename Src, typename SrcProperty, typename Shape, typename Sync>
   __host__ __device__
   void memcpy_async(Dst* dst, cuda::annotated_ptr<Src, SrcProperty> src, Shape size, Sync& sync);

   // (6)
   template<typename Dst, typename DstProperty, typename Src, typename SrcProperty, typename Shape, typename Sync>
   __host__ __device__
   void memcpy_async(cuda::annotated_ptr<Dst, DstProperty> dst, cuda::annotated_ptr<Src, SrcProperty> src, Shape size, Sync& sync);

   // (7)
   template<typename Group, typename Dst, typename Src, typename SrcProperty, typename Shape, typename Sync>
   __host__ __device__
   void memcpy_async(Group const& group, Dst* dst, cuda::annotated_ptr<Src, SrcProperty> src, Shape size, Sync& sync);

   // (8)
   template<typename Group, typename Dst, typename DstProperty, typename Src, typename SrcProperty, typename Shape, typename Sync>
   __host__ __device__
   void memcpy_async(Group const& group, cuda::annotated_ptr<Dst, DstProperty> dst, cuda::annotated_ptr<Src, SrcProperty> src, Shape size, Sync& sync);

``cuda::memcpy_async`` asynchronously copies ``size`` bytes from the
memory location pointed to by ``source`` to the memory location pointed
to by ``destination``. Both objects are reinterpreted as arrays of
``unsigned char``.

1. Binds the asynchronous copy completion to ``cuda::barrier`` and
   issues the copy in the current thread.
2. Binds the asynchronous copy completion to ``cuda::barrier`` and
   cooperatively issues the copy across all threads in ``group``.
3. Binds the asynchronous copy completion to ``cuda::pipeline`` and
   issues the copy in the current thread.
4. Binds the asynchronous copy completion to ``cuda::pipeline`` and
   cooperatively issues the copy across all threads in ``group``.
5. 5-8: convenience wrappers using ``cuda::annotated_ptr`` where
   ``Sync`` is either ``cuda::barrier`` or ``cuda::pipeline``.

Notes
-----

``cuda::memcpy_async`` have similar constraints to `std::memcpy <https://en.cppreference.com/w/cpp/string/byte/memcpy>`_,
namely:

   - If the objects overlap, the behavior is undefined.
   - If either ``destination`` or ``source`` is an invalid or null pointer, the behavior is undefined
     (even if ``count`` is zero).
   - If the objects are `potentially-overlapping <https://en.cppreference.com/w/cpp/language/object#Subobjects>`_
     the behavior is undefined.
   - If the objects are not of `TriviallyCopyable <https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable>`_
     type the program is ill-formed, no diagnostic required.
   - If *Shape* is :ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-aligned-size>`, ``source``
     and ``destination`` are both required to be aligned on ``cuda::aligned_size_t::align``, else the behavior is
     undefined.
   - If ``cuda::pipeline`` is in a *quitted state*
     (see :ref:`cuda::pipeline::quit <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`),
     the behavior is undefined.
   - For cooperative variants, if the parameters are not the same across all threads in ``group``, the behavior is
     undefined.

Template Parameters
-------------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Group``
     - A type satisfying the [*Group*] concept.
   * - ``Shape``
     - Either `cuda::std::size_t <https://en.cppreference.com/w/c/types/size_t>`_
       or :ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-aligned-size>`.

Parameters
----------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``group``
     - The group of threads.
   * - ``destination``
     - Pointer to the memory location to copy to.
   * - ``source``
     - Pointer to the memory location to copy from.
   * - ``size``
     - The number of bytes to copy.
   * - ``barrier``
     - The barrier object used to wait on the copy completion.
   * - ``pipeline``
     - The pipeline object used to wait on the copy completion.

Examples
--------

.. code:: cuda

   #include <cuda/barrier>

   __global__ void example_kernel(char* dst, char* src) {
     cuda::barrier<cuda::thread_scope_system> bar;
     init(&bar, 1);

     cuda::memcpy_async(dst,     src,      1, bar);
     cuda::memcpy_async(dst + 1, src + 8,  1, bar);
     cuda::memcpy_async(dst + 2, src + 16, 1, bar);
     cuda::memcpy_async(dst + 3, src + 24, 1, bar);

     bar.arrive_and_wait();
   }

`See it on Godbolt <https://godbolt.org/z/od6q9s8fq>`_
