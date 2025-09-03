.. _libcudacxx-module:

libcu++
=======

.. toctree::
   :hidden:
   :maxdepth: 3

   Overview <self>
   setup
   releases
   standard_api
   extended_api
   ptx_api
   API reference <api/index>

``libcu++`` (``libcudacxx``) provides fundamental, idiomatic C++ abstractions that aim to make the lives of CUDA C++
developers easier.

Specifically, ``libcu++`` provides:
   - C++ Standard Library features usable in both host and device code
   - Extensions to C++ Standard Library features
   - Fundamental, CUDA-specific programming model abstractions

C++ Standard Library Features
-----------------------------

If you are a C++ developer, then you know the C++ Standard Library (`sometimes referred to as “The
STL” <https://stackoverflow.com/questions/5205491/whats-the-difference-between-stl-and-c-standard-library>`_)
as what comes along with your compiler and provides things like ``std::string`` or ``std::vector`` or ``std::atomic``.
It provides the fundamental abstractions that C++ developers need to build high quality applications and libraries.

By default, these abstractions aren't available when writing CUDA C++ device code because they don't have the necessary
``__host__ __device__`` decorators, and their implementation may not be suitable for using in and across host and device
code.

libcu++ aims to solve this problem by providing an opt-in, incremental, heterogeneous implementation of C++
Standard Library features:

   1. **Opt-in**: It does not replace the Standard Library provided by your host compiler (aka anything in ``std::``)
   2. **Incremental**: It does not provide a complete C++ Standard Library implementation
   3. **Heterogeneous**: It works in both host and device code, as well as passing between host and device code.

If you know how to use things like the ``<atomic>`` or ``<type_traits>`` headers from the C++ Standard Library, then
you know how to use libcu++.

All you have to do is add ``cuda/std/`` to the start of your includes and ``cuda::`` before any uses of ``std::``:

.. code:: cuda

   #include <cuda/std/atomic>
   cuda::std::atomic<int> x;

.. note::

   libcu++ does not provide its own documentation for Standard Library features. Instead, libcu++
   :ref:`documents which Standard Library headers <libcudacxx-standard-api>` are made available, and defers documentation of
   individual features within those headers to other sources like `cppreference <https://en.cppreference.com/w/>`_.

C++ Standard Library Extensions
-------------------------------

libcu++ provides CUDA C++ developers with familiar Standard Library utilities to improve productivity and flatten the
learning curve of learning CUDA. However, there are many aspects of writing high-performance CUDA C++ code that cannot
be expressed through purely Standard conforming APIs. For these cases, libcu++ also provides *extensions* of Standard
Library utilities.

For example, libcu++ extends ``atomic<T>`` and other synchronization primitives with the notion of a “thread scope”
that controls the strength of the memory fence.

To use utilities that are extensions to Standard Library features, drop the ``std``:

.. code:: cuda

   #include <cuda/atomic>
   cuda::atomic<int, cuda::thread_scope_device> x;

See the :ref:`Extended API <libcudacxx-extended-api>` section for more information.

Fundamental CUDA-specific Abstractions
--------------------------------------

Some abstractions that libcu++ provide have no equivalent in the C++ Standard Library, but are otherwise abstractions
fundamental to the CUDA C++ programming model.

For example, :ref:`cuda::memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>` is a vital abstraction
for asynchronous data movement between global and shared memory. This abstracts hardware features such as ``LDGSTS`` on
Ampere, and the Tensor Memory Accelerator (TMA) on Hopper.

See the :ref:`Extended API <libcudacxx-extended-api>` section for more information.

Summary: ``std::``, ``cuda::`` and ``cuda::std::``
--------------------------------------------------

-  ``std::`` / ``<*>``: This is your host compiler's Standard Library that works in ``__host__`` code only, although you
   can use the ``--expt-relaxed-constexpr`` flag to use any ``constexpr`` functions in ``__device__`` code.
   libcu++ does not replace or interfere with host compiler's Standard Library.
-  ``cuda::std::`` / ``<cuda/std/*>``: Conforming implementations of facilities from the Standard Library that work in
   ``__host__`` and  ``__device__`` code.
-  ``cuda::`` / ``<cuda/*>``: Conforming extensions to the Standard Library that work in ``__host__`` and ``__device__`` code.
-  ``cuda::device`` / ``<cuda/device/*>``: Conforming extensions to the Standard Library that work only in
   ``__device__`` code.
-  ``cuda::ptx``: C++ convenience wrappers for inline PTX that work only in ``__device__`` code.

Any libcu++ feature that works in ``__host__`` code also works when being compiled by a non-CUDA compiler (e.g., GCC).

.. code:: cuda

   // Standard C++, __host__ only.
   #include <atomic>
   std::atomic<int> x;

   // CUDA C++, __host__ __device__.
   // Strictly conforming to the C++ Standard.
   #include <cuda/std/atomic>
   cuda::std::atomic<int> x;

   // CUDA C++, __host__ __device__.
   // Conforming extensions to the C++ Standard.
   #include <cuda/atomic>
   cuda::atomic<int, cuda::thread_scope_block> x;

Licensing
---------

libcu++ is an open source project developed on `GitHub <https://github.com/nvidia/cccl>`_. It is NVIDIA's variant of
`LLVM's libc++ <https://libcxx.llvm.org>`_. libcu++ is distributed under the `Apache License v2.0 with LLVM
Exceptions <https://llvm.org/LICENSE.txt>`_.

Conformance
-----------

libcu++ aims to be a conforming implementation of the C++ Standard, `ISO/IEC IS 14882 <https://eel.is/c++draft>`_,
Clause 16 through 32.

ABI Evolution
-------------

libcu++ does not maintain long-term ABI stability. Promising long-term ABI stability would prevent us from fixing
mistakes and providing best in class performance. So, we make no such promises.

Every major CUDA Toolkit release, the ABI will be broken. The life cycle of an ABI version is approximately one year.
Long-term support for an ABI version ends after approximately two years. Please see the
:ref:`versioning section <libcudacxx-releases-versioning>` for more details.

We recommend that you always recompile your code and dependencies with the latest NVIDIA SDKs and use the latest
NVIDIA C++ Standard Library ABI. `Live at head <https://www.youtube.com/watch?v=tISy7EJQPzI&t=1032s>`_.
