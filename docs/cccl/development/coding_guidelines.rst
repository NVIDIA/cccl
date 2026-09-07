.. _cccl-development-coding-guidelines:

CCCL C++ Coding Guidelines
============================

The following guidelines must generally be followed when contributing to any of the CCCL C++ libraries,
including tests, benchmarks, examples and utility libraries.
General guidelines apply universally, but may be overridden by library-specific or path-specific guidelines.

General
-------

#. Always prefer entities from ``cuda::std::`` over ``std::``
   and other functionality or macros from the ``<cuda/...>`` and ``<cuda/std/...>`` headers.
   They generally work in host and device code, often work with NVRTC, and help testing our implementation.
#. Use :doc:`CCCL internal macros <macro>` over compiler/vendor-specific keywords in library implementation
   headers. E.g., use ``_CCCL_HOST_DEVICE`` instead of ``__host__ __device__``, or ``_CCCL_FORCEINLINE``
   over ``__forceinline``. Generally, prefer ``_CCCL_HOST_API``, ``_CCCL_DEVICE_API``,
   ``_CCCL_HOST_DEVICE_API``, ``_CCCL_TILE_API``, or ``_CCCL_API`` for any function inside CCCL.
   Examples and documentation must not use these macros and should support vendor
   attributes and keywords instead. Tests should only use macros if they are strictly required for the
   test to work. For instance, ``_CCCL_HOST_DEVICE`` may be required for tests targeting non-CUDA
   backends.
#. Fully qualify any reference to a libcu++ entity in any C++ library header with ``::cuda::std::``,
   ``::cuda::``, etc. This avoids ambiguities when users define namespaces called ``cuda`` themselves.
   Use just ``cuda::std::`` etc. in examples, tests and documentation.
#. Fully qualify any references to a host standard library entity in any C++ library header with
   ``::std::``. This avoids ambiguities with entities from ``::cuda::std::``. Use just ``std::`` in
   examples, tests and documentation.
#. Doxygen comments should start with ``//!``
#. In documentation comments, we prefer the use of ``@`` to start Doxygen commands.
#. Prefer ``@c`` when referring to code entities
#. Macros for function attributes, like ``_CCCL_HOST_DEVICE``, should be ordered before declaration
   specifiers like ``constexpr`` or ``static``
#. Include libcu++ detail headers (starting with ``__``) also in downstream projects (like CUB, Thrust,
   etc.) to reduce compile time. In tests and examples, always prefer the public headers.
#. Comments should express what the code cannot say. They complement code. Before writing an
   explanatory comment, consider whether refactoring the code could allow the code to express the same.
   This avoids code and comments getting out of sync.
#. Do not mark kernels with ``__global__`` but with ``_CCCL_KERNEL_ATTRIBUTES``
#. All user-defined names for entities should use ``snake_case``, except for template parameters, which
   use ``PascalCase``, and macros, which use ``ALL_CAPS``.

libcu++
-------

#. Always fully qualify function calls, even to functions in the same namespace. This avoids ADL.
#. Defaulted constructors should be marked with ``_CCCL_HIDE_FROM_ABI``
#. libcu++ headers like ``<cuda/foo>`` are strict supersets of ``<cuda/std/foo>`` and thus always
   include the corresponding ``<cuda/std/...>`` header.
#. All user-defined names for entities which are not part of the public API
   must be prefixed with ``__`` when they use ``snake_case``,
   and with ``_`` when they use ``PascalCase`` or ``ALL_CAPS``.
   This turns them into C++ reserved identifiers to avoid name collisions with user code and macros.

CUB and Thrust
--------------

#. Non-public entities, which are not macros, should be put inside a ``detail`` namespace (preferred)
   or prefixed with ``__``.

CUB
----

#. Entities in the public API, which are not macros, must be written in ``PascalCase``
   to stay consistent with existing practice.
#. Unit tests must use ``c2h::[host|device]_vector`` as containers over Thrust vectors,
   since they handle OOM conditions gracefully on some testing systems.
#. Header files use ``.cuh`` as file extension

Thrust
------

#. Header files use ``.h`` as file extension

Licensing
---------

CCCL uses a mix of different licenses for historical reasons.

#. Any net new file should be licensed under Apache-2.0 WITH LLVM-exception.
#. For files with license headers, strongly prefer a 2-line SPDX header like:

   .. code-block:: text

      // SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
      // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
