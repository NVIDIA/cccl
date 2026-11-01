Developer Overview
##################

.. toctree::
   :hidden:
   :maxdepth: 1

   developer/cmake_options
   developer/systems

.. important::

   **APIs in the ``thrust::`` namespace are considered in maintenance mode. New development happens in the ``cuda::`` namespace.**

   Thrust is not deprecated: we will continue to provide updates and support for it, including performance updates.

   Thrust has accumulated a number of design issues that we cannot fix without breaking existing users,
   and we know how to build these facilities better.
   We are therefore concentrating our effort on the parallel algorithms, iterators, memory resources, and containers in
   :ref:`libcu++ <libcudacxx-module>`
   (see :ref:`fancy iterators <libcudacxx-extended-api-iterators>`,
   :ref:`memory resources <libcudacxx-extended-api-memory-resources>`, and
   :ref:`CUDA buffers <cccl-runtime-buffer>`),
   and we encourage users to migrate to the corresponding ``cuda::`` facilities.

   Consequently, we **no longer accept new features or extensions** in the ``thrust::`` namespace,
   such as new algorithms, iterators, containers, execution policies, or backends.
   We continue to accept:

   - bug fixes,
   - performance improvements to existing functionality, and
   - maintenance work required to keep Thrust building and working with supported compilers and CUDA Toolkits.

   Please direct new contributions and proposals for new functionality to features in the ``cuda::`` namespace.
   If you are unsure whether a change fits, open an issue on `GitHub <https://github.com/NVIDIA/cccl/issues>`__ before starting work.

This living document serves as a guide to the design of the internal structure of Thrust.
