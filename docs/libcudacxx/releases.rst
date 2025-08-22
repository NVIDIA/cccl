.. _libcudacxx-releases:

.. warning::
    This table is no longer maintained since libcu++ was merged into the CCCL project.
    For the latest releases, checkout the CCCL releases
    on `GitHub <https://github.com/NVIDIA/cccl/releases>`_.

Releases
========

.. toctree::
   :maxdepth: 1

   releases/changelog
   releases/versioning

The latest ABI version is always the default.

.. list-table::
   :widths: 10 10 20 60
   :header-rows: 1

   * - API Version
     - ABI Version(s)
     - Included In
     - Summary
   * - 1.0.0
     - 1
     - CUDA 10.2
     - ``<cuda/std/atomic>``, ``<cuda/std/type_traits>``
   * - 1.1.0
     - 2
     - CUDA 11.0 Early Access
     - barriers, latches, semaphores, clocks
   * - 1.2.0
     - 3, 2
     - CUDA 11.1
     - pipelines, asynchronous copies
   * - 1.3.0
     - 3, 2
     - CUDA 11.2
     - ``<cuda/std/tuple>``, ``<cuda/std/utility>``
   * - 1.4.0
     - 3, 2
     -
     - ``<cuda/std/complex>``, calendars, dates
   * - 1.4.1
     - 3, 2
     - Cuda 11.3
     - Bugfixes
   * - 1.5.0
     - 4, 3, 2
     - Cuda 11.4
     - ``<nv/target>``
   * - 1.6.0
     - 4, 3, 2
     - Cuda 11.5
     - ``cuda::annotated_ptr``, atomic refactor
   * - 1.7.0
     - 4, 3, 2
     - Cuda 11.6
     - ``atomic_ref``, 128 bit support
   * - 1.8.0
     - 4, 3, 2
     - Cuda 11.7
     - ``<cuda/std/array>``, ``<cuda/std/bit>``
   * - 1.8.1
     - 4, 3, 2
     - Cuda 11.8
     - Bugfixes and documentation updates
   * - 1.9.0
     - 4, 3, 2
     - Cuda 12.0
     - ``float`` and ``double`` support for atomics
   * - 2.1.0
     - 4, 3, 2
     - Cuda 12.2
     - ``<cuda/std/span>``, ``<cuda/std/mdspan>``, ``<cuda/std/concepts>``
   * - 2.1.1
     - 4, 3, 2
     - Cuda 12.2.1
     - Bugfixes and compiler support changes
   * - 2.2.0
     - 4, 3, 2
     - CUDA 12.3
     - ``<cuda/memory_resource>`` and ``<cuda/stream_ref>``, ``<cuda/std/cfloat>``, ``<cuda/std/cmath>``,
       ``<cuda/std/cstdint>``, ``<cuda/std/cstdlib>``,
   * - 2.3.0
     - 4, 3, 2
     - CUDA 12.4
     - ``<cuda/ptx>``, ranges machinery in ``<cuda/std/iterator>``, ``<cuda/std/optional>``, ``<cuda/std/variant>``, ``<cuda/std/expected>``
   * - 2.4.0
     - 4, 3, 2
     - CUDA 12.5
     - More ptx exposure in ``<cuda/ptx>``, ranges machinery in ``<cuda/std/ranges>``
