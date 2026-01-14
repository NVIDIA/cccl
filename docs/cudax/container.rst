.. _cudax-containers:

Containers library
===================

.. toctree::
   :glob:
   :maxdepth: 1

   api/class*uninitialized__buffer*

The headers of the container library provide facilities to store elements on the heap. They are heavily inspired by the
C++ `containers library <https://en.cppreference.com/w/cpp/container>`__ but deviate from the standard provided ones due to different requirements from
heterogeneous systems.

They build upon :ref:`memory_resources <libcudacxx-extended-api-memory-resources>` to ensure that e.g. execution space
annotations are checked by the type system.

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`<cuda/experimental/container.cuh> <libcudacxx-containers-uninitialized-buffer>`
     - Facilities providing uninitialized *heterogeneous* potentially stream ordered storage satisfying a set of properties
     - cudax 2.7.0 / CCCL 2.7.0
   * - :ref:`<cuda/experimental/container.cuh> <libcudacxx-containers-heterogeneous-iterator>`
     - Iterator providing type safe iteration and memory access through a set of properties
     - cudax 2.7.0 / CCCL 2.7.0
   * - :ref:`<cuda/experimental/container.cuh> <libcudacxx-containers-async-vector>`
     - High level containers, that provide *heterogeneous* storage satisfying a set of properties
     - cudax 2.7.0 / CCCL 2.7.0
