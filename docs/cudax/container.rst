.. _cudax-containers:

Containers library
===================

The headers of the container library provide facilities to store elements on the heap. They are heavily inspired by the
C++ `containers library <https://en.cppreference.com/w/cpp/container>`__ but deviate from the standard provided ones due to different requirements from
heterogeneous systems.

They build upon :ref:`memory_resources <libcudacxx-extended-api-memory-resources>` to ensure that e.g. execution space
annotations are checked by the type system.

Uninitialized buffers
---------------------

The ``<cuda/experimental/buffer>`` header contains facilities, that provide *heterogeneous* allocations to store objects
in uninitialized memory. This is a common request in HPC due to the high cost of initialization of large arrays.

.. warning::

   It is the users responsibility to ensure that any object is properly initialized before it is used and also destroyed
   before the underlying storage is deallocated.

.. toctree::
   :maxdepth: 3

   container/uninitialized_buffer
