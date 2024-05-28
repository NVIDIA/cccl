.. _libcudacxx-ptx-instructions-elect-sync:

elect.sync
==========

-  PTX ISA:
   `elect.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync>`__

*Implementation note:* Since C++ does not support returning multiple
values, the variant of the instruction that returns both a predicate and
an updated membermask is not supported.

This instruction can also be accessed through the cooperative groups
`invoke_one API <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#invoke-one-and-invoke-one-broadcast>`__.

elect.sync
^^^^^^^^^^
.. code:: cuda

   // elect.sync _|is_elected, membermask; // PTX ISA 80, SM_90
   template <typename=void>
   __device__ static inline bool elect_sync(
     const uint32_t& membermask);
