..
   This file was automatically generated. Do not edit.

elect.sync
^^^^^^^^^^
.. code-block:: cuda

   // elect.sync _|is_elected, membermask; // PTX ISA 80, SM_90
   template <typename = void>
   __device__ static inline bool elect_sync(
     const uint32_t& membermask);
