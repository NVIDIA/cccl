
.. _libcudacxx-ptx-pragma-enable_smem_spilling:

.pragma "enable_smem_spilling"
==============================

-  PTX ISA:
   `.pragma "enable_smem_spilling" <https://docs.nvidia.com/cuda/parallel-thread-execution/#pragma-strings-enable-smem-spilling>`__

.. code:: cuda

  // PTX ISA 9.0
  // .pragma "enable_smem_spilling";

  namespace cuda::ptx {

  __device__ static __forceinline__
  void enable_smem_spilling() noexcept;

  } // namespace cuda::ptx
