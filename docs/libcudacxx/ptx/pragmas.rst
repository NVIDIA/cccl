.. _libcudacxx-ptx-pragmas:

PTX Pragmas
================

.. toctree::
   :maxdepth: 1

   pragma/enable_smem_spilling

.. list-table:: `.pragma Strings <https://docs.nvidia.com/cuda/parallel-thread-execution/#descriptions-pragma-strings>`__
   :widths: 50 50
   :header-rows: 1

   * - Pragma
     - Available in libcu++
   * - `"nounroll" <https://docs.nvidia.com/cuda/parallel-thread-execution/#pragma-strings-nounroll>`__
     - No
   * - `"used_bytes_mask" <https://docs.nvidia.com/cuda/parallel-thread-execution/#pragma-strings-used-bytes-mask>`__
     - No
   * - `"enable_smem_spilling" <https://docs.nvidia.com/cuda/parallel-thread-execution/#pragma-strings-enable-smem-spilling>`__
     - Yes, CCCL 3.2.0 / CUDA 13.2
   * - `"frequency" <https://docs.nvidia.com/cuda/parallel-thread-execution/#pragma-strings-frequency>`__
     - No
