.. _libcudacxx-extended-api-warp-communitioncal:

Warp Communication
==================

.. toctree::
   :hidden:
   :maxdepth: 1

   cuda::warp_shuffle <warp_communication/warp_shuffle>

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`warp_shuffle_idx <libcudacxx-extended-api-warp-communitioncal-warp-shuffle>`
     - Warp shuffle from a specific lane
     - CCCL 3.0.0 / CUDA 13.0

   * - :ref:`warp_shuffle_up <libcudacxx-extended-api-warp-communitioncal-warp-shuffle>`
     - Warp shuffle from original lane index - delta
     - CCCL 3.0.0 / CUDA 13.0

   * - :ref:`warp_shuffle_down <libcudacxx-extended-api-warp-communitioncal-warp-shuffle>`
     - Warp shuffle from original lane index + delta
     - CCCL 3.0.0 / CUDA 13.0

   * - :ref:`warp_shuffle_xor <libcudacxx-extended-api-warp-communitioncal-warp-shuffle>`
     - Warp shuffle from original lane index xor mask
     - CCCL 3.0.0 / CUDA 13.0
