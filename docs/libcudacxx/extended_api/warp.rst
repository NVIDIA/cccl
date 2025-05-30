.. _libcudacxx-extended-api-warp:

Warp
====

.. toctree::
   :hidden:
   :maxdepth: 1

   cuda::device::warp_shuffle <warp/warp_shuffle>
   cuda::device::lane_mask <warp/lane_mask>

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`warp_shuffle_idx <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from a specific lane
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`warp_shuffle_up <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from original lane index - delta
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`warp_shuffle_down <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from original lane index + delta
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`warp_shuffle_xor <libcudacxx-extended-api-warp-warp-shuffle>`
     - Warp shuffle from original lane index xor mask
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`lane_mask <libcudacxx-extended-api-warp-lane-mask>`
     - Class to represent a mask of lanes in a warp
     - CCCL 3.1.0
     - CUDA 13.1
