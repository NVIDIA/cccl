.. _libcudacxx-extended-api-macros:

======
Macros
======

CCCL provides a set of convenience macros for detecting various system and compile-time
properties via the preprocessor. These macros are available when any CCCL header is
included, and do not require including a specific header file.

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Macro**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - .. toctree::
          :maxdepth: 1

          ../api/macro_cccl_os
     - Detecting the current operating system.
     - CCCL 3.4.0
     - CUDA 13.3
