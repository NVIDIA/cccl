.. _cudax-coop:

CCCL Cooperative
================

CCCL Cooperative is a set of device-side APIs for organizing groups of collaborating threads and executing collective algorithms. They are placed in the ``cuda::experimental::coop`` namespace and available through ``<cuda/experimental/coop/_header_>`` include files.

These APIs are successors to both CTK Cooperative Groups (``<cooperative_groups.h>``) and CUB thread-, warp-, and block-wide collective algorithms. The goal is to provide a more capable API that provides better flexibility, performance, and safety.

.. toctree::
  :maxdepth: 1

  group/index
  algorithm/index

.. warning:: The CCCL Cooperative Groups implementation requires some experimental code to be enabled in libcu++ by defining ``_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX`` before including any CCCL header.
