.. _cuda_cooperative-module:

CUDA Cooperative
================

The ``cuda.cooperative`` module exposes CUB's :ref:`cub:warp-module` and
:ref:`cub:block-module` for use within Numba Python CUDA kernels.

.. warning::
   This module is in public beta.  Not all CUB C++ CUDA kernel primitives
   have Python counterparts yet.  The API is subject to change without
   notice.

.. toctree::
   :maxdepth: 3

   developer_overview
   api

.. vim: set filetype=rst expandtab ts=8 sw=2 sts=2 tw=72 :
