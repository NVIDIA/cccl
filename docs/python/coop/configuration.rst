.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-configuration:

Configuration
=============

Installation
------------

.. code-block:: bash

   python -m pip install cuda-coop

The base package contains the shared API, Block and Warp Load/Store planning,
and the CCCL headers needed by compiler integrations. Backend integrations are
added separately. The base package has no Python package dependencies. Importing the base package does not require a CUDA device.
