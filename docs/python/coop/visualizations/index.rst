.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualizations:

Visualizations
==============

Explore how ``cuda.coop`` moves values between memory and per-thread
payloads, converts layouts, shifts values between neighbors, reduces a
group's inputs to one aggregate, and computes ordered prefixes.
Change the settings, step through the stages, and select a value to follow
its ownership. These diagrams show data movement; their timing and geometry
do not predict GPU performance.

.. toctree::
   :maxdepth: 1

   load
   store
   exchange
   shuffle
   reduce
   scan
