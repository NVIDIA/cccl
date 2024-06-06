.. _device-module:

Device-Wide Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   ${repo_docs_api_path}/struct*Device*


CUB device-level single-problem parallel algorithms:

* :cpp:class:`cub::DeviceAdjacentDifference <cub::DeviceAdjacentDifference>` computes the difference between adjacent elements residing within device-accessible memory
* :cpp:struct:`cub::DeviceFor <cub::DeviceFor>` provides device-wide, parallel operations for iterating over data residing within device-accessible memory
* :cpp:class:`cub::DeviceHistogram <cub::DeviceHistogram>` constructs histograms from data samples residing within device-accessible memory
* :cpp:struct:`cub::DevicePartition <cub::DevicePartition>` partitions data residing within device-accessible memory
* :cpp:struct:`cub::DeviceMerge <cub::DeviceMerge>` merges two sorted sequences in device-accessible memory into a single one
* :cpp:class:`cub::DeviceMergeSort <cub::DeviceMergeSort>` sorts items residing within device-accessible memory
* :cpp:class:`cub::DeviceRadixSort <cub::DeviceRadixSort>` sorts items residing within device-accessible memory using radix sorting method
* :cpp:struct:`cub::DeviceReduce <cub::DeviceReduce>` computes reduction of items residing within device-accessible memory
* :cpp:class:`cub::DeviceRunLengthEncode <cub::DeviceRunLengthEncode>` demarcating "runs" of same-valued items withing a sequence residing within device-accessible memory
* :cpp:struct:`cub::DeviceScan <cub::DeviceScan>` computes a prefix scan across a sequence of data items residing within device-accessible memory
* :cpp:struct:`cub::DeviceSelect <cub::DeviceSelect>` compacts data residing within device-accessible memory


CUB device-level segmented-problem (batched) parallel algorithms:

* :cpp:struct:`cub::DeviceSegmentedSort <cub::DeviceSegmentedSort>` computes batched sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceSegmentedRadixSort <cub::DeviceSegmentedRadixSort>` computes batched radix sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceSegmentedReduce <cub::DeviceSegmentedReduce>` computes reductions across multiple sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceCopy <cub::DeviceCopy>` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceMemcpy <cub::DeviceMemcpy>` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
