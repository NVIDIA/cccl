..
   This file was automatically generated. Do not edit.

tid.x
^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%tid.x; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_tid_x();

tid.y
^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%tid.y; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_tid_y();

tid.z
^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%tid.z; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_tid_z();

ntid.x
^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%ntid.x; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_ntid_x();

ntid.y
^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%ntid.y; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_ntid_y();

ntid.z
^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%ntid.z; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_ntid_z();

laneid
^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%laneid; // PTX ISA 13
   template <typename = void>
   __device__ static inline uint32_t get_sreg_laneid();

warpid
^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%warpid; // PTX ISA 13
   template <typename = void>
   __device__ static inline uint32_t get_sreg_warpid();

nwarpid
^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nwarpid; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nwarpid();

ctaid.x
^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%ctaid.x; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_ctaid_x();

ctaid.y
^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%ctaid.y; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_ctaid_y();

ctaid.z
^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%ctaid.z; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_ctaid_z();

nctaid.x
^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nctaid.x; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nctaid_x();

nctaid.y
^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nctaid.y; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nctaid_y();

nctaid.z
^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nctaid.z; // PTX ISA 20
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nctaid_z();

smid
^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%smid; // PTX ISA 13
   template <typename = void>
   __device__ static inline uint32_t get_sreg_smid();

nsmid
^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nsmid; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nsmid();

gridid
^^^^^^
.. code-block:: cuda

   // mov.u64 sreg_value, %%gridid; // PTX ISA 30
   template <typename = void>
   __device__ static inline uint64_t get_sreg_gridid();

is_explicit_cluster
^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.pred sreg_value, %%is_explicit_cluster; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline bool get_sreg_is_explicit_cluster();

clusterid.x
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%clusterid.x; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_clusterid_x();

clusterid.y
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%clusterid.y; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_clusterid_y();

clusterid.z
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%clusterid.z; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_clusterid_z();

nclusterid.x
^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nclusterid.x; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nclusterid_x();

nclusterid.y
^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nclusterid.y; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nclusterid_y();

nclusterid.z
^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%nclusterid.z; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_nclusterid_z();

cluster_ctaid.x
^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_ctaid.x; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_ctaid_x();

cluster_ctaid.y
^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_ctaid.y; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_ctaid_y();

cluster_ctaid.z
^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_ctaid.z; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_ctaid_z();

cluster_nctaid.x
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_nctaid.x; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_nctaid_x();

cluster_nctaid.y
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_nctaid.y; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_nctaid_y();

cluster_nctaid.z
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_nctaid.z; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_nctaid_z();

cluster_ctarank
^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_ctarank; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_ctarank();

cluster_nctarank
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%cluster_nctarank; // PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_cluster_nctarank();

lanemask_eq
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%lanemask_eq; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_lanemask_eq();

lanemask_le
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%lanemask_le; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_lanemask_le();

lanemask_lt
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%lanemask_lt; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_lanemask_lt();

lanemask_ge
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%lanemask_ge; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_lanemask_ge();

lanemask_gt
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%lanemask_gt; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_lanemask_gt();

clock
^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%clock; // PTX ISA 10
   template <typename = void>
   __device__ static inline uint32_t get_sreg_clock();

clock_hi
^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%clock_hi; // PTX ISA 50, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_clock_hi();

clock64
^^^^^^^
.. code-block:: cuda

   // mov.u64 sreg_value, %%clock64; // PTX ISA 20, SM_35
   template <typename = void>
   __device__ static inline uint64_t get_sreg_clock64();

globaltimer
^^^^^^^^^^^
.. code-block:: cuda

   // mov.u64 sreg_value, %%globaltimer; // PTX ISA 31, SM_35
   template <typename = void>
   __device__ static inline uint64_t get_sreg_globaltimer();

globaltimer_lo
^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%globaltimer_lo; // PTX ISA 31, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_globaltimer_lo();

globaltimer_hi
^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%globaltimer_hi; // PTX ISA 31, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_globaltimer_hi();

total_smem_size
^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%total_smem_size; // PTX ISA 41, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_total_smem_size();

aggr_smem_size
^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%aggr_smem_size; // PTX ISA 81, SM_90
   template <typename = void>
   __device__ static inline uint32_t get_sreg_aggr_smem_size();

dynamic_smem_size
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u32 sreg_value, %%dynamic_smem_size; // PTX ISA 41, SM_35
   template <typename = void>
   __device__ static inline uint32_t get_sreg_dynamic_smem_size();

current_graph_exec
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mov.u64 sreg_value, %%current_graph_exec; // PTX ISA 80, SM_50
   template <typename = void>
   __device__ static inline uint64_t get_sreg_current_graph_exec();
