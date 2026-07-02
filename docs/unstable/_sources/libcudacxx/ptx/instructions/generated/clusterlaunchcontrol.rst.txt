..
   This file was automatically generated. Do not edit.

clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [smem_bar]; // PTX ISA 86, SM_100
   template <typename = void>
   __device__ static inline void clusterlaunchcontrol_try_cancel(
     void* addr,
     uint64_t* smem_bar);

clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [addr], [smem_bar]; // PTX ISA 86, SM_100a, SM_110a
   template <typename = void>
   __device__ static inline void clusterlaunchcontrol_try_cancel_multicast(
     void* addr,
     uint64_t* smem_bar);

clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred_is_canceled, try_cancel_response; // PTX ISA 86, SM_100
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline bool clusterlaunchcontrol_query_cancel_is_canceled(
     B128 try_cancel_response);

clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret_dim, try_cancel_response; // PTX ISA 86, SM_100
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_x(
     B128 try_cancel_response);

clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 ret_dim, try_cancel_response; // PTX ISA 86, SM_100
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_y(
     B128 try_cancel_response);

clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 ret_dim, try_cancel_response; // PTX ISA 86, SM_100
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_z(
     B128 try_cancel_response);

clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 block_dim, try_cancel_response; // PTX ISA 86, SM_100
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void clusterlaunchcontrol_query_cancel_get_first_ctaid(
     B32 (&block_dim)[4],
     B128 try_cancel_response);
