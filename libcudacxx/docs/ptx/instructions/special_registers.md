# Special registers

-  PTX ISA: [`Special Registers`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers)

| C++ | PTX |
| [(0)](#0-get_sreg) `cuda::ptx::get_sreg_tid_x`| `mov.u32` |
| [(1)](#1-get_sreg) `cuda::ptx::get_sreg_tid_y`| `mov.u32` |
| [(2)](#2-get_sreg) `cuda::ptx::get_sreg_tid_z`| `mov.u32` |
| [(3)](#3-get_sreg) `cuda::ptx::get_sreg_ntid_x`| `mov.u32` |
| [(4)](#4-get_sreg) `cuda::ptx::get_sreg_ntid_y`| `mov.u32` |
| [(5)](#5-get_sreg) `cuda::ptx::get_sreg_ntid_z`| `mov.u32` |
| [(6)](#6-get_sreg) `cuda::ptx::get_sreg_laneid`| `mov.u32` |
| [(7)](#7-get_sreg) `cuda::ptx::get_sreg_warpid`| `mov.u32` |
| [(8)](#8-get_sreg) `cuda::ptx::get_sreg_nwarpid`| `mov.u32` |
| [(9)](#9-get_sreg) `cuda::ptx::get_sreg_ctaid_x`| `mov.u32` |
| [(10)](#10-get_sreg) `cuda::ptx::get_sreg_ctaid_y`| `mov.u32` |
| [(11)](#11-get_sreg) `cuda::ptx::get_sreg_ctaid_z`| `mov.u32` |
| [(12)](#12-get_sreg) `cuda::ptx::get_sreg_nctaid_x`| `mov.u32` |
| [(13)](#13-get_sreg) `cuda::ptx::get_sreg_nctaid_y`| `mov.u32` |
| [(14)](#14-get_sreg) `cuda::ptx::get_sreg_nctaid_z`| `mov.u32` |
| [(15)](#15-get_sreg) `cuda::ptx::get_sreg_smid`| `mov.u32` |
| [(16)](#16-get_sreg) `cuda::ptx::get_sreg_nsmid`| `mov.u32` |
| [(17)](#17-get_sreg) `cuda::ptx::get_sreg_gridid`| `mov.u64` |
| [(18)](#18-get_sreg) `cuda::ptx::get_sreg_is_explicit_cluster`| `mov.pred` |
| [(19)](#19-get_sreg) `cuda::ptx::get_sreg_clusterid_x`| `mov.u32` |
| [(20)](#20-get_sreg) `cuda::ptx::get_sreg_clusterid_y`| `mov.u32` |
| [(21)](#21-get_sreg) `cuda::ptx::get_sreg_clusterid_z`| `mov.u32` |
| [(22)](#22-get_sreg) `cuda::ptx::get_sreg_nclusterid_x`| `mov.u32` |
| [(23)](#23-get_sreg) `cuda::ptx::get_sreg_nclusterid_y`| `mov.u32` |
| [(24)](#24-get_sreg) `cuda::ptx::get_sreg_nclusterid_z`| `mov.u32` |
| [(25)](#25-get_sreg) `cuda::ptx::get_sreg_cluster_ctaid_x`| `mov.u32` |
| [(26)](#26-get_sreg) `cuda::ptx::get_sreg_cluster_ctaid_y`| `mov.u32` |
| [(27)](#27-get_sreg) `cuda::ptx::get_sreg_cluster_ctaid_z`| `mov.u32` |
| [(28)](#28-get_sreg) `cuda::ptx::get_sreg_cluster_nctaid_x`| `mov.u32` |
| [(29)](#29-get_sreg) `cuda::ptx::get_sreg_cluster_nctaid_y`| `mov.u32` |
| [(30)](#30-get_sreg) `cuda::ptx::get_sreg_cluster_nctaid_z`| `mov.u32` |
| [(31)](#31-get_sreg) `cuda::ptx::get_sreg_cluster_ctarank`| `mov.u32` |
| [(32)](#32-get_sreg) `cuda::ptx::get_sreg_cluster_nctarank`| `mov.u32` |
| [(33)](#33-get_sreg) `cuda::ptx::get_sreg_lanemask_eq`| `mov.u32` |
| [(34)](#34-get_sreg) `cuda::ptx::get_sreg_lanemask_le`| `mov.u32` |
| [(35)](#35-get_sreg) `cuda::ptx::get_sreg_lanemask_lt`| `mov.u32` |
| [(36)](#36-get_sreg) `cuda::ptx::get_sreg_lanemask_ge`| `mov.u32` |
| [(37)](#37-get_sreg) `cuda::ptx::get_sreg_lanemask_gt`| `mov.u32` |
| [(38)](#38-get_sreg) `cuda::ptx::get_sreg_clock`| `mov.u32` |
| [(39)](#39-get_sreg) `cuda::ptx::get_sreg_clock_hi`| `mov.u32` |
| [(40)](#40-get_sreg) `cuda::ptx::get_sreg_clock64`| `mov.u64` |
| [(41)](#41-get_sreg) `cuda::ptx::get_sreg_globaltimer`| `mov.u64` |
| [(42)](#42-get_sreg) `cuda::ptx::get_sreg_globaltimer_lo`| `mov.u32` |
| [(43)](#43-get_sreg) `cuda::ptx::get_sreg_globaltimer_hi`| `mov.u32` |
| [(44)](#44-get_sreg) `cuda::ptx::get_sreg_total_smem_size`| `mov.u32` |
| [(45)](#45-get_sreg) `cuda::ptx::get_sreg_aggr_smem_size`| `mov.u32` |
| [(46)](#46-get_sreg) `cuda::ptx::get_sreg_dynamic_smem_size`| `mov.u32` |
| [(47)](#47-get_sreg) `cuda::ptx::get_sreg_current_graph_exec`| `mov.u64` |


### [(0)](#0-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%tid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_tid_x();
```

### [(1)](#1-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%tid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_tid_y();
```

### [(2)](#2-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%tid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_tid_z();
```

### [(3)](#3-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%ntid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ntid_x();
```

### [(4)](#4-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%ntid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ntid_y();
```

### [(5)](#5-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%ntid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ntid_z();
```

### [(6)](#6-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%laneid; // PTX ISA 13
template <typename=void>
__device__ static inline uint32_t get_sreg_laneid();
```

### [(7)](#7-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%warpid; // PTX ISA 13
template <typename=void>
__device__ static inline uint32_t get_sreg_warpid();
```

### [(8)](#8-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nwarpid; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_nwarpid();
```

### [(9)](#9-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%ctaid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ctaid_x();
```

### [(10)](#10-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%ctaid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ctaid_y();
```

### [(11)](#11-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%ctaid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ctaid_z();
```

### [(12)](#12-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nctaid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_nctaid_x();
```

### [(13)](#13-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nctaid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_nctaid_y();
```

### [(14)](#14-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nctaid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_nctaid_z();
```

### [(15)](#15-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%smid; // PTX ISA 13
template <typename=void>
__device__ static inline uint32_t get_sreg_smid();
```

### [(16)](#16-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nsmid; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_nsmid();
```

### [(17)](#17-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u64 sreg_value, %%gridid; // PTX ISA 30
template <typename=void>
__device__ static inline uint64_t get_sreg_gridid();
```

### [(18)](#18-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.pred sreg_value, %%is_explicit_cluster; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool get_sreg_is_explicit_cluster();
```

### [(19)](#19-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%clusterid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_clusterid_x();
```

### [(20)](#20-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%clusterid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_clusterid_y();
```

### [(21)](#21-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%clusterid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_clusterid_z();
```

### [(22)](#22-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nclusterid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_nclusterid_x();
```

### [(23)](#23-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nclusterid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_nclusterid_y();
```

### [(24)](#24-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%nclusterid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_nclusterid_z();
```

### [(25)](#25-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_ctaid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_x();
```

### [(26)](#26-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_ctaid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_y();
```

### [(27)](#27-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_ctaid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_z();
```

### [(28)](#28-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_nctaid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_x();
```

### [(29)](#29-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_nctaid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_y();
```

### [(30)](#30-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_nctaid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_z();
```

### [(31)](#31-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_ctarank; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctarank();
```

### [(32)](#32-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%cluster_nctarank; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctarank();
```

### [(33)](#33-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%lanemask_eq; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_eq();
```

### [(34)](#34-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%lanemask_le; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_le();
```

### [(35)](#35-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%lanemask_lt; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_lt();
```

### [(36)](#36-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%lanemask_ge; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_ge();
```

### [(37)](#37-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%lanemask_gt; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_gt();
```

### [(38)](#38-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%clock; // PTX ISA 10
template <typename=void>
__device__ static inline uint32_t get_sreg_clock();
```

### [(39)](#39-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%clock_hi; // PTX ISA 50, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_clock_hi();
```

### [(40)](#40-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u64 sreg_value, %%clock64; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint64_t get_sreg_clock64();
```

### [(41)](#41-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u64 sreg_value, %%globaltimer; // PTX ISA 31, SM_35
template <typename=void>
__device__ static inline uint64_t get_sreg_globaltimer();
```

### [(42)](#42-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%globaltimer_lo; // PTX ISA 31, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_globaltimer_lo();
```

### [(43)](#43-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%globaltimer_hi; // PTX ISA 31, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_globaltimer_hi();
```

### [(44)](#44-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%total_smem_size; // PTX ISA 41, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_total_smem_size();
```

### [(45)](#45-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%aggr_smem_size; // PTX ISA 81, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_aggr_smem_size();
```

### [(46)](#46-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u32 sreg_value, %%dynamic_smem_size; // PTX ISA 41, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_dynamic_smem_size();
```

### [(47)](#47-get_sreg) `get_sreg`
{: .no_toc }
```cuda
// mov.u64 sreg_value, %%current_graph_exec; // PTX ISA 80, SM_50
template <typename=void>
__device__ static inline uint64_t get_sreg_current_graph_exec();
```
