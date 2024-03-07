# tensormap.replace

- PTX ISA: [`tensormap.replace`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)

| C++ | PTX |
| [(0)](#0-tensormap_replace) `cuda::ptx::tensormap_replace_global_address`| `tensormap.replace.tile.global_address.global.b1024.b64` |
| [(1)](#1-tensormap_replace) `cuda::ptx::tensormap_replace_global_address`| `tensormap.replace.tile.global_address.shared::cta.b1024.b64` |
| [(2)](#2-tensormap_replace) `cuda::ptx::tensormap_replace_rank`| `tensormap.replace.tile.rank.global.b1024.b32` |
| [(3)](#3-tensormap_replace) `cuda::ptx::tensormap_replace_rank`| `tensormap.replace.tile.rank.shared::cta.b1024.b32` |
| [(4)](#4-tensormap_replace) `cuda::ptx::tensormap_replace_box_dim`| `tensormap.replace.tile.box_dim.global.b1024.b32` |
| [(5)](#5-tensormap_replace) `cuda::ptx::tensormap_replace_box_dim`| `tensormap.replace.tile.box_dim.shared::cta.b1024.b32` |
| [(6)](#6-tensormap_replace) `cuda::ptx::tensormap_replace_global_dim`| `tensormap.replace.tile.global_dim.global.b1024.b32` |
| [(7)](#7-tensormap_replace) `cuda::ptx::tensormap_replace_global_dim`| `tensormap.replace.tile.global_dim.shared::cta.b1024.b32` |
| [(8)](#8-tensormap_replace) `cuda::ptx::tensormap_replace_global_stride`| `tensormap.replace.tile.global_stride.global.b1024.b64` |
| [(9)](#9-tensormap_replace) `cuda::ptx::tensormap_replace_global_stride`| `tensormap.replace.tile.global_stride.shared::cta.b1024.b64` |
| [(10)](#10-tensormap_replace) `cuda::ptx::tensormap_replace_element_size`| `tensormap.replace.tile.element_stride.global.b1024.b32` |
| [(11)](#11-tensormap_replace) `cuda::ptx::tensormap_replace_element_size`| `tensormap.replace.tile.element_stride.shared::cta.b1024.b32` |
| [(12)](#12-tensormap_replace) `cuda::ptx::tensormap_replace_elemtype`| `tensormap.replace.tile.elemtype.global.b1024.b32` |
| [(13)](#13-tensormap_replace) `cuda::ptx::tensormap_replace_elemtype`| `tensormap.replace.tile.elemtype.shared::cta.b1024.b32` |
| [(14)](#14-tensormap_replace) `cuda::ptx::tensormap_replace_interleave_layout`| `tensormap.replace.tile.interleave_layout.global.b1024.b32` |
| [(15)](#15-tensormap_replace) `cuda::ptx::tensormap_replace_interleave_layout`| `tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32` |
| [(16)](#16-tensormap_replace) `cuda::ptx::tensormap_replace_swizzle_mode`| `tensormap.replace.tile.swizzle_mode.global.b1024.b32` |
| [(17)](#17-tensormap_replace) `cuda::ptx::tensormap_replace_swizzle_mode`| `tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32` |
| [(18)](#18-tensormap_replace) `cuda::ptx::tensormap_replace_fill_mode`| `tensormap.replace.tile.fill_mode.global.b1024.b32` |
| [(19)](#19-tensormap_replace) `cuda::ptx::tensormap_replace_fill_mode`| `tensormap.replace.tile.fill_mode.shared::cta.b1024.b32` |


### [(0)](#0-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.global_address.space.b1024.b64    [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <typename B64>
__device__ static inline void tensormap_replace_global_address(
  cuda::ptx::space_global_t,
  void* tm_addr,
  B64 new_val);
```

### [(1)](#1-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.global_address.space.b1024.b64    [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <typename B64>
__device__ static inline void tensormap_replace_global_address(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  B64 new_val);
```

### [(2)](#2-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.rank.space.b1024.b32              [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <typename B32>
__device__ static inline void tensormap_replace_rank(
  cuda::ptx::space_global_t,
  void* tm_addr,
  B32 new_val);
```

### [(3)](#3-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.rank.space.b1024.b32              [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <typename B32>
__device__ static inline void tensormap_replace_rank(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  B32 new_val);
```

### [(4)](#4-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.box_dim.space.b1024.b32           [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_box_dim(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
```

### [(5)](#5-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.box_dim.space.b1024.b32           [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_box_dim(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
```

### [(6)](#6-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.global_dim.space.b1024.b32        [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_global_dim(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
```

### [(7)](#7-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.global_dim.space.b1024.b32        [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_global_dim(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
```

### [(8)](#8-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.global_stride.space.b1024.b64     [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B64>
__device__ static inline void tensormap_replace_global_stride(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B64 new_val);
```

### [(9)](#9-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.global_stride.space.b1024.b64     [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B64>
__device__ static inline void tensormap_replace_global_stride(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B64 new_val);
```

### [(10)](#10-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.element_stride.space.b1024.b32    [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_element_size(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
```

### [(11)](#11-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.element_stride.space.b1024.b32    [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_element_size(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
```

### [(12)](#12-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.elemtype.space.b1024.b32          [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_elemtype(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(13)](#13-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.elemtype.space.b1024.b32          [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_elemtype(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(14)](#14-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_interleave_layout(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(15)](#15-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_interleave_layout(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(16)](#16-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.swizzle_mode.space.b1024.b32      [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_mode(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(17)](#17-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.swizzle_mode.space.b1024.b32      [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_mode(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(18)](#18-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.fill_mode.space.b1024.b32         [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_fill_mode(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```

### [(19)](#19-tensormap_replace) `tensormap_replace`
{: .no_toc }
```cuda
// tensormap.replace.tile.fill_mode.space.b1024.b32         [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_fill_mode(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
```
