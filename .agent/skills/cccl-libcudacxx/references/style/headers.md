# Header conventions

## Include syntax

All inclusions must use angle-bracket form: `#include <header>`. No quoted includes.

## Self-sufficiency

Each file must include every header for every symbol it uses. Transitive inclusion is not
allowed — do not rely on a symbol being pulled in by another header you include.
Unneeded headers must be removed.

## Precision

Use the most precise header available. Prefer the internal single-symbol header over the
umbrella:

```cpp
#include <cuda/std/__type_traits/is_array.h>   // correct
#include <cuda/std/type_traits>                  // too broad
```

## Forward declarations

Prefer forward declarations over implementation includes when only the symbol's name is
needed. Use a `__fwd/` forwarding header when one exists, or declare the type directly:

```cpp
#include <cuda/std/__fwd/array.h>   // preferred over <cuda/std/array>
```

## Headers in `cuda/std/__cccl/`

Do not include headers from `cuda/std/__cccl/` directly. They are provided by
`<cuda/std/detail/__config>` or the prologue/epilogue mechanism.

## Required boilerplate — order

Every header must follow this structure in order:

```cpp
// 1. License block

// 2. Include guard
#ifndef _CUDA_STD_<UPPER_FULL_PATH>
#define _CUDA_STD_<UPPER_FULL_PATH>

// 3. Config header (immediately after guard)
#include <cuda/std/detail/__config>

// 4. System-header pragmas
#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// 5. Other includes

// 6. Prologue (last include before code)
#include <cuda/std/__cccl/prologue.h>

// ... file content ...

// 7. Epilogue (end of file)
#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD_<UPPER_FULL_PATH>
```

## Include guard naming

The guard name is the full path from the repo root, uppercased, with `/` and `.`
replaced by `_`:

- `libcudacxx/include/cuda/std/atomic` → `_CUDA_STD_ATOMIC`
- `libcudacxx/include/cuda/atomic` → `_CUDA_ATOMIC`

The closing `#endif` always carries a comment repeating the guard name:
```cpp
#endif // _CUDA_STD_ATOMIC
```

## License selection

| Directory                                                | License                               |
|------------------------------------------------------------|---------------------------------------|
| `libcudacxx/include/cuda/std/` (ported from LLVM libc++) | LLVM license                          |
| `libcudacxx/include/cuda/` (CCCL-only extensions)        | Apache License v2.0 with LLVM Exceptions |

License follows directory location, not content origin.
