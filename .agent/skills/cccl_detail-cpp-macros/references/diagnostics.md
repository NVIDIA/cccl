# Diagnostics reference

## Push/pop pattern

Always pair push with pop. Nesting is allowed.

```cpp
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-parameter")
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-parameter")
_CCCL_DIAG_SUPPRESS_MSVC(4100)
// ... suppressed region ...
_CCCL_DIAG_POP
```

Per-compiler suppression macros are no-ops on other compilers — include all
relevant ones in the same block.

## Host-compiler suppress macros

| Macro                          | Active compiler | Argument                                   |
|--------------------------------|-----------------|-------------------------------------------|
| `_CCCL_DIAG_SUPPRESS_CLANG(W)` | Clang           | quoted warning flag, e.g. `"-Wshadow"`     |
| `_CCCL_DIAG_SUPPRESS_GCC(W)`   | GCC             | quoted warning flag, e.g. `"-Wdeprecated"` |
| `_CCCL_DIAG_SUPPRESS_NVHPC(W)` | NVHPC           | diagnostic name, e.g. `deprecated_entity`  |
| `_CCCL_DIAG_SUPPRESS_MSVC(C)`  | MSVC            | numeric code, e.g. `4996`                  |

## NVCC/NVRTC suppress macros

Use numeric diagnostic codes. Multiple codes accepted.

```cpp
_CCCL_BEGIN_NV_DIAG_SUPPRESS(20012)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(20012, 20013, 1444)
// ... suppressed region ...
_CCCL_END_NV_DIAG_SUPPRESS()
```

`_CCCL_NV_DIAG_PUSH()` / `_CCCL_NV_DIAG_POP()` are the lower-level forms
used when you need NVCC push/pop without immediately suppressing.
`_CCCL_DIAG_SUPPRESS_NVCC(N)` suppresses a single code without push/pop.

## Compound shortcuts

`_CCCL_SUPPRESS_DEPRECATED_PUSH` / `_CCCL_SUPPRESS_DEPRECATED_POP` — covers
all compilers in one call. Suppresses:
- Clang: `-Wdeprecated`, `-Wdeprecated-declarations`
- GCC: `-Wdeprecated`, `-Wdeprecated-declarations`
- NVHPC: `deprecated_entity`, `deprecated_entity_with_custom_message`
- MSVC: C4996
- NVCC: 1444, 20199

Use this to call deprecated CCCL APIs internally without leaking warnings:

```cpp
_CCCL_SUPPRESS_DEPRECATED_PUSH
old_function();
_CCCL_SUPPRESS_DEPRECATED_POP
```

## Common NVCC numeric codes

| Code  | Meaning                                         |
|-------|------------------------------------------------|
| 1444  | deprecated entity                               |
| 1675  | unrecognized `#pragma`                          |
| 20012 | `__host__` annotation on `__device__`-only function |
| 20013 | `__device__` annotation on `__host__`-only function |
| 20199 | deprecated API                                  |

## `_CCCL_WARNING`

Emits a portable compiler warning at the call site:

```cpp
_CCCL_WARNING("this overload is slow on MSVC")
```

Expands to `#pragma message` on MSVC, `#pragma GCC warning` elsewhere.

## Deprecation macros

Defined in `deprecated.h`, not `diagnostic.h`:

| Macro                         | Use                                  |
|-------------------------------|--------------------------------------|
| `CCCL_DEPRECATED`             | Mark a function or type deprecated   |
| `CCCL_DEPRECATED_BECAUSE(MSG)` | Deprecated with a custom message     |
| `_CCCL_DEPRECATED_IN_CXX20`   | Active only when `_CCCL_STD_VER >= 2020` |
| `_CCCL_DEPRECATED_IN_CXX23`   | Active only when `_CCCL_STD_VER >= 2023` |

Opt-out defines (user-facing): `CCCL_IGNORE_DEPRECATED_API`,
`CCCL_IGNORE_DEPRECATED_COMPILER`, `CCCL_IGNORE_DEPRECATED_CPP_DIALECT`.
These are public contract — do not remove.
