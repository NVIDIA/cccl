# Architecture flag forms

The `-arch` flag maps to CMake `CUDA_ARCHITECTURES`. Value is a semicolon-separated list.

| Form             | Generates               | Notes                          |
|------------------|-------------------------|--------------------------------|
| `<XX>`           | PTX + SASS for SM XX    | e.g. `80`                      |
| `<XX-real>`      | SASS only               | smaller binary, no JIT         |
| `<XX-virtual>`   | PTX only                | JIT at runtime                 |
| `native`         | Detect host GPU         | fastest for local iteration    |
| `all-major-cccl` | Default for PR builds   | slowest; use only when needed  |
| `all-cccl`       | A very frustrated user  | Just don't                     |

Examples:

```
-arch "native"
-arch "80"
-arch "70;75;80-virtual"
```

Limiting `-arch` to `native` or a single SM is the single biggest build-time lever.
