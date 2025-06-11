"""
Initialise the cuda-cccl cooperative plug-in for Numba.

This function is discovered by Numba through the
[project.entry-points."numba_extensions"] group in pyproject.toml.
It is executed exactly once, **before the first compilation** triggered
by `numba.njit`/`cuda.jit`.
"""


def _init_extension() -> None:
    # Is this the idiomatic way of registering a Numba extension?!
    from numba.cuda.target import CUDATypingContext

    from cuda.cccl.cooperative.experimental._decls import registry

    if not hasattr(CUDATypingContext, "_cccl_patched"):
        _orig = CUDATypingContext.load_additional_registries

        def _patched(self):
            _orig(self)
            self.install_registry(registry)

        CUDATypingContext.load_additional_registries = _patched
        CUDATypingContext._cccl_patched = True

    try:
        from numba.cuda.cudadrv.driver import driver

        if hasattr(driver, "target_context"):
            driver.target_context.install_registry(reg)
    except Exception:
        pass

    import os

    val = os.environ.get("NUMBA_CCCL_COOP_DEBUG", "0")
    NUMBA_CCCL_COOP_DEBUG = (val.lower() in ("1", "true", "yes")) if val else False
    if NUMBA_CCCL_COOP_DEBUG or True:
        msg = "cuda.cccl.cooperative Numba extension initialized."
        print(msg)
