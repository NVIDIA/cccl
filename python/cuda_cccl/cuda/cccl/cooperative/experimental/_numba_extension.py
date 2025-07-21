"""
Initialise the cuda-cccl cooperative plug-in for Numba.

This function is discovered by Numba through the
[project.entry-points."numba_extensions"] group in pyproject.toml.
It is executed exactly once, **before the first compilation** triggered
by `numba.njit`/`cuda.jit`.
"""

import os

NUMBA_CCCL_COOP_DEBUG = False
NUMBA_CCCL_COOP_INJECT_PRINTFS = False
NUMBA_CCCL_COOP_SOURCE_CODE_REWRITER = None


def _set_source_code_rewriter(rewriter) -> None:
    global NUMBA_CCCL_COOP_SOURCE_CODE_REWRITER
    NUMBA_CCCL_COOP_SOURCE_CODE_REWRITER = rewriter


def _get_source_code_rewriter():
    global NUMBA_CCCL_COOP_SOURCE_CODE_REWRITER
    return NUMBA_CCCL_COOP_SOURCE_CODE_REWRITER


def _get_env_boolean(name: str, default: bool = False) -> bool:
    """
    Helper function to get an environment variable as a boolean.
    """
    val = os.environ.get(name, None)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _init_extension() -> None:
    # import cuda.cccl.cooperative.experimental._rewrite
    from ._rewrite import _init_rewriter

    _init_rewriter()

    # Is this the idiomatic way of registering a Numba extension?!
    from numba.cuda.target import CUDATypingContext

    from ._decls import registry

    if not hasattr(CUDATypingContext, "_cccl_patched"):
        _orig = CUDATypingContext.load_additional_registries

        def _patched(self):
            _orig(self)
            self.install_registry(registry)

        CUDATypingContext.load_additional_registries = _patched
        CUDATypingContext._cccl_patched = True

    from numba.cuda.cudadrv.driver import driver

    if hasattr(driver, "target_context"):
        driver.target_context.install_registry(registry)

    global NUMBA_CCCL_COOP_DEBUG
    global NUMBA_CCCL_COOP_INJECT_PRINTFS

    NUMBA_CCCL_COOP_DEBUG = _get_env_boolean("NUMBA_CCCL_COOP_DEBUG")
    NUMBA_CCCL_COOP_INJECT_PRINTFS = _get_env_boolean("NUMBA_CCCL_COOP_INJECT_PRINTFS")
    if NUMBA_CCCL_COOP_DEBUG:
        msg = "cuda.cccl.cooperative Numba extension initialized."
        print(msg)
