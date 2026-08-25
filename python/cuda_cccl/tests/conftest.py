import faulthandler
import os
from pathlib import Path

import pytest

# pytest-xdist workers talk to the controller over a channel and their stderr is
# never relayed. A faulthandler traceback written to stderr therefore dies with
# the worker, and the controller reports only "node down: Not properly
# terminated". Point faulthandler at a per-worker file so a native crash leaves
# evidence behind that CI can print after the run.
#
# Opt-in via CCCL_FAULTHANDLER_DIR so ordinary local runs write nothing.
_faulthandler_file = None


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    # trylast: pytest's own faulthandler plugin enables faulthandler against its
    # stderr during pytest_configure. Registering last means this redirect wins.
    global _faulthandler_file

    dump_dir = os.environ.get("CCCL_FAULTHANDLER_DIR")
    if not dump_dir:
        return

    # Only workers need this. The controller's stderr reaches the log already, so
    # pytest's own faulthandler covers it and we avoid leaving a file open there.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return

    path = Path(dump_dir) / f"faulthandler-{worker}-{os.getpid()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Held open for the process lifetime: faulthandler writes to this descriptor
    # from a fatal-signal handler, long after any close() would have run.
    _faulthandler_file = path.open("w")
    faulthandler.enable(file=_faulthandler_file)
