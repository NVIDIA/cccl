.. _infra-devcontainer-launching:

Launching a container
=====================

``.devcontainer/launch.sh`` starts a CCCL development container with a chosen CUDA
toolkit and host compiler, mounts the repo, and either opens VSCode or drops you into
a shell. It is Linux-only (including WSL2). With no toolchain flags, it uses the default devcontainer
in ``.devcontainer/devcontainer.json``, which uses the latest CTK + gcc.

Launch in VSCode
----------------

**Open the default container.** Run the script with no flags.

.. code-block:: bash

   .devcontainer/launch.sh

The script copies the selected ``devcontainer.json`` into a temporary directory and
opens VSCode against it.

**Select a toolchain.** Pass ``--cuda`` and ``--host`` to open a specific variant.

.. code-block:: bash

   .devcontainer/launch.sh --cuda <cuda-version> --host <host-compiler>

The temporary-directory copy lets you run multiple variants of the same environment
side by side, each in its own VSCode window.

Launch directly in Docker
-------------------------

**Drop into a shell.** Add ``--docker`` to skip VSCode and run a bash shell inside the
container.

.. code-block:: bash

   .devcontainer/launch.sh --docker --cuda <cuda-version> --host <host-compiler>

The container mounts the repo at ``/home/coder/cccl`` and removes itself on exit. Any
trailing arguments after the flags run as a command instead of an interactive shell.

Specify the toolchain
---------------------

``--cuda`` selects the CUDA toolkit version. ``--host`` selects the host compiler.
The two flags resolve to ``.devcontainer/cuda<cuda>-<host>/devcontainer.json``; an
unknown combination exits with an error.

Valid values come from the devcontainers located under ``.devcontainer/``.

Pass through GPUs
-----------------

**Add host GPUs.** Pass ``--gpus all`` to expose every host GPU to the container.

.. code-block:: bash

   .devcontainer/launch.sh --docker --cuda <cuda-version> --host <host-compiler> --gpus all

``--gpus`` takes any Docker GPU request string. It overrides the ``hostRequirements.gpu``
default read from the devcontainer config. Without it, the container starts without GPU
access, which is sufficient for building tests.

Launch from a git worktree
--------------------------

``launch.sh`` handles linked worktrees automatically. A worktree's ``.git`` is a file
pointing at the main repository's git directory, which the container cannot reach through
the worktree mount alone. The script bind-mounts the main repo's git common directory at
its host path so git operations resolve inside the container.

The ``cccl-build`` and ``cccl-wheelhouse`` Docker volumes are shared across all worktrees
and the main checkout, so build artifacts collide between them. Do not run multiple
worktree containers concurrently unless you are careful to avoid conflicts.

First-time git auth in a worktree needs the main checkout's ``.config/gh``. If the main
checkout has never run its container, ``launch.sh`` warns that startup will block on an
interactive ``gh auth login``. Launch the main checkout's container once, complete the
login, then re-launch the worktree.

Forward SSH keys
----------------

If ``SSH_AUTH_SOCK`` is set in your environment, ``launch.sh`` forwards the agent socket
into the container automatically. No flag is required. Git operations over SSH inside the
container use your host agent's keys.
