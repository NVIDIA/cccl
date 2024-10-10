#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

devcontainer-utils-post-create-command;
devcontainer-utils-init-git;
devcontainer-utils-post-attach-command;

# cd /home/coder/cccl/
echo "CLANG .."
clang-format -i cudax/include/cuda/experimental/__stf/graph/graph_ctx.cuh
clang-format --version

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
