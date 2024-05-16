#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

. devcontainer-utils-post-create-command;
. devcontainer-utils-post-start-command;
. /home/coder/cccl/ci/rapids/post-create-command.sh;
. rapids-post-start-command;
. devcontainer-utils-post-attach-command;

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
