#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

ci/rapids/post-create-command.sh;
rapids-post-start-command -f;

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
