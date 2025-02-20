#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

SKIP_INIT=false

echo "$@"

while true; do
    case "$1" in
        --skip-init)
            echo "Skipping initializing devcontainer"
            SKIP_INIT=true;
            shift 1;
            ;;
        *)
            break;
    esac
done

if ! $SKIP_INIT; then
    devcontainer-utils-post-create-command;
    devcontainer-utils-init-git;
    devcontainer-utils-post-attach-command;

    cd /home/coder/cccl/;
fi

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
