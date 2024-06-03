#!/usr/bin/env bash

# Maybe change the UID/GID of the container's non-root user to match the host's UID/GID

: "${REMOTE_USER:="coder"}";
: "${OLD_UID:=}";
: "${OLD_GID:=}";
: "${NEW_UID:=}";
: "${NEW_GID:=}";

eval "$(sed -n "s/${REMOTE_USER}:[^:]*:\([^:]*\):\([^:]*\):[^:]*:\([^:]*\).*/OLD_UID=\1;OLD_GID=\2;HOME_FOLDER=\3/p" /etc/passwd)";
eval "$(sed -n "s/\([^:]*\):[^:]*:${NEW_UID}:.*/EXISTING_USER=\1/p" /etc/passwd)";
eval "$(sed -n "s/\([^:]*\):[^:]*:${NEW_GID}:.*/EXISTING_GROUP=\1/p" /etc/group)";

if [ -z "$OLD_UID" ]; then
    echo "Remote user not found in /etc/passwd ($REMOTE_USER).";
    exec "$(pwd)/.devcontainer/cccl-entrypoint.sh" "$@";
elif [ "$OLD_UID" = "$NEW_UID" ] && [ "$OLD_GID" = "$NEW_GID" ]; then
    echo "UIDs and GIDs are the same ($NEW_UID:$NEW_GID).";
    exec "$(pwd)/.devcontainer/cccl-entrypoint.sh" "$@";
elif [ "$OLD_UID" != "$NEW_UID" ] && [ -n "$EXISTING_USER" ]; then
    echo "User with UID exists ($EXISTING_USER=$NEW_UID).";
    exec "$(pwd)/.devcontainer/cccl-entrypoint.sh" "$@";
else
    if [ "$OLD_GID" != "$NEW_GID" ] && [ -n "$EXISTING_GROUP" ]; then
        echo "Group with GID exists ($EXISTING_GROUP=$NEW_GID).";
        NEW_GID="$OLD_GID";
    fi
    echo "Updating UID:GID from $OLD_UID:$OLD_GID to $NEW_UID:$NEW_GID.";
    sed -i -e "s/\(${REMOTE_USER}:[^:]*:\)[^:]*:[^:]*/\1${NEW_UID}:${NEW_GID}/" /etc/passwd;
    if [ "$OLD_GID" != "$NEW_GID" ]; then
        sed -i -e "s/\([^:]*:[^:]*:\)${OLD_GID}:/\1${NEW_GID}:/" /etc/group;
    fi

    # Fast parallel `chown -R`
    find "$HOME_FOLDER/" -not -user "$REMOTE_USER" -print0 \
  | xargs -0 -r -n1 -P"$(nproc --all)" chown "$NEW_UID:$NEW_GID"

    # shellcheck disable=SC2155
    # Create a list of the container startup environment variable names to pass to su
    declare -a _vars="($(env | grep '=' | grep -v '/root' | grep -Pv '^\s' | cut -d= -f1 | grep -Pv '^(.*HOME.*|PS1|_)$'))";
    # Run the container command as $REMOTE_USER, preserving the container startup environment
    exec su -s "$SHELL" -w "$(IFS=,; echo "${_vars[*]}")" - "$REMOTE_USER" -- "$(pwd)/.devcontainer/cccl-entrypoint.sh" "$@";
fi
