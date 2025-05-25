#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 num_tries sleep_time command [args...]"
    echo "  num_tries: Number of attempts to run the command"
    echo "  sleep_time: Time to wait between attempts (in seconds)"
    echo "  command: The command to run"
    echo "  args: Arguments to pass to the command"
    exit 1
fi

num_tries=$1
sleep_time=$2
shift 2
command="$@"

# Loop until the command succeeds or we reach the maximum number of attempts:
for ((i=1; i<=num_tries; i++)); do
    echo "Attempt ${i} of ${num_tries}: Running command '${command}'"
    eval "$command"
    status=$?

    if [ $status -eq 0 ]; then
        echo "Command '${command}' succeeded on attempt ${i}."
        exit 0
    else
        echo "Command '${command}' failed with status ${status}. Retrying in ${sleep_time} seconds..."
        sleep $sleep_time
    fi
done
echo "Command '${command}' failed after ${num_tries} attempts."
exit 1
