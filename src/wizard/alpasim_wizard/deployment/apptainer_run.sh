#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
# Generated deployments append their service commands to this lifecycle.
set -euo pipefail

declare -A pids=()

cleanup() {
    trap '' INT TERM
    local pid alive deadline=$((SECONDS + 10))
    for pid in "${pids[@]}"; do
        kill -TERM -- "$pid" "-$pid" 2>/dev/null || true
    done
    while (( SECONDS < deadline )); do
        alive=false
        for pid in "${pids[@]}"; do
            if kill -0 -- "$pid" 2>/dev/null || kill -0 -- "-$pid" 2>/dev/null; then alive=true; fi
        done
        if ! "$alive"; then break; fi
        sleep 0.1
    done
    for pid in "${pids[@]}"; do
        kill -KILL -- "$pid" "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch() {
    local name=$1 command=$2 log=$3
    echo "Starting $name (output -> $log)"
    setsid bash -euc "$command" >> "$log" 2>&1 &
    last_pid=$!
    pids[$name]=$last_pid
}

wait_for_port() {
    local name=$1 host=$2 port=$3
    while true; do
        if ! kill -0 "${pids[$name]}" 2>/dev/null; then
            echo "$name exited while starting; see txt-logs/out-$name-log.txt" >&2
            return 1
        fi
        if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then return; fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $name at $host:$port" >&2
            return 1
        fi
        sleep 0.1
    done
}

wait_for_services() {
    local pid
    while true; do
        for pid in "$@"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                return $?
            fi
        done
        sleep 0.1
    done
}
