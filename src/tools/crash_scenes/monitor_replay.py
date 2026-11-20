# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Run the two-second monitor over a saved rollout instead of a live one.

The monitor only consumes log entries, in the order AlpaSim broadcasts them.
``rollout.asl`` stores the same entries in the same order, so feeding the file
to the monitor reproduces what it would have written during the run.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from types import ModuleType

from alpasim_utils.logs import async_read_pb_log
from trajdata.maps import VectorMap

logger = logging.getLogger(__name__)

ALERTS_FILENAME = "two_second_alerts.jsonl"


def load_monitor_module(path: str) -> ModuleType:
    """Import ``two_second_monitor.py`` from where its owner keeps it."""
    spec = importlib.util.spec_from_file_location("two_second_monitor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import the monitor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def replay_two_second_monitor(
    monitor_module: ModuleType,
    asl_path: str,
    output_dir: str,
    vector_map: VectorMap | None,
) -> dict:
    """Replay ``asl_path`` through the monitor; alerts land in ``output_dir``.

    A live run logs a monitor error and carries on, so the replay does the
    same and reports how many entries failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    alerts_path = os.path.join(output_dir, ALERTS_FILENAME)
    if os.path.exists(alerts_path):
        os.remove(alerts_path)  # the monitor appends

    monitor = monitor_module.TwoSecondMonitor(output_dir, vector_map=vector_map)
    entries = 0
    failures = 0
    async for entry in async_read_pb_log(asl_path, raise_on_malformed=True):
        entries += 1
        try:
            monitor.handle(entry)
        except Exception:
            failures += 1
            if failures == 1:
                logger.exception("Monitor failed on a log entry of %s", asl_path)

    return {
        "asl_path": asl_path,
        "alerts_path": alerts_path,
        "log_entries": entries,
        "monitor_failures": failures,
        "wrote_alerts": os.path.exists(alerts_path),
    }
