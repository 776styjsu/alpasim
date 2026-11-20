# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Forced-crash scenes for testing safety monitors.

Recorded scenes replay their actors without reacting to the ego, so a crash is
pure geometry: change the ego's path and it drives into whatever is there.
This package

* labels crashes in a ``rollout.asl`` -- what the ego hit and where on its body
  (``label``),
* replays a saved rollout through a log-driven monitor (``monitor_replay``),
* searches recorded scenes for ego maneuvers that end in a crash (``mine``),
* turns the chosen maneuvers into plan files (``plans``) that the scripted
  driver (``scripted_driver``) follows inside AlpaSim, and
* checks which planned crashes really happened under the controller and
  vehicle model (``confirm``).

See docs/CRASH_SCENES.md.
"""
