#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run skill_runner while logging human time/position/room with dual-clock support.

Dual-clock system:
- wall_clock: realistic household timestamps for the trace.
- sim_time: simulator elapsed time rendered in wall-equivalent seconds.

Important timing rule:
- For normal simulator execution, 1 simulator/world second represents
  SIM_TIME_SCALE real-world wall seconds.
- For long Wait skills clipped before simulator execution, wall time is
  interpolated across the wall-clock interval allocated to that wait while
  pose/room remain fixed by the simulator. If the callback misses the exact
  clipped boundary, the wrapper force-writes the final wait row.

Usage:
python -m habitat_llm.examples.skill_runner_human_trace --config-name examples/skill_runner_decentralized_config.yaml \
    +skill_runner_commands_file=skill_runner_commands_mapped.txt
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
import sys
from typing import Any, Dict, Optional, Union

import habitat_llm.examples.skill_runner as base_skill_runner
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.examples.example_utils import execute_skill as base_execute_skill
from habitat_llm.examples.skill_activity_parser import (
    load_activity_blocks,
    CLIP_THRESHOLD,
)

TRACE_FILE_ENV_VAR = "SKILL_RUNNER_HUMAN_TRACE_FILE"
TRACE_TIME_SCALE_ENV_VAR = "SKILL_RUNNER_TRACE_TIME_SCALE"
DEFAULT_TRACE_FILE = "human_room_trace.txt"
TRACE_START_CLOCK = datetime(2026, 1, 5, 0, 0, 0)  # Monday
UNKNOWN_ACTIVITY = "unknown"
DEFAULT_SIM_TIME_SCALE = 4.0  # 1 simulator second = 4 real-world seconds
# Wall-time reservation for skills that appear after a long wait.
# These are only used to prevent a middle-of-block long wait from consuming
# the entire activity block.
MIN_NON_WAIT_WALL_SECONDS = 4.0
MIN_SHORT_WAIT_WALL_SECONDS = 1.0



def _resolve_sim_time_scale() -> float:
    """Resolve simulator-to-wall-clock scale.

    The default matches the current project assumption: 1 simulator second
    corresponds to 4 real-world seconds. An environment override is supported
    for experiments without editing this file.
    """
    value = os.environ.get(TRACE_TIME_SCALE_ENV_VAR)
    if not value:
        return DEFAULT_SIM_TIME_SCALE
    try:
        parsed = float(value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return DEFAULT_SIM_TIME_SCALE


SIM_TIME_SCALE = _resolve_sim_time_scale()


def _new_dual_clock_state() -> dict[str, Any]:
    return {
        "blocks": [],
        "commands_file": None,
        "week_start": TRACE_START_CLOCK,
        "next_block_idx": 0,
        "next_skill_idx": 0,
        "wall_cursor": TRACE_START_CLOCK,
        "active_block_idx": None,
        "active_skill_idx": None,
        "active_wall_start": None,
        "active_world_time_start": None,
        "active_last_elapsed_second": 0,
        "active_last_wall_clock": None,
        "long_wait_start_written": set(),
        "long_wait_end_written": set(),
        "active_block_expired": False,
        "last_room_pose": None,
        "last_total_elapsed_seconds": 0.0,
    }


_DUAL_CLOCK_STATE: dict[str, Any] = _new_dual_clock_state()

_TRACE_TIME_STATE: dict[str, Optional[Union[int, float, tuple]]] = {
    "sim_time_start": None,
    "last_logged_key": None,
    "last_written_trace_key": None,
}


def _reset_trace_state() -> None:
    """Reset all mutable module state for a fresh run in this Python process."""
    global _DUAL_CLOCK_STATE
    _DUAL_CLOCK_STATE = _new_dual_clock_state()
    _TRACE_TIME_STATE["sim_time_start"] = None
    _TRACE_TIME_STATE["last_logged_key"] = None
    _TRACE_TIME_STATE["last_written_trace_key"] = None


def _resolve_commands_file_from_argv() -> Optional[Path]:
    for arg in sys.argv:
        if arg.startswith("+skill_runner_commands_file="):
            value = arg.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return Path(value).expanduser()
    return None


def _resolve_trace_file_path() -> Path:
    env_override = os.environ.get(TRACE_FILE_ENV_VAR)
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path(Path.cwd(), DEFAULT_TRACE_FILE)


def _ensure_blocks_loaded(commands_file: Optional[Path]) -> None:
    """Load/reload schedule blocks and initialize the schedule cursor."""
    if commands_file is None:
        return

    resolved = commands_file.resolve()
    if _DUAL_CLOCK_STATE["commands_file"] == resolved and _DUAL_CLOCK_STATE["blocks"]:
        return

    blocks = sorted(load_activity_blocks(commands_file), key=lambda b: b.wall_start)
    _DUAL_CLOCK_STATE.update(
        {
            "blocks": blocks,
            "commands_file": resolved,
            "week_start": TRACE_START_CLOCK,
            "next_block_idx": 0,
            "next_skill_idx": 0,
            "wall_cursor": blocks[0].wall_start if blocks else TRACE_START_CLOCK,
            "active_block_idx": None,
            "active_skill_idx": None,
            "active_wall_start": None,
            "active_world_time_start": None,
            "active_last_elapsed_second": 0,
            "active_last_wall_clock": None,
            "long_wait_start_written": set(),
            "long_wait_end_written": set(),
            "active_block_expired": False,
            "last_room_pose": None,
            "last_total_elapsed_seconds": 0.0,
        }
    )


def _resolve_room_and_position(
    env_interface: EnvironmentInterface,
) -> tuple[str, float, float, float]:
    try:
        human_pos = list(env_interface.sim.agents_mgr[1].articulated_agent.base_pos)
        region_id_to_name = {}
        try:
            region_id_to_name = dict(getattr(env_interface.perception, "region_id_to_name", {}))
        except Exception:
            region_id_to_name = {}

        region_names: dict[str, int] = {}
        for region in env_interface.sim.semantic_scene.regions:
            region_key = region.category.name().split("/")[0].replace(" ", "_")
            region_names[region_key] = region_names.get(region_key, 0) + 1
            room_name = f"{region_key}_{region_names[region_key]}"
            if region.contains(human_pos):
                mapped_room_name = region_id_to_name.get(region.id)
                if mapped_room_name:
                    room_name = str(mapped_room_name)
                return room_name, float(human_pos[0]), float(human_pos[1]), float(human_pos[2])
    except Exception:
        pass

    try:
        human_wg = env_interface.world_graph[1]
        human_node = human_wg.get_human()
        room_node = human_wg.get_room_for_entity(human_node)
        human_pos = list(env_interface.sim.agents_mgr[1].articulated_agent.base_pos)
        return str(room_node.name), float(human_pos[0]), float(human_pos[1]), float(human_pos[2])
    except Exception:
        return UNKNOWN_ACTIVITY, float("nan"), float("nan"), float("nan")


def _write_trace_line(
    trace_file: Path,
    wall_day_str: str,
    wall_time_str: str,
    sim_time_str: str,
    room_name: str,
    xyz: tuple[float, float, float],
    activity: str,
) -> None:
    """Write one trace line.

    Coordinate order is x z y_world, matching the 2D floor-plane convention
    while preserving the vertical world coordinate as y_world.
    """
    x, y_world, z = xyz
    line = (
        f"{wall_day_str} {wall_time_str} {sim_time_str} "
        f"{x:.3f} {z:.3f} {y_world:.3f} {room_name} {activity}\n"
    )
    with trace_file.open("a", encoding="utf-8") as f:
        f.write(line)

def _current_block_has_expired(wall_clock: datetime, block: Any) -> bool:
    """Return True if the computed wall time has reached/passed this block end."""
    return wall_clock >= block.wall_end


def _advance_cursor_to_next_block_after_expiry(block_idx: int) -> None:
    """Move schedule cursor to the next block after the active block expires.

    This does not cancel the simulator skill. It only tells the trace scheduler
    that any remaining commands from the expired block should no longer be
    logged under that expired activity.
    """
    blocks = _DUAL_CLOCK_STATE["blocks"]
    next_block_idx = block_idx + 1

    if next_block_idx < len(blocks):
        _DUAL_CLOCK_STATE["next_block_idx"] = next_block_idx
        _DUAL_CLOCK_STATE["next_skill_idx"] = 0
        _DUAL_CLOCK_STATE["wall_cursor"] = blocks[next_block_idx].wall_start
    else:
        _DUAL_CLOCK_STATE["next_block_idx"] = next_block_idx
        _DUAL_CLOCK_STATE["next_skill_idx"] = 0
        _DUAL_CLOCK_STATE["wall_cursor"] = blocks[block_idx].wall_end

    _DUAL_CLOCK_STATE["active_block_idx"] = None
    _DUAL_CLOCK_STATE["active_skill_idx"] = None
    _DUAL_CLOCK_STATE["active_wall_start"] = None
    _DUAL_CLOCK_STATE["active_world_time_start"] = None
    _DUAL_CLOCK_STATE["active_last_elapsed_second"] = 0
    _DUAL_CLOCK_STATE["active_last_wall_clock"] = None
    _DUAL_CLOCK_STATE["active_block_expired"] = False

def _clamp_wall_clock_to_block(wall_clock: datetime, block: Any) -> datetime:
    """Keep logged wall time inside the active activity block."""
    if wall_clock < block.wall_start:
        return block.wall_start
    if wall_clock > block.wall_end:
        return block.wall_end
    return wall_clock

def _format_hms(total_seconds: float) -> str:
    total = max(0, int(total_seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _estimate_skill_wall_seconds(skill: Any) -> float:
    """Estimate how much wall-clock time to reserve for a future skill.

    This is only used for skills after an active long wait, so that a long wait
    in the middle of a block does not stretch all the way to block.wall_end.
    """
    if getattr(skill, "is_wait", False):
        wait_duration = getattr(skill, "wait_duration", None)

        if wait_duration is None:
            return MIN_SHORT_WAIT_WALL_SECONDS

        if getattr(skill, "is_long_wait", False):
            # Another future long wait needs enough trace time to execute its
            # clipped simulator wait. This is a conservative reservation.
            return max(CLIP_THRESHOLD * SIM_TIME_SCALE, MIN_SHORT_WAIT_WALL_SECONDS)

        # Short waits should keep their normal scaled wall duration.
        return max(float(wait_duration) * SIM_TIME_SCALE, MIN_SHORT_WAIT_WALL_SECONDS)

    # Future Navigate/Open/Pick/Place/Close skills need some wall time too.
    # Navigate usually gets real callback timing; this is just a reservation
    # so earlier long waits leave space for it.
    return MIN_NON_WAIT_WALL_SECONDS


def _estimate_remaining_skill_wall_seconds(block: Any, after_skill_idx: int) -> float:
    """Estimate wall time needed by skills after after_skill_idx in the same block."""
    remaining = 0.0

    for future_skill in block.skills[after_skill_idx + 1:]:
        remaining += _estimate_skill_wall_seconds(future_skill)

    return remaining


def _compute_active_long_wait_wall_end(
    block: Any,
    skill_idx: int,
    active_wall_start: datetime,
) -> datetime:
    """Return the wall-clock end allocated to the active long wait.

    If the long wait is the last skill in the block, it ends at block.wall_end.
    If there are skills after it, reserve time for them and end this wait earlier.
    """
    block_end = block.wall_end

    # Last skill: keep the old behavior.
    if skill_idx >= len(block.skills) - 1:
        return block_end

    reserved_after_seconds = _estimate_remaining_skill_wall_seconds(block, skill_idx)
    target_end = block_end - timedelta(seconds=reserved_after_seconds)

    # Do not allow the target end to go before the wait starts.
    if target_end <= active_wall_start:
        return active_wall_start

    return target_end

def _skill_type_name(skill: Any) -> str:
    return str(getattr(skill, "skill_type", ""))


def _action_type_name(action: Any) -> Optional[str]:
    if not action or len(action) < 1:
        return None
    return str(action[0])


def _extract_primary_action(high_level_skill_actions: Optional[Dict[int, Any]]) -> Any:
    """Return the action most likely to correspond to the human schedule.

    Habitat examples commonly use agent id 1 for the human. If id 1 is not
    present, fall back to the first action to preserve previous behavior.
    """
    if not high_level_skill_actions:
        return None
    if 1 in high_level_skill_actions:
        return high_level_skill_actions[1]
    for _, action in high_level_skill_actions.items():
        return action
    return None


def _extract_primary_agent_uid(high_level_skill_actions: Optional[Dict[int, Any]]) -> Optional[int]:
    """Return the agent uid for the primary action."""
    if not high_level_skill_actions:
        return None
    if 1 in high_level_skill_actions:
        return 1
    for agent_uid in high_level_skill_actions.keys():
        return int(agent_uid)
    return None


def _normalize_action_args(action: Any) -> list[str]:
    """Convert simulator action arguments to comparable strings."""
    if not action or len(action) <= 1:
        return []
    return [str(v) for v in list(action[1:])]


def _normalize_skill_args(skill: Any) -> list[str]:
    """Convert parsed skill arguments to comparable strings."""
    return [str(v) for v in getattr(skill, "args", [])]


def _last_float(values: list[str]) -> Optional[float]:
    """Return the last value that can be parsed as float.

    This handles both:
    - ["1500"]
    - ["1", "1500"]
    """
    for value in reversed(values):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _wait_args_equal(action_args: list[str], skill_args: list[str], skill: Any) -> bool:
    """Compare Wait duration arguments robustly.

    Important:
    A long wait may reach this function either as its original scheduled
    duration, e.g. Wait 1500, or already clipped, e.g. Wait 20.
    Both must match the same parsed scheduled long-wait skill.
    """
    action_duration = _last_float(action_args)
    if action_duration is None:
        return False

    scheduled_duration = getattr(skill, "wait_duration", None)
    if scheduled_duration is None:
        scheduled_duration = _last_float(skill_args)

    if scheduled_duration is None:
        return False

    try:
        scheduled_duration = float(scheduled_duration)
    except (TypeError, ValueError):
        return False

    # Normal/original match: Wait 1500 == scheduled Wait 1500.
    if abs(action_duration - scheduled_duration) < 1e-6:
        return True

    # Clipped match: runtime Wait 20 should still match scheduled Wait 1500.
    if getattr(skill, "is_long_wait", False):
        clipped_duration = getattr(skill, "clipped_duration", None)
        if clipped_duration is None:
            clipped_duration = min(scheduled_duration, CLIP_THRESHOLD)

        try:
            clipped_duration = float(clipped_duration)
        except (TypeError, ValueError):
            clipped_duration = float(CLIP_THRESHOLD)

        if abs(action_duration - clipped_duration) < 1e-6:
            return True

        if abs(action_duration - float(CLIP_THRESHOLD)) < 1e-6:
            return True

    return False


def _skill_matches_action(skill: Any, action: Any, agent_uid: Optional[int]) -> bool:
    """Return True if a parsed scheduled skill matches the raw simulator action.

    Important:
    - This must be called before Wait clipping.
    - Wait 940 must match the parsed Wait 940, not the clipped Wait 20.
    - Matching only by skill type is unsafe because many blocks contain Wait,
      Navigate, Pick, Place, etc.
    """
    if not action or len(action) < 1:
        return False

    if agent_uid is not None:
        try:
            if int(getattr(skill, "agent_uid", -1)) != int(agent_uid):
                return False
        except (TypeError, ValueError):
            return False

    action_type = str(action[0])
    skill_type = _skill_type_name(skill)

    if skill_type != action_type:
        return False

    action_args = _normalize_action_args(action)
    skill_args = _normalize_skill_args(skill)

    if action_type == "Wait":
        return _wait_args_equal(action_args, skill_args, skill)

    return action_args == skill_args


def _action_matches_skill_type_and_agent(
    skill: Any,
    action: Any,
    agent_uid: Optional[int],
) -> bool:
    if not action or len(action) < 1:
        return False

    if agent_uid is not None:
        try:
            if int(getattr(skill, "agent_uid", -1)) != int(agent_uid):
                return False
        except (TypeError, ValueError):
            return False

    return _skill_type_name(skill) == str(action[0])


def _select_next_scheduled_skill(
    action: Any,
    agent_uid: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """Select the next scheduled skill.

    Key behavior:
    - Wait must match original duration before clipping.
    - Non-wait skills match sequentially by skill type + agent.
    - Do not use exact arg matching for Navigate/Pick/Place/Open/Close because
      runtime args may differ from command-file text.
    """
    blocks = _DUAL_CLOCK_STATE["blocks"]
    if not blocks or not action:
        return None, None

    start_block = int(_DUAL_CLOCK_STATE["next_block_idx"])
    start_skill = int(_DUAL_CLOCK_STATE["next_skill_idx"])
    action_type = str(action[0])

    # 1. Prefer the current cursor.
    if start_block < len(blocks):
        block = blocks[start_block]
        if start_skill < len(block.skills):
            skill = block.skills[start_skill]

            if action_type == "Wait":
                if _skill_matches_action(skill, action, agent_uid):
                    return start_block, start_skill
            else:
                if _action_matches_skill_type_and_agent(skill, action, agent_uid):
                    return start_block, start_skill

    # 2. For Wait, scan forward for exact original duration match.
    if action_type == "Wait" and start_block < len(blocks):
        block = blocks[start_block]
        for skill_idx in range(start_skill, len(block.skills)):
            skill = block.skills[skill_idx]
            if _skill_matches_action(skill, action, agent_uid):
                return start_block, skill_idx

    # 3. For non-wait, scan only a few commands ahead in the same block.
    # This handles harmless drift but prevents jumping straight to a later wait.
    # 3. For non-wait, scan ahead only until the next scheduled Wait.
    # Never jump over a Wait. If we jump over a Wait, long waits in the middle
    # of a block can disappear from the trace.
    if action_type != "Wait" and start_block < len(blocks):
        block = blocks[start_block]

        for skill_idx in range(start_skill, len(block.skills)):
            skill = block.skills[skill_idx]

            if skill_idx != start_skill and _skill_type_name(skill) == "Wait":
                break

            if _action_matches_skill_type_and_agent(skill, action, agent_uid):
                return start_block, skill_idx

    return None, None


def _set_active_scheduled_skill(block_idx: Optional[int], skill_idx: Optional[int]) -> None:
    if block_idx is None or skill_idx is None:
        _DUAL_CLOCK_STATE.update(
            {
                "active_block_idx": None,
                "active_skill_idx": None,
                "active_wall_start": None,
                "active_world_time_start": None,
                "active_last_elapsed_second": 0,
                "active_last_wall_clock": None,
                "active_block_expired": False,
            }
        )
        return

    blocks = _DUAL_CLOCK_STATE["blocks"]
    block = blocks[block_idx]

    # If we jumped to a new block, anchor the cursor to that block start.
    if block_idx != _DUAL_CLOCK_STATE["next_block_idx"]:
        _DUAL_CLOCK_STATE["wall_cursor"] = block.wall_start

    _DUAL_CLOCK_STATE.update(
        {
            "active_block_idx": block_idx,
            "active_skill_idx": skill_idx,
            "active_wall_start": _DUAL_CLOCK_STATE["wall_cursor"],
            "active_world_time_start": None,
            "active_last_elapsed_second": 0,
            "active_last_wall_clock": None,
            "active_block_expired": False,
        }
    )

def _write_active_long_wait_end_if_needed(trace_file: Path) -> None:
    """Force-write the final interpolated row for an active long wait.

    The simulator may return from execute_skill without calling the callback
    exactly at the clipped wait boundary. This helper writes that missing final
    row using the last known pose.

    Long waits in the middle of a block end at the wall time returned by
    _compute_active_long_wait_wall_end(), which reserves time for the remaining
    skills. Only a long wait that is the final skill should end at block.wall_end.
    """
    block_idx = _DUAL_CLOCK_STATE.get("active_block_idx")
    skill_idx = _DUAL_CLOCK_STATE.get("active_skill_idx")

    if block_idx is None or skill_idx is None:
        return

    blocks = _DUAL_CLOCK_STATE["blocks"]
    if block_idx >= len(blocks) or skill_idx >= len(blocks[block_idx].skills):
        return

    block = blocks[block_idx]
    skill = block.skills[skill_idx]

    if not getattr(skill, "is_long_wait", False):
        return

    active_wall_start = _DUAL_CLOCK_STATE.get("active_wall_start")
    if active_wall_start is None:
        return

    last_room_pose = _DUAL_CLOCK_STATE.get("last_room_pose")
    if last_room_pose is None:
        return

    room_name, x, y_world, z = last_room_pose

    wall_clock = _compute_active_long_wait_wall_end(
        block,
        skill_idx,
        active_wall_start,
    )
    wall_clock = _clamp_wall_clock_to_block(wall_clock, block)

    last_total_elapsed_seconds = float(
        _DUAL_CLOCK_STATE.get("last_total_elapsed_seconds") or 0.0
    )
    active_elapsed_so_far = float(
        _DUAL_CLOCK_STATE.get("active_last_elapsed_second") or 0.0
    )
    clipped_duration = float(getattr(skill, "clipped_duration", CLIP_THRESHOLD))

    # Make sim_time monotonic. If the callback did not reach the clipped
    # boundary, advance by only the missing active elapsed seconds. If it did,
    # do not move backward.
    missing_active_elapsed = max(0.0, clipped_duration - active_elapsed_so_far)
    forced_total_elapsed = last_total_elapsed_seconds + missing_active_elapsed

    sim_time_str = _format_hms(forced_total_elapsed * SIM_TIME_SCALE)
    wall_day_str = wall_clock.strftime("%A")
    wall_time_str = wall_clock.strftime("%H:%M:%S")
    activity = str(getattr(block, "activity", UNKNOWN_ACTIVITY))

    write_key = (
        wall_day_str,
        wall_time_str,
        sim_time_str,
        round(float(x), 3),
        round(float(z), 3),
        round(float(y_world), 3),
        room_name,
        activity,
    )

    if _TRACE_TIME_STATE.get("last_written_trace_key") == write_key:
        return

    _TRACE_TIME_STATE["last_written_trace_key"] = write_key
    _DUAL_CLOCK_STATE["active_last_wall_clock"] = wall_clock
    _DUAL_CLOCK_STATE["active_last_elapsed_second"] = max(
        active_elapsed_so_far,
        clipped_duration,
    )
    _DUAL_CLOCK_STATE["last_total_elapsed_seconds"] = forced_total_elapsed

    _write_trace_line(
        trace_file,
        wall_day_str=wall_day_str,
        wall_time_str=wall_time_str,
        sim_time_str=sim_time_str,
        room_name=room_name,
        xyz=(x, y_world, z),
        activity=activity,
    )

def _advance_after_active_skill() -> None:
    """Advance schedule and wall cursor after the active execute_skill call."""
    block_idx = _DUAL_CLOCK_STATE.get("active_block_idx")
    skill_idx = _DUAL_CLOCK_STATE.get("active_skill_idx")
    if block_idx is None or skill_idx is None:
        return

    blocks = _DUAL_CLOCK_STATE["blocks"]
    if block_idx >= len(blocks) or skill_idx >= len(blocks[block_idx].skills):
        return

    block = blocks[block_idx]
    skill = block.skills[skill_idx]

    # If this skill hit the block boundary while it was still running,
    # jump directly to the next block. Remaining commands from this block
    # should not continue to consume/log wall time.
    if _DUAL_CLOCK_STATE.get("active_block_expired"):
        _advance_cursor_to_next_block_after_expiry(block_idx)
        return

    if getattr(skill, "is_long_wait", False):
        active_wall_start = _DUAL_CLOCK_STATE.get("active_wall_start")
        last_wall_clock = _DUAL_CLOCK_STATE.get("active_last_wall_clock")

        if last_wall_clock is not None:
            # Use the actual callback/forced long-wait end as the next cursor.
            # This is critical for long waits in the middle of a block.
            _DUAL_CLOCK_STATE["wall_cursor"] = min(last_wall_clock, block.wall_end)
        elif active_wall_start is not None:
            computed_cursor = _compute_active_long_wait_wall_end(
                block,
                skill_idx,
                active_wall_start,
            )
            _DUAL_CLOCK_STATE["wall_cursor"] = min(computed_cursor, block.wall_end)
        else:
            _DUAL_CLOCK_STATE["wall_cursor"] = block.wall_end
    else:
        last_wall_clock = _DUAL_CLOCK_STATE.get("active_last_wall_clock")
        active_wall_start = _DUAL_CLOCK_STATE.get("active_wall_start")
        if last_wall_clock is not None:
            _DUAL_CLOCK_STATE["wall_cursor"] = min(last_wall_clock, block.wall_end)
        elif active_wall_start is not None:
            elapsed = float(_DUAL_CLOCK_STATE.get("active_last_elapsed_second") or 0)
            computed_cursor = active_wall_start + timedelta(
                seconds=elapsed * SIM_TIME_SCALE
            )
            _DUAL_CLOCK_STATE["wall_cursor"] = min(computed_cursor, block.wall_end)

    next_block_idx = block_idx
    next_skill_idx = skill_idx + 1

    if next_skill_idx >= len(block.skills):
        next_block_idx += 1
        next_skill_idx = 0
        if next_block_idx < len(blocks):
            _DUAL_CLOCK_STATE["wall_cursor"] = blocks[next_block_idx].wall_start

    _DUAL_CLOCK_STATE["next_block_idx"] = next_block_idx
    _DUAL_CLOCK_STATE["next_skill_idx"] = next_skill_idx
    _DUAL_CLOCK_STATE["active_block_idx"] = None
    _DUAL_CLOCK_STATE["active_skill_idx"] = None
    _DUAL_CLOCK_STATE["active_wall_start"] = None
    _DUAL_CLOCK_STATE["active_world_time_start"] = None
    _DUAL_CLOCK_STATE["active_block_expired"] = False


def _make_step_callback(trace_file: Path, commands_file: Optional[Path] = None):
    """Create a step callback that logs the active scheduled skill."""
    _ensure_blocks_loaded(commands_file)

    def _callback(
        env_interface: EnvironmentInterface,
        observations: Dict[str, Any],
        high_level_skill_actions: Dict[int, Any],
        skill_step_idx: int,
    ) -> None:
        blocks = _DUAL_CLOCK_STATE["blocks"]
        block_idx = _DUAL_CLOCK_STATE.get("active_block_idx")
        skill_idx = _DUAL_CLOCK_STATE.get("active_skill_idx")
        active_wall_start = _DUAL_CLOCK_STATE.get("active_wall_start")

        if not blocks or block_idx is None or skill_idx is None or active_wall_start is None:
            return
        if block_idx >= len(blocks) or skill_idx >= len(blocks[block_idx].skills):
            return

        try:
            world_time = float(env_interface.sim.get_world_time())
        except Exception as exc:
            print(f"DEBUG: world_time failed: {exc}")
            world_time = 0.0

        if _TRACE_TIME_STATE["sim_time_start"] is None:
            _TRACE_TIME_STATE["sim_time_start"] = world_time

        if _DUAL_CLOCK_STATE["active_world_time_start"] is None:
            _DUAL_CLOCK_STATE["active_world_time_start"] = world_time

        total_elapsed_seconds = max(0.0, world_time - float(_TRACE_TIME_STATE["sim_time_start"]))
        _DUAL_CLOCK_STATE["last_total_elapsed_seconds"] = total_elapsed_seconds

        active_elapsed_seconds = max(
            0.0,
            world_time - float(_DUAL_CLOCK_STATE["active_world_time_start"]),
        )
        active_elapsed_second = int(active_elapsed_seconds)

        block = blocks[block_idx]
        skill = block.skills[skill_idx]

        is_long_wait = getattr(skill, "is_long_wait", False)
        is_long_wait_end = False

        if is_long_wait:
            clip_end_sim = int(float(getattr(skill, "clipped_duration", CLIP_THRESHOLD)))
            is_long_wait_end = active_elapsed_second >= clip_end_sim

        log_key = (block_idx, skill_idx, active_elapsed_second, is_long_wait_end)

        if _TRACE_TIME_STATE["last_logged_key"] == log_key:
            return

        _TRACE_TIME_STATE["last_logged_key"] = log_key
        _DUAL_CLOCK_STATE["active_last_elapsed_second"] = active_elapsed_second

        activity = str(getattr(block, "activity", UNKNOWN_ACTIVITY))

        try:
            room_name, x, y_world, z = _resolve_room_and_position(env_interface)
        except Exception:
            room_name, x, y_world, z = UNKNOWN_ACTIVITY, 0.0, 0.0, 0.0

        _DUAL_CLOCK_STATE["last_room_pose"] = (room_name, x, y_world, z)

        if getattr(skill, "is_long_wait", False):
            clip_end_sim = int(float(getattr(skill, "clipped_duration", CLIP_THRESHOLD)))

            if clip_end_sim <= 0:
                progress = 1.0
            else:
                progress = min(
                    1.0,
                    max(0.0, active_elapsed_second / float(clip_end_sim)),
                )

            wait_wall_end = _compute_active_long_wait_wall_end(
                block,
                skill_idx,
                active_wall_start,
            )

            total_wall_seconds = max(
                0.0,
                (wait_wall_end - active_wall_start).total_seconds(),
            )

            wall_clock = active_wall_start + timedelta(
                seconds=progress * total_wall_seconds
            )

        else:
            wall_clock = active_wall_start + timedelta(
                seconds=active_elapsed_second * SIM_TIME_SCALE
            )

        wall_clock = _clamp_wall_clock_to_block(wall_clock, block)
        _DUAL_CLOCK_STATE["active_last_wall_clock"] = wall_clock
        # Existing traces used scaled simulator elapsed time. Keep that behavior
        # so that the time column aligns with wall-equivalent motion timing.
        # Hard schedule boundary:
        # If this skill's computed wall time reaches/passes the activity end,
        # write at most one final row at block.wall_end, mark the block expired,
        # and suppress any later callback rows from this still-running skill.
        if _DUAL_CLOCK_STATE.get("active_block_expired"):
            return

        if _current_block_has_expired(wall_clock, block):
            wall_clock = block.wall_end

            # Only expire the whole block if this is the final skill. A long
            # wait in the middle of a block must leave the cursor inside the
            # same activity so later skills can still be logged there.
            if skill_idx >= len(block.skills) - 1:
                _DUAL_CLOCK_STATE["active_block_expired"] = True

        sim_time_str = _format_hms(total_elapsed_seconds * SIM_TIME_SCALE)
        wall_day_str = wall_clock.strftime("%A")
        wall_time_str = wall_clock.strftime("%H:%M:%S")

        write_key = (
            wall_day_str,
            wall_time_str,
            sim_time_str,
            round(float(x), 3),
            round(float(z), 3),
            round(float(y_world), 3),
            room_name,
            activity,
        )
        if _TRACE_TIME_STATE.get("last_written_trace_key") == write_key:
            return
        _TRACE_TIME_STATE["last_written_trace_key"] = write_key

        _write_trace_line(
            trace_file,
            wall_day_str=wall_day_str,
            wall_time_str=wall_time_str,
            sim_time_str=sim_time_str,
            room_name=room_name,
            xyz=(x, y_world, z),
            activity=activity,
        )

    return _callback


def _find_wait_duration_index(action: Any) -> Optional[int]:
    """Find the tuple/list index containing the Wait duration.

    Supports both:
    - ("Wait", 150)
    - ("Wait", 1, 150)
    """
    if not action or len(action) < 2:
        return None

    for idx in range(len(action) - 1, 0, -1):
        try:
            float(action[idx])
            return idx
        except (TypeError, ValueError):
            continue

    return None


def _cap_wait_actions(high_level_skill_actions: Optional[Dict[int, Any]]) -> None:
    """Cap Wait durations in-place before simulator execution.

    Important:
    This must clip the duration argument, not the agent id.
    For commands like Wait 1 1500, the duration is the last numeric argument.
    """
    if high_level_skill_actions is None:
        return

    for agent_uid in list(high_level_skill_actions.keys()):
        action = high_level_skill_actions[agent_uid]
        if not action or len(action) < 2:
            continue

        skill_type = action[0]
        if skill_type != "Wait":
            continue

        duration_idx = _find_wait_duration_index(action)
        if duration_idx is None:
            continue

        try:
            original_duration = float(action[duration_idx])
            capped_duration = min(original_duration, CLIP_THRESHOLD)

            if capped_duration != original_duration:
                action_list = list(action)
                action_list[duration_idx] = capped_duration
                high_level_skill_actions[agent_uid] = tuple(action_list)
        except (ValueError, IndexError, TypeError):
            pass


def _initialize_trace_file(trace_file: Path) -> None:
    """Start each run with a clean trace file and a matching header."""
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with trace_file.open("w", encoding="utf-8") as f:
        f.write("day wall_time sim_time x z y_world room activity\n")




def _patched_execute_skill(*args, **kwargs):
    """Patched execute_skill that caps Wait durations and logs dual-clock traces."""
    trace_file = _resolve_trace_file_path()
    commands_file = _resolve_commands_file_from_argv()
    _ensure_blocks_loaded(commands_file)

    high_level_skill_actions = kwargs.get("high_level_skill_actions")
    if high_level_skill_actions is None and len(args) > 0:
        high_level_skill_actions = args[0]

    action = _extract_primary_action(high_level_skill_actions)
    agent_uid = _extract_primary_agent_uid(high_level_skill_actions)

    # Important: select scheduled skill before clipping Wait.
    block_idx, skill_idx = _select_next_scheduled_skill(action, agent_uid)
    _set_active_scheduled_skill(block_idx, skill_idx)

    # Now cap long waits for simulator execution.
    _cap_wait_actions(high_level_skill_actions)

    kwargs["step_callback"] = _make_step_callback(
        trace_file,
        commands_file=commands_file,
    )

    try:
        return base_execute_skill(*args, **kwargs)
    finally:
        _write_active_long_wait_end_if_needed(trace_file)
        _advance_after_active_skill()


def main() -> None:
    _reset_trace_state()
    _initialize_trace_file(_resolve_trace_file_path())
    base_skill_runner.execute_skill = _patched_execute_skill
    base_skill_runner.run_skills()


if __name__ == "__main__":
    main()