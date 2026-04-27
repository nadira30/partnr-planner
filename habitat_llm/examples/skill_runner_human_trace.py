#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run skill_runner while logging human time/position/room to a text file.

Usage:
python -m habitat_llm.examples.skill_runner_human_trace --config-name examples/skill_runner_decentralized_config.yaml \
    +skill_runner_commands_file=skill_runner_commands_mapped.txt
"""

from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional, Union

import habitat_llm.examples.skill_runner as base_skill_runner
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.examples.example_utils import execute_skill as base_execute_skill

TRACE_FILE_ENV_VAR = "SKILL_RUNNER_HUMAN_TRACE_FILE"
DEFAULT_TRACE_FILE = "human_room_trace.txt"
TRACE_START_CLOCK = datetime(2026, 1, 5, 23, 0, 0)
UNKNOWN_ACTIVITY = "unknown"
WEEKDAY_TO_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_ACTIVITY_BLOCK_CACHE: dict[str, Any] = {
    "commands_file": None,
    "entries": [],
    "cursor": 0,
    "last_activity": UNKNOWN_ACTIVITY,
}
_TRACE_TIME_STATE: dict[str, Optional[Union[int, float]]] = {
    "sim_time_start": None,
    "last_logged_second": None,
}


def _resolve_commands_file_from_argv() -> Optional[Path]:
    for arg in sys.argv:
        if arg.startswith("+skill_runner_commands_file="):
            value = arg.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return Path(value).expanduser()
    return None


def _parse_schedule_time(time_str: str) -> int:
    hours_str, minutes_str = time_str.split(":", 1)
    return int(hours_str) * 60 + int(minutes_str)


def _load_activity_blocks(commands_file: Optional[Path]) -> list[tuple[int, str, str, str]]:
    if commands_file is None or not commands_file.exists():
        return []

    day_header = re.compile(
        r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*:\s*$",
        re.IGNORECASE,
    )
    schedule_line = re.compile(
        r"^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*,\s*[^:]+:\s*(.+?)\s*$"
    )

    current_activity = UNKNOWN_ACTIVITY
    entries: list[tuple[int, str, str, str]] = []

    try:
        for raw_line in commands_file.read_text(encoding="utf-8").splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            if day_header.match(stripped_line):
                continue

            schedule_match = schedule_line.match(stripped_line)
            if schedule_match:
                parsed_activity = schedule_match.group(3).strip()
                if parsed_activity:
                    current_activity = parsed_activity
                continue

            normalized_line = re.sub(
                r"^([-*]\s+|\d+[.)]\s+)", "", stripped_line
            ).strip()
            skill_components = normalized_line.split(" ", 2)
            if len(skill_components) < 3:
                continue

            skill_name = skill_components[0]
            try:
                agent_uid = int(skill_components[1])
            except ValueError:
                continue

            target = skill_components[2].strip()
            entries.append((agent_uid, skill_name, target, current_activity))
    except Exception:
        return []

    return entries


def _select_primary_action(
    high_level_skill_actions: Dict[int, Any], blocking_agent_ids: Optional[list[int]]
) -> Optional[tuple[int, str, str]]:
    if not high_level_skill_actions:
        return None

    selected_agent_uid: Optional[int] = None
    if blocking_agent_ids:
        for agent_uid in blocking_agent_ids:
            if agent_uid in high_level_skill_actions:
                selected_agent_uid = agent_uid
                break

    if selected_agent_uid is None:
        if 1 in high_level_skill_actions:
            selected_agent_uid = 1
        else:
            selected_agent_uid = sorted(high_level_skill_actions.keys())[0]

    action = high_level_skill_actions.get(selected_agent_uid)
    if action is None:
        return None

    skill_name = str(action[0]) if len(action) > 0 else ""
    target = str(action[1]) if len(action) > 1 else ""
    return selected_agent_uid, skill_name, target


def _resolve_activity_for_skill_block(
    commands_file: Optional[Path],
    high_level_skill_actions: Dict[int, Any],
    blocking_agent_ids: Optional[list[int]],
) -> str:
    if commands_file is None:
        return UNKNOWN_ACTIVITY

    resolved_file = commands_file.resolve()
    cached_file = _ACTIVITY_BLOCK_CACHE["commands_file"]
    if cached_file != resolved_file:
        _ACTIVITY_BLOCK_CACHE["commands_file"] = resolved_file
        _ACTIVITY_BLOCK_CACHE["entries"] = _load_activity_blocks(commands_file)
        _ACTIVITY_BLOCK_CACHE["cursor"] = 0
        _ACTIVITY_BLOCK_CACHE["last_activity"] = UNKNOWN_ACTIVITY

    primary_action = _select_primary_action(high_level_skill_actions, blocking_agent_ids)
    if primary_action is None:
        return str(_ACTIVITY_BLOCK_CACHE["last_activity"])

    action_agent_uid, action_skill, action_target = primary_action
    entries: list[tuple[int, str, str, str]] = _ACTIVITY_BLOCK_CACHE["entries"]
    cursor = int(_ACTIVITY_BLOCK_CACHE["cursor"])

    matched_index: Optional[int] = None
    for idx in range(cursor, len(entries)):
        entry_agent_uid, entry_skill, entry_target, _ = entries[idx]
        if (
            entry_agent_uid == action_agent_uid
            and entry_skill == action_skill
            and entry_target == action_target
        ):
            matched_index = idx
            break

    if matched_index is None:
        for idx, (entry_agent_uid, entry_skill, entry_target, _) in enumerate(entries):
            if (
                entry_agent_uid == action_agent_uid
                and entry_skill == action_skill
                and entry_target == action_target
            ):
                matched_index = idx
                break

    if matched_index is None:
        return str(_ACTIVITY_BLOCK_CACHE["last_activity"])

    activity = entries[matched_index][3]
    _ACTIVITY_BLOCK_CACHE["cursor"] = matched_index + 1
    _ACTIVITY_BLOCK_CACHE["last_activity"] = activity
    return activity


def _load_activity_schedule(commands_file: Optional[Path]) -> list[tuple[int, int, int, str]]:
    if commands_file is None or not commands_file.exists():
        return []

    day_header = re.compile(
        r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*:\s*$",
        re.IGNORECASE,
    )
    schedule_line = re.compile(
        r"^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*,\s*[^:]+:\s*(.+?)\s*$"
    )

    current_day_index: Optional[int] = None
    schedule: list[tuple[int, int, int, str]] = []

    try:
        for raw_line in commands_file.read_text(encoding="utf-8").splitlines():
            day_match = day_header.match(raw_line)
            if day_match:
                current_day_index = WEEKDAY_TO_INDEX[day_match.group(1).lower()]
                continue

            if current_day_index is None:
                continue

            match = schedule_line.match(raw_line)
            if not match:
                continue

            activity = match.group(3).strip()
            if not activity:
                continue

            schedule.append(
                (
                    current_day_index,
                    _parse_schedule_time(match.group(1)),
                    _parse_schedule_time(match.group(2)),
                    activity,
                )
            )
    except Exception:
        return []

    return schedule


def _resolve_activity_for_clock(
    schedule: list[tuple[int, int, int, str]], sim_clock: datetime
) -> str:
    if not schedule:
        return UNKNOWN_ACTIVITY

    week_start = sim_clock - timedelta(
        days=sim_clock.weekday(),
        hours=sim_clock.hour,
        minutes=sim_clock.minute,
        seconds=sim_clock.second,
        microseconds=sim_clock.microsecond,
    )

    for day_index, start_minutes, end_minutes, activity in schedule:
        start = week_start + timedelta(days=day_index, minutes=start_minutes)
        end_day_offset = 1 if end_minutes <= start_minutes else 0
        end = week_start + timedelta(
            days=day_index + end_day_offset,
            minutes=end_minutes,
        )
        if start <= sim_clock < end:
            return activity

    return UNKNOWN_ACTIVITY


def _resolve_room_from_position(env_interface: EnvironmentInterface) -> str:
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
                    return str(mapped_room_name)
                return room_name
    except Exception:
        pass

    try:
        human_wg = env_interface.world_graph[1]
        human_node = human_wg.get_human()
        room_node = human_wg.get_room_for_entity(human_node)
        return str(room_node.name)
    except Exception:
        return "unknown"


def _resolve_trace_file_path() -> Path:
    path_from_env = Path(Path.cwd(), DEFAULT_TRACE_FILE)
    env_override = None
    try:
        import os

        env_override = os.environ.get(TRACE_FILE_ENV_VAR)
    except Exception:
        env_override = None

    if env_override:
        path_from_env = Path(env_override).expanduser().resolve()
    return path_from_env


def _write_trace_line(
    trace_file: Path,
    env_interface: EnvironmentInterface,
    sim_day_str: str,
    sim_time_str: str,
    activity: str,
) -> None:
    x = float("nan")
    y = float("nan")
    z = float("nan")
    room_name = "unknown"

    try:
        human_agent = env_interface.sim.agents_mgr[1].articulated_agent
        pos = human_agent.base_pos
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])
    except Exception:
        pass

    room_name = _resolve_room_from_position(env_interface)

    # x and y here represent the house floor-plane coordinates (x, z in 3D world).
    line = (
        f"{sim_day_str} {sim_time_str} x={x:.3f} y={z:.3f}"
        f" y_world={y:.3f} room={room_name} activity={activity}\n"
    )
    with trace_file.open("a", encoding="utf-8") as f:
        f.write(line)


def _make_step_callback(trace_file: Path, activity: str):
    def _format_sim_time(env_interface: EnvironmentInterface) -> tuple[str, str, int]:
        try:
            world_time = float(env_interface.sim.get_world_time())
            if _TRACE_TIME_STATE["sim_time_start"] is None:
                _TRACE_TIME_STATE["sim_time_start"] = world_time
            elapsed_seconds = max(
                0.0, world_time - float(_TRACE_TIME_STATE["sim_time_start"])
            )
        except Exception:
            elapsed_seconds = 0.0

        sim_clock = TRACE_START_CLOCK + timedelta(seconds=elapsed_seconds)
        return sim_clock.strftime("%A"), sim_clock.strftime("%H:%M:%S"), int(elapsed_seconds)

    def _callback(
        env_interface: EnvironmentInterface,
        observations: Dict[str, Any],
        high_level_skill_actions: Dict[int, Any],
        skill_step_idx: int,
    ) -> None:
        sim_day_str, sim_time_str, elapsed_second = _format_sim_time(env_interface)
        if _TRACE_TIME_STATE["last_logged_second"] == elapsed_second:
            return
        _TRACE_TIME_STATE["last_logged_second"] = elapsed_second

        _write_trace_line(
            trace_file,
            env_interface,
            sim_day_str=sim_day_str,
            sim_time_str=sim_time_str,
            activity=activity,
        )

    return _callback


def _patched_execute_skill(*args, **kwargs):
    trace_file = _resolve_trace_file_path()
    commands_file = _resolve_commands_file_from_argv()
    high_level_skill_actions = kwargs.get("high_level_skill_actions")
    if high_level_skill_actions is None and len(args) > 0:
        high_level_skill_actions = args[0]
    if high_level_skill_actions is None:
        high_level_skill_actions = {}
    blocking_agent_ids = kwargs.get("blocking_agent_ids")
    activity = _resolve_activity_for_skill_block(
        commands_file,
        high_level_skill_actions,
        blocking_agent_ids,
    )

    if not trace_file.exists():
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        with trace_file.open("w", encoding="utf-8") as f:
            f.write("day time x y y_world room activity\n")

    kwargs["step_callback"] = _make_step_callback(trace_file, activity)
    return base_execute_skill(*args, **kwargs)


def main() -> None:
    base_skill_runner.execute_skill = _patched_execute_skill
    base_skill_runner.run_skills()


if __name__ == "__main__":
    main()
