"""Parser for activity blocks with wait-clip awareness and dual-clock support.

This module provides:
- Activity blocks with schedule times (wall_start, wall_end) for dual-clock simulation
- Skill details including Wait duration detection and clipping
- Timeline tracking for activity block sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any, Dict, Optional

UNKNOWN_ACTIVITY = "unknown"
CLIP_THRESHOLD = 120  # seconds — waits longer than this get clipped
DENSE_WINDOW = 10  # seconds — logging density at start/end of clipped waits


@dataclass(frozen=True)
class Skill:
    """Represents a single skill command parsed from the commands file."""
    source_line_no: int
    skill_type: str  # "Navigate", "Wait", "Pick", "Place", etc.
    agent_uid: int
    args: list[str]  # all arguments after skill name

    @property
    def is_wait(self) -> bool:
        return self.skill_type == "Wait"

    @property
    def is_long_wait(self) -> bool:
        """True if this is a Wait with duration > CLIP_THRESHOLD."""
        if not self.is_wait or len(self.args) < 1:
            return False
        try:
            duration = float(self.args[0])
            return duration > CLIP_THRESHOLD
        except (ValueError, IndexError, TypeError):
            return False

    @property
    def wait_duration(self) -> Optional[float]:
        """Return the Wait duration in seconds, or None if not a Wait."""
        if not self.is_wait or len(self.args) < 1:
            return None
        try:
            return float(self.args[0])
        except (ValueError, IndexError, TypeError):
            return None

    @property
    def clipped_duration(self) -> float:
        """Return the simulator execution duration in sim seconds.

        For Wait skills: capped at CLIP_THRESHOLD if the wait is long.
        For Navigate/Pick/Place: 0.0 — their duration is not declared,
        it is implicit from elapsed sim time between waits.
        """
        if not self.is_wait:
            return 0.0
        duration = self.wait_duration
        if duration is None:
            return 0.0
        return min(duration, CLIP_THRESHOLD)

    @property
    def real_duration(self) -> float:
        """Return the full unclipped wait duration in sim seconds.

        This is used to compute how far the wall clock should jump when a
        long wait finishes. For non-wait skills this is 0.0 — their wall
        time contribution is implicit from the block's wall_end schedule.

        NOTE: this is in sim seconds, not real-world seconds.
        Multiply by SIM_TIME_SCALE to get real-world seconds.
        """
        if not self.is_wait:
            return 0.0
        duration = self.wait_duration
        if duration is None:
            return 0.0
        return duration  # always the full original value, never clipped


@dataclass(frozen=True)
class ActivityBlock:
    """Represents a scheduled activity block with its skills."""
    activity: str       # e.g. "sleeping"
    room: str           # e.g. "bedroom_1"
    wall_start: datetime  # realistic schedule start
    wall_end: datetime    # realistic schedule end
    day_index: int        # 0=Monday, 6=Sunday
    skills: tuple[Skill, ...]  # ordered list of skills

    @property
    def wall_duration(self) -> timedelta:
        """Total wall-clock time for this block."""
        return self.wall_end - self.wall_start

    @property
    def total_wait_seconds(self) -> float:
        """Sum of all Wait durations in this block (unclipped, sim seconds)."""
        return sum(
            skill.wait_duration or 0.0
            for skill in self.skills
            if skill.is_wait
        )

    @property
    def total_clipped_wait_seconds(self) -> float:
        """Sum of all clipped Wait durations (actual sim execution seconds for waits)."""
        return sum(
            skill.clipped_duration
            for skill in self.skills
            if skill.is_wait
        )

    @property
    def has_long_wait(self) -> bool:
        """True if any skill in this block is a long wait."""
        return any(skill.is_long_wait for skill in self.skills)


# ---------------------------------------------------------------------------
# Legacy dataclasses — kept for backward compatibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillActivityEntry:
    source_line_no: int
    activity: str
    agent_uid: int
    skill_name: str
    target: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str) -> int:
    """Convert 'HH:MM' or 'HH:MM:SS' to seconds since midnight."""
    parts = time_str.strip().split(":")

    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time format: {time_str!r}")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0

    if not (0 <= hours <= 23):
        raise ValueError(f"Invalid hour in time: {time_str!r}")
    if not (0 <= minutes <= 59):
        raise ValueError(f"Invalid minute in time: {time_str!r}")
    if not (0 <= seconds <= 59):
        raise ValueError(f"Invalid second in time: {time_str!r}")

    return hours * 3600 + minutes * 60 + seconds


def _normalize_skill_line(raw_line: str) -> str:
    return re.sub(r"^([-*]\s+|\d+[.)]\s+)", "", raw_line).strip()


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


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_activity_blocks(
    commands_file: Optional[Path],
    week_start: Optional[datetime] = None,
) -> list[ActivityBlock]:
    """Parse commands file into structured activity blocks with schedule times.

    Args:
        commands_file: Path to the commands file
        week_start: Base datetime for computing absolute schedule times
                    (e.g. Monday 00:00:00). Defaults to 2026-01-05 00:00:00.

    Returns:
        List of ActivityBlock objects in file order.
    """
    if commands_file is None or not commands_file.exists():
        return []

    if week_start is None:
        week_start = datetime(2026, 1, 5, 0, 0, 0)  # Monday

    day_header = re.compile(
        r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*:\s*$",
        re.IGNORECASE,
    )
    schedule_line = re.compile(
    r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*,\s*([^:]+):\s*(.+?)\s*$"
)

    weekday_to_index = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }

    blocks: list[ActivityBlock] = []
    current_day_index: Optional[int] = None
    current_skills: list[Skill] = []
    # (room, activity, wall_start_str, wall_end_str)
    current_block_info: Optional[tuple[str, str, str, str]] = None

    def _flush_block(
        room: str,
        activity: str,
        wall_start_str: str,
        wall_end_str: str,
        day_index: int,
        skills: list[Skill],
    ) -> ActivityBlock:
        wall_start_seconds = _time_str_to_seconds(wall_start_str)
        wall_end_seconds = _time_str_to_seconds(wall_end_str)
        base_day = week_start + timedelta(days=day_index)
        base_day = base_day.replace(hour=0, minute=0, second=0, microsecond=0)

        wall_start_time = base_day + timedelta(seconds=wall_start_seconds)

        if wall_end_seconds < wall_start_seconds:
            # Activity crosses midnight — end is on the next calendar day
            next_day = base_day + timedelta(days=1)
            wall_end_time = next_day + timedelta(seconds=wall_end_seconds)
        else:
            wall_end_time = base_day + timedelta(seconds=wall_end_seconds)

        return ActivityBlock(
            activity=activity,
            room=room,
            wall_start=wall_start_time,
            wall_end=wall_end_time,
            day_index=day_index,
            skills=tuple(skills),
        )

    try:
        lines = commands_file.read_text(encoding="utf-8").splitlines()

        for source_line_no, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Day header
            day_match = day_header.match(stripped)
            if day_match:
                current_day_index = weekday_to_index[day_match.group(1).lower()]
                current_skills = []
                current_block_info = None
                continue

            # Schedule line — flush previous block and start a new one
            schedule_match = schedule_line.match(stripped)
            if schedule_match:
                if current_block_info is not None and current_day_index is not None:
                    room, activity, ws, we = current_block_info
                    blocks.append(_flush_block(room, activity, ws, we, current_day_index, current_skills))

                wall_start_str = schedule_match.group(1)
                wall_end_str = schedule_match.group(2)
                room = schedule_match.group(3).strip()
                activity = schedule_match.group(4).strip() or UNKNOWN_ACTIVITY
                current_block_info = (room, activity, wall_start_str, wall_end_str)
                current_skills = []
                continue

            # Skill line
            if current_day_index is None or current_block_info is None:
                continue

            normalized = re.sub(r"^([-*]\s+|\d+[.)]\s+)", "", stripped).strip()
            parts = normalized.split(None, 2)
            if len(parts) < 3:
                continue

            skill_type = parts[0]
            try:
                agent_uid = int(parts[1])
            except (ValueError, IndexError):
                continue

            args_str = parts[2] if len(parts) > 2 else ""
            args = args_str.split()

            current_skills.append(Skill(
                source_line_no=source_line_no,
                skill_type=skill_type,
                agent_uid=agent_uid,
                args=args,
            ))

        # Flush last block
        if current_block_info is not None and current_day_index is not None:
            room, activity, ws, we = current_block_info
            blocks.append(_flush_block(room, activity, ws, we, current_day_index, current_skills))

    except Exception:
        return []

    return blocks


# ---------------------------------------------------------------------------
# Legacy helpers
# ---------------------------------------------------------------------------

def load_skill_activity_entries(commands_file: Optional[Path]) -> list[SkillActivityEntry]:
    """Legacy function: load activity entries for backward compatibility."""
    if commands_file is None or not commands_file.exists():
        return []

    day_header = re.compile(
        r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*:\s*$",
        re.IGNORECASE,
    )
    schedule_line = re.compile(
    r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*,\s*[^:]+:\s*(.+?)\s*$"
)

    current_activity = UNKNOWN_ACTIVITY
    entries: list[SkillActivityEntry] = []

    try:
        for source_line_no, raw_line in enumerate(
            commands_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
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

            normalized_line = _normalize_skill_line(stripped_line)
            parts = normalized_line.split(" ", 2)
            if len(parts) < 3:
                continue

            skill_name = parts[0]
            try:
                agent_uid = int(parts[1])
            except ValueError:
                continue

            target = parts[2].strip()
            entries.append(
                SkillActivityEntry(
                    source_line_no=source_line_no,
                    activity=current_activity,
                    agent_uid=agent_uid,
                    skill_name=skill_name,
                    target=target,
                )
            )
    except Exception:
        return []

    return entries


class SkillActivityTimeline:
    """Legacy: timeline for backward compatibility."""

    def __init__(self, commands_file: Optional[Path]) -> None:
        self._commands_file = commands_file.resolve() if commands_file is not None else None
        self._entries = load_skill_activity_entries(commands_file)
        self._cursor = 0
        self._last_signature: Optional[tuple[int, str, str]] = None
        self._last_activity = UNKNOWN_ACTIVITY

    def resolve_activity(
        self,
        high_level_skill_actions: Dict[int, Any],
        blocking_agent_ids: Optional[list[int]],
    ) -> str:
        primary_action = _select_primary_action(high_level_skill_actions, blocking_agent_ids)
        if primary_action is None:
            return self._last_activity

        signature = primary_action
        if signature == self._last_signature:
            return self._last_activity

        matched_index = self._find_next_match(signature)
        if matched_index is None and self._last_signature is None:
            matched_index = self._find_any_match(signature)
        if matched_index is None:
            return self._last_activity

        entry = self._entries[matched_index]
        self._cursor = matched_index + 1
        self._last_signature = signature
        self._last_activity = entry.activity
        return entry.activity

    def _find_next_match(self, signature: tuple[int, str, str]) -> Optional[int]:
        for idx in range(self._cursor, len(self._entries)):
            entry = self._entries[idx]
            if (entry.agent_uid, entry.skill_name, entry.target) == signature:
                return idx
        return None

    def _find_any_match(self, signature: tuple[int, str, str]) -> Optional[int]:
        for idx, entry in enumerate(self._entries):
            if (entry.agent_uid, entry.skill_name, entry.target) == signature:
                return idx
        return None