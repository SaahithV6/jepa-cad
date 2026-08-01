#!/usr/bin/env python3
"""Shared /goal harness: durable state under <workspace>/.cursor/goal/."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

GOAL_DIRNAME = ".cursor/goal"
STATE_NAME = "state.json"
EVAL_NAME = "last_eval.json"
LOG_NAME = "hook.log"
SCHEMA_VERSION = 1
DEFAULT_MAX_ITERATIONS = 25


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_log(goal_dir: Path, line: str) -> None:
    try:
        goal_dir.mkdir(parents=True, exist_ok=True)
        with (goal_dir / LOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write(f"{now_iso()} {line}\n")
    except OSError:
        pass


def workspace_candidates(payload: dict[str, Any] | None = None) -> list[Path]:
    roots: list[Path] = []
    payload = payload or {}
    for key in ("workspace_roots",):
        val = payload.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    roots.append(Path(item).expanduser().resolve())
    for key in ("cwd", "workspace_root"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            roots.append(Path(val).expanduser().resolve())
    env_root = os.environ.get("CURSOR_GOAL_ROOT") or os.environ.get("CURSOR_PROJECT_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    # Walk up from process cwd looking for an existing goal or .git
    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        roots.append(parent)
        if len(roots) > 12:
            break
    # De-dupe preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def goal_dir_for(root: Path) -> Path:
    return root / ".cursor" / "goal"


def find_goal_dir(payload: dict[str, Any] | None = None) -> Path | None:
    """Prefer a workspace that already has goal state; else first writable workspace root."""
    candidates = workspace_candidates(payload)
    for root in candidates:
        state = goal_dir_for(root) / STATE_NAME
        if state.is_file():
            return goal_dir_for(root)
    for root in candidates:
        # Prefer real project roots (have .git or .cursor) over home
        if (root / ".git").exists() or (root / ".cursor").exists():
            return goal_dir_for(root)
    if candidates:
        return goal_dir_for(candidates[0])
    return None


def state_path(goal_dir: Path) -> Path:
    return goal_dir / STATE_NAME


def eval_path(goal_dir: Path) -> Path:
    return goal_dir / EVAL_NAME


def empty_state(condition: str, **extra: Any) -> dict[str, Any]:
    data = {
        "version": SCHEMA_VERSION,
        "status": "active",
        "condition": condition.strip(),
        "verify_command": None,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "conversation_id": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "iterations": 0,
        "hook_loop_count": 0,
        "last_reason": None,
        "evidence": [],
        "blocked_reason": None,
        "achieved_at": None,
        "cleared_at": None,
    }
    data.update(extra)
    return data


def read_state(goal_dir: Path) -> dict[str, Any] | None:
    raw = load_json(state_path(goal_dir))
    return raw if isinstance(raw, dict) else None


def write_state(goal_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(state_path(goal_dir), state)


def read_eval(goal_dir: Path) -> dict[str, Any] | None:
    raw = load_json(eval_path(goal_dir))
    return raw if isinstance(raw, dict) else None


def format_status(state: dict[str, Any] | None) -> str:
    if not state:
        return "No goal set."
    status = state.get("status", "unknown")
    condition = state.get("condition") or "(empty)"
    lines = [
        f"status: {status}",
        f"condition: {condition}",
        f"iterations: {state.get('iterations', 0)}/{state.get('max_iterations', DEFAULT_MAX_ITERATIONS)}",
    ]
    if state.get("verify_command"):
        lines.append(f"verify: {state['verify_command']}")
    if state.get("last_reason"):
        lines.append(f"last_reason: {state['last_reason']}")
    if state.get("blocked_reason"):
        lines.append(f"blocked_reason: {state['blocked_reason']}")
    if state.get("achieved_at"):
        lines.append(f"achieved_at: {state['achieved_at']}")
    if state.get("evidence"):
        lines.append("evidence:")
        for item in state["evidence"][-5:]:
            lines.append(f"  - {item}")
    return "\n".join(lines)
