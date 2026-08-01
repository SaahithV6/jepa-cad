#!/usr/bin/env python3
"""sessionStart hook: inject active /goal into context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_lib import find_goal_dir, format_status, read_state  # noqa: E402


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    goal_dir = find_goal_dir(payload if isinstance(payload, dict) else None)
    if goal_dir is None:
        print("{}")
        return 0
    state = read_state(goal_dir)
    if not state or state.get("status") != "active":
        print("{}")
        return 0

    cli = Path(__file__).resolve().parent / "goal_cli.py"
    context = (
        "ACTIVE /goal harness is engaged for this workspace.\n"
        "You must keep working until the completion condition is met, "
        "or explicitly mark the goal blocked.\n\n"
        f"{format_status(state)}\n\n"
        "Protocol:\n"
        f"1. Before ending any turn, run: python3 {cli} eval --met|--not-met --reason '...'\n"
        f"2. If stuck needing the user: python3 {cli} blocked --reason '...'\n"
        "3. Prefer a deterministic verify command when setting goals "
        "(goal_cli.py set --verify '…').\n"
        "4. Do not claim done without evidence that satisfies the condition.\n"
    )
    out = {
        "additional_context": context,
        "env": {"CURSOR_GOAL_ACTIVE": "1", "CURSOR_GOAL_DIR": str(goal_dir)},
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[goal sessionStart] {exc}\n")
        print("{}")
        raise SystemExit(0)
