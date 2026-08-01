#!/usr/bin/env python3
"""CLI for the Cursor /goal harness.

Usage:
  goal_cli.py set "<condition>" [--verify CMD] [--max N] [--conversation ID]
  goal_cli.py status
  goal_cli.py clear
  goal_cli.py eval --met|--not-met --reason TEXT [--evidence ITEM ...]
  goal_cli.py blocked --reason TEXT
  goal_cli.py done --reason TEXT [--evidence ITEM ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from hooks/goal/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_lib import (  # noqa: E402
    DEFAULT_MAX_ITERATIONS,
    empty_state,
    eval_path,
    find_goal_dir,
    format_status,
    now_iso,
    read_state,
    write_json,
    write_state,
)


def resolve_dir() -> Path:
    goal_dir = find_goal_dir()
    if goal_dir is None:
        print("error: could not resolve workspace for .cursor/goal/", file=sys.stderr)
        sys.exit(2)
    return goal_dir


def cmd_set(args: argparse.Namespace) -> int:
    condition = " ".join(args.condition).strip()
    if not condition:
        print("error: condition required", file=sys.stderr)
        return 2
    if condition.lower() in {"clear", "stop", "off", "reset", "none", "cancel", "status"}:
        print(
            "error: that looks like a subcommand; use: goal_cli.py clear|status",
            file=sys.stderr,
        )
        return 2
    goal_dir = resolve_dir()
    state = empty_state(
        condition,
        verify_command=args.verify,
        max_iterations=args.max,
        conversation_id=args.conversation,
    )
    write_state(goal_dir, state)
    # Clear stale eval so stop hook doesn't inherit a previous met=true
    eval_file = eval_path(goal_dir)
    if eval_file.exists():
        eval_file.unlink()
    print(f"Goal set ({goal_dir})")
    print(format_status(state))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    goal_dir = resolve_dir()
    state = read_state(goal_dir)
    print(f"goal_dir: {goal_dir}")
    print(format_status(state))
    return 0 if state else 1


def cmd_clear(_: argparse.Namespace) -> int:
    goal_dir = resolve_dir()
    state = read_state(goal_dir)
    if not state or state.get("status") not in {"active", "blocked"}:
        print("No goal set")
        return 0
    condition = state.get("condition")
    state["status"] = "cleared"
    state["cleared_at"] = now_iso()
    state["last_reason"] = "cleared by user"
    write_state(goal_dir, state)
    print(f"Goal cleared: {condition}")
    return 0


def write_eval(goal_dir: Path, *, met: bool, reason: str, evidence: list[str], blocked: bool = False) -> None:
    payload = {
        "met": bool(met),
        "blocked": bool(blocked),
        "reason": reason,
        "evidence": evidence,
        "written_at": now_iso(),
    }
    write_json(eval_path(goal_dir), payload)
    state = read_state(goal_dir) or empty_state(reason)
    state["last_reason"] = reason
    if evidence:
        existing = list(state.get("evidence") or [])
        existing.extend(evidence)
        state["evidence"] = existing[-20:]
    if blocked:
        state["status"] = "blocked"
        state["blocked_reason"] = reason
    elif met:
        state["status"] = "achieved"
        state["achieved_at"] = now_iso()
    write_state(goal_dir, state)


def cmd_eval(args: argparse.Namespace) -> int:
    if args.met == args.not_met:
        print("error: pass exactly one of --met or --not-met", file=sys.stderr)
        return 2
    goal_dir = resolve_dir()
    state = read_state(goal_dir)
    if not state or state.get("status") != "active":
        print("error: no active goal", file=sys.stderr)
        return 1
    write_eval(
        goal_dir,
        met=bool(args.met),
        reason=args.reason,
        evidence=list(args.evidence or []),
        blocked=False,
    )
    print(format_status(read_state(goal_dir)))
    return 0


def cmd_blocked(args: argparse.Namespace) -> int:
    goal_dir = resolve_dir()
    state = read_state(goal_dir)
    if not state or state.get("status") != "active":
        print("error: no active goal", file=sys.stderr)
        return 1
    write_eval(goal_dir, met=False, reason=args.reason, evidence=[], blocked=True)
    print(format_status(read_state(goal_dir)))
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    goal_dir = resolve_dir()
    state = read_state(goal_dir)
    if not state or state.get("status") != "active":
        print("error: no active goal", file=sys.stderr)
        return 1
    write_eval(
        goal_dir,
        met=True,
        reason=args.reason,
        evidence=list(args.evidence or []),
        blocked=False,
    )
    print(format_status(read_state(goal_dir)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goal_cli.py", description="Cursor /goal harness CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    set_p = sub.add_parser("set", help="Set / replace active goal")
    set_p.add_argument("--verify", default=None, help="Optional shell check; exit 0 = done")
    set_p.add_argument("--max", type=int, default=DEFAULT_MAX_ITERATIONS)
    set_p.add_argument("--conversation", default=None)
    set_p.add_argument("condition", nargs="+", help="Completion condition")
    set_p.set_defaults(func=cmd_set)

    st = sub.add_parser("status", help="Show goal status")
    st.set_defaults(func=cmd_status)

    cl = sub.add_parser("clear", help="Clear active goal")
    cl.set_defaults(func=cmd_clear)

    ev = sub.add_parser("eval", help="Record end-of-turn evaluation")
    g = ev.add_mutually_exclusive_group(required=True)
    g.add_argument("--met", action="store_true")
    g.add_argument("--not-met", action="store_true")
    ev.add_argument("--reason", required=True)
    ev.add_argument("--evidence", action="append", default=[])
    ev.set_defaults(func=cmd_eval)

    bl = sub.add_parser("blocked", help="Mark goal blocked (needs user)")
    bl.add_argument("--reason", required=True)
    bl.set_defaults(func=cmd_blocked)

    dn = sub.add_parser("done", help="Mark goal achieved (prefer verify_command when possible)")
    dn.add_argument("--reason", required=True)
    dn.add_argument("--evidence", action="append", default=[])
    dn.set_defaults(func=cmd_done)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
