#!/usr/bin/env python3
"""stop hook: keep the agent iterating until the active /goal is met."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_lib import (  # noqa: E402
    append_log,
    eval_path,
    find_goal_dir,
    now_iso,
    read_eval,
    read_state,
    write_state,
)

VERIFY_TIMEOUT_SEC = 180
MAX_FOLLOWUP_CHARS = 3500


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def truncate(text: str, limit: int = MAX_FOLLOWUP_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[truncated]..."


def run_verify(cmd: str, cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, f"verify timed out after {VERIFY_TIMEOUT_SEC}s: {cmd}"
    except OSError as exc:
        return False, f"verify failed to start: {exc}"
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    summary = truncate(out) if out.strip() else f"(no output, exit={proc.returncode})"
    return ok, summary


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    status = payload.get("status") or "completed"
    loop_count = int(payload.get("loop_count") or 0)
    conversation_id = payload.get("conversation_id") or payload.get("session_id")
    generation_id = payload.get("generation_id") or f"loop-{loop_count}"

    goal_dir = find_goal_dir(payload if isinstance(payload, dict) else None)
    if goal_dir is None:
        emit({})
        return 0

    state = read_state(goal_dir)
    if not state or state.get("status") != "active":
        emit({})
        return 0

    # If both user + project hooks fire for the same turn, only the first continues.
    gate = goal_dir / f".handled-{generation_id}"
    if gate.exists():
        append_log(goal_dir, f"skip duplicate stop generation_id={generation_id}")
        emit({})
        return 0
    try:
        gate.write_text(now_iso(), encoding="utf-8")
    except OSError:
        pass

    # Do not fight an explicit abort
    if status in {"aborted", "error"}:
        append_log(goal_dir, f"stop status={status} — not continuing")
        emit({})
        return 0

    workspace = goal_dir.parent.parent  # <root>/.cursor/goal -> <root>
    state["hook_loop_count"] = loop_count
    if conversation_id and not state.get("conversation_id"):
        state["conversation_id"] = conversation_id

    max_iter = int(state.get("max_iterations") or 25)
    iterations = int(state.get("iterations") or 0) + 1
    state["iterations"] = iterations

    condition = state.get("condition") or "(missing condition)"
    verify_cmd = state.get("verify_command")

    append_log(
        goal_dir,
        f"stop loop_count={loop_count} iterations={iterations}/{max_iter} condition={condition!r}",
    )

    if iterations > max_iter:
        state["status"] = "exhausted"
        state["last_reason"] = f"hit max_iterations={max_iter}"
        write_state(goal_dir, state)
        emit(
            {
                "followup_message": (
                    f"/goal exhausted after {max_iter} iterations without meeting the condition.\n"
                    f"Condition: {condition}\n"
                    "Stop auto-continue. Summarize blockers and what remains. "
                    "User can `/goal` again or raise --max."
                )
            }
        )
        return 0

    # Prefer deterministic verify command when present
    if isinstance(verify_cmd, str) and verify_cmd.strip():
        ok, summary = run_verify(verify_cmd.strip(), workspace)
        if ok:
            state["status"] = "achieved"
            state["achieved_at"] = now_iso()
            state["last_reason"] = f"verify passed: {verify_cmd}"
            evidence = list(state.get("evidence") or [])
            evidence.append(f"verify ok: {verify_cmd}")
            state["evidence"] = evidence[-20:]
            write_state(goal_dir, state)
            # Clear eval so a later goal doesn't inherit
            ep = eval_path(goal_dir)
            if ep.exists():
                ep.unlink()
            append_log(goal_dir, "achieved via verify_command")
            emit({})  # success: do NOT emit a followup
            return 0
        state["last_reason"] = "verify failed"
        write_state(goal_dir, state)
        emit(
            {
                "followup_message": (
                    f"Active /goal not met (iteration {iterations}/{max_iter}).\n"
                    f"Condition: {condition}\n"
                    f"Verify command failed: `{verify_cmd}`\n"
                    f"Output:\n```\n{truncate(summary, 2500)}\n```\n"
                    "Fix the failure, re-run the verify path, then stop only when it passes. "
                    "Before ending the turn, record eval via:\n"
                    f"`python3 {Path(__file__).resolve().parent / 'goal_cli.py'} eval --not-met "
                    "--reason '...'`\n"
                    "If truly blocked on the user, run `goal_cli.py blocked --reason '...'`."
                )
            }
        )
        return 0

    evaluation = read_eval(goal_dir)
    if evaluation is None:
        state["last_reason"] = "missing last_eval.json"
        write_state(goal_dir, state)
        emit(
            {
                "followup_message": (
                    f"Active /goal — you stopped without writing an evaluation "
                    f"(iteration {iterations}/{max_iter}).\n"
                    f"Condition: {condition}\n"
                    "Continue working toward the condition. Before you stop again, run:\n"
                    f"`python3 {Path(__file__).resolve().parent / 'goal_cli.py'} eval --not-met "
                    "--reason '<why not met yet>'`\n"
                    "or `--met --reason '<evidence>'` when done, "
                    "or `blocked --reason '...'` if you need the user."
                )
            }
        )
        return 0

    if evaluation.get("blocked"):
        state["status"] = "blocked"
        state["blocked_reason"] = evaluation.get("reason") or "blocked"
        state["last_reason"] = state["blocked_reason"]
        write_state(goal_dir, state)
        append_log(goal_dir, "blocked — stopping")
        emit({})
        return 0

    if evaluation.get("met"):
        state["status"] = "achieved"
        state["achieved_at"] = now_iso()
        state["last_reason"] = evaluation.get("reason") or "agent reported met"
        write_state(goal_dir, state)
        append_log(goal_dir, "achieved via last_eval")
        emit({})
        return 0

    reason = evaluation.get("reason") or "condition not met yet"
    state["last_reason"] = reason
    write_state(goal_dir, state)
    # Consume eval so the next turn must write a fresh one
    ep = eval_path(goal_dir)
    if ep.exists():
        ep.unlink()

    emit(
        {
            "followup_message": (
                f"Active /goal continues (iteration {iterations}/{max_iter}).\n"
                f"Condition: {condition}\n"
                f"Evaluator reason: {reason}\n"
                "Keep going. Do not stop until the condition holds with evidence, "
                "or mark blocked. End every turn with goal_cli.py eval."
            )
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — hooks must fail open
        sys.stderr.write(f"[goal stop] {exc}\n")
        emit({})
        raise SystemExit(0)
