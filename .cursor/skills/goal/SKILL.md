---
name: goal
description: >-
  Durable /goal harness that keeps the agent iterating until a completion
  condition is met. Use when the user runs /goal, asks to set a goal, clear a
  goal, check goal status, or wants Claude-Code-style autonomous iteration
  until tests/build/acceptance criteria pass.
---

# /goal

Cursor has no native Goal Mode. This skill + stop hook provide one:

- durable state in `<workspace>/.cursor/goal/state.json`
- auto-continue via the `stop` hook (`followup_message`) until done / blocked / exhausted
- optional deterministic `--verify` shell check (preferred over model self-report)

## Parse

Accept `/goal [args…]` (slash command or plain text):

| Input | Action |
|-------|--------|
| `/goal` or `/goal status` | Show status via CLI; stop |
| `/goal clear` (aliases: `stop`, `off`, `reset`, `cancel`, `none`) | Clear active goal; stop |
| `/goal <condition>` | Set/replace goal and **start working immediately** |
| `/goal <condition> --verify '<cmd>'` | Same, with exit-0 shell proof |
| `/goal <condition> --max N` | Cap auto-iterations (default 25) |

If the user pastes a long condition without flags, treat the whole remainder as the condition.

## CLI

Always use the harness CLI (do not hand-edit state unless recovering corruption):

```bash
python3 ~/.cursor/hooks/goal/goal_cli.py set "<condition>" [--verify 'cmd'] [--max 25]
python3 ~/.cursor/hooks/goal/goal_cli.py status
python3 ~/.cursor/hooks/goal/goal_cli.py clear
python3 ~/.cursor/hooks/goal/goal_cli.py eval --not-met --reason "..."
python3 ~/.cursor/hooks/goal/goal_cli.py eval --met --reason "..." --evidence "..."
python3 ~/.cursor/hooks/goal/goal_cli.py blocked --reason "..."
```

If this repo vendors the same scripts under `.cursor/hooks/goal/`, prefer the repo copy when present.

## Contract (while status=active)

1. **Work the condition** — run the checks the condition names; fix failures; repeat.
2. **End every turn with an eval** — before you would naturally stop:
   - not done → `eval --not-met --reason '…'`
   - done → `eval --met --reason '…'` (+ evidence), **or** rely on `--verify` exit 0
   - need the user → `blocked --reason 'exact blocker + smallest ask'`
3. **Never stop silent** — missing `last_eval.json` causes the stop hook to bounce you back.
4. **Evidence over vibes** — "probably done" is not done. Prefer `--verify` for tests/builds.
5. **One active goal** — a new `set` replaces the previous active goal.

## Writing good conditions

Good:

- `pytest tests/auth -q exits 0 and no files under generated/ are modified`
- `npm test && npm run lint both exit 0`
- `rocket CFD curated queue empty and overnight_status.json train_gate ready`

Bad:

- `make it better`
- `finish the refactor` (no check)
- conditions the transcript cannot demonstrate and no `--verify` can prove

Include turn/time bounds in the condition text when useful (`or stop after 20 turns`).

## Status replies

Keep them short:

```
◎ /goal active
condition: …
iterations: k/N
last_reason: …
```

On achieve/clear/block/exhaust, say why and do not continue the loop.

## Relationship to /loop

- `/goal` = continue **when the previous turn ends**, until a **condition** holds
- `/loop` = continue on a **timer**, regardless of condition

Do not arm `/loop` for the same objective while `/goal` is active unless the user asks.
