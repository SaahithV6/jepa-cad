# /goal

Act as Cursor's missing Goal Mode. Read and follow the skill at `~/.cursor/skills/goal/SKILL.md` (or `.cursor/skills/goal/SKILL.md` if present in this repo).

## Arguments

Treat everything the user typed after `/goal` as the argument:

- empty / `status` → show goal status, then stop
- `clear` / `stop` / `off` / `reset` / `cancel` / `none` → clear the active goal, then stop
- otherwise → set that text as the completion condition (honor `--verify '…'` and `--max N` if present) and **start working immediately**

## Required first tool action

Run the harness CLI under `~/.cursor/hooks/goal/goal_cli.py` (or the repo copy at `.cursor/hooks/goal/goal_cli.py`) for set/status/clear before doing other work.

## While the goal is active

Keep iterating until the condition is met with evidence. Before ending any turn, write an evaluation with `goal_cli.py eval`. If blocked on the user, run `goal_cli.py blocked`. Prefer a `--verify` shell check when the condition is test/build shaped.

Do not declare victory without proof.
