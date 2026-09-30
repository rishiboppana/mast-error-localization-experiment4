You are checking ONE specific step in a multi-agent system's trajectory for
evidence of any of 14 known failure modes.

DEFAULT TO NONE. Most steps are normal. Only report a mode if the step
gives clear, direct evidence — not a plausible inference, not speculation.
You must be able to quote the exact text that shows it.

Judge only the CURRENT STEP. Use the prefix only as context for modes that
require history (repetition, lost context, ignored input, reset) — do not
flag a prior step's problem as if it belongs to the current step.

FRAMEWORK-NOISE CHECK: Some text in a step is scaffolding inserted by the
multi-agent framework itself, not a statement or decision made by the
agent. Do not treat it as evidence of anything:
- Routing lines like "Next speaker X" are the framework choosing who acts
  next. They are not a statement by any agent and never count as evidence
  of a conversation reset (2.1) or anything else — the agent didn't say
  this, the framework did.
- An orchestrating/delegating agent restating or re-issuing the CURRENT,
  still-unfinished instruction to a worker agent (even in near-identical
  wording across consecutive turns) is normal delegation scaffolding, not
  Step Repetition (1.3) — 1.3 requires an action that was already
  completed being redundantly done again, not an unfinished instruction
  being repeated. Before flagging 1.3, confirm the earlier occurrence
  describes a completed action, not an open request.

AGENCY CHECK: If the acting agent is a pure execution/tool component (e.g.
Computer_terminal) and its step is only reporting the mechanical result of
code it ran (an exit code, a traceback, "no code to execute"), it has no
independent judgment, planning, or communication role. Do not apply modes
that require agency or decision-making (1.2, 2.2, 2.4, 2.5, 2.6, 3.2, 3.3)
to such a step based on the execution result alone. At most, 1.1 or 3.1 may
apply if the step itself directly and independently caused the task to end
incorrectly — not merely because the code it ran was wrong.

A step may show more than one mode. List every mode with clear evidence,
or NONE.

Before answering, think step by step:
1. Strip out anything covered by the FRAMEWORK-NOISE CHECK — routing lines,
   and (for 1.3 candidates specifically) confirm the repeated thing is a
   completed action, not a restated open instruction.
2. What is this step's agent actually doing — acting, reporting, deciding,
   or communicating?
3. Does the AGENCY CHECK rule out any modes outright for this step?
4. For each remaining mode, is there a literal quote in THIS step (not the
   prefix, not framework scaffolding) that satisfies it? If you have to
   infer or assume, it doesn't count — move on.
5. Only after checking every mode this way, decide the final verdict.

Keep this reasoning brief (2-4 sentences). Do not use it to justify
borderline calls — if step 4 didn't turn up a literal quote from the
agent's own actual content, the answer is NONE regardless of what the
reasoning suggests.

FAILURE MODES:
{modes}

OUTPUT FORMAT (strict — REASONING first, then the final block with no
other text after it):
REASONING: <brief step-by-step check above>
MODE: <comma-separated codes> or NONE
EVIDENCE: <exact quote supporting each mode listed, omit if NONE>
=====USER=====
TASK: {task_description}

RECENT STEPS (context only — judge only the CURRENT STEP below):
{windowed_prefix}

CURRENT STEP TO JUDGE:
step {step_id} | agent: {agent_name}
"""
{step_text}
"""
