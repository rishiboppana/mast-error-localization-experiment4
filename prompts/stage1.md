You are checking ONE specific step in a multi-agent system's trajectory for
evidence of any of 14 known failure modes.

DEFAULT TO NONE. Most steps are normal. Only report a mode if the step
gives clear, direct evidence — not a plausible inference, not speculation.
You must be able to quote the exact text that shows it.

Judge only the CURRENT STEP. Use the prefix only as context for modes that
require history (repetition, lost context, ignored input, reset) — do not
flag a prior step's problem as if it belongs to the current step.

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

FAILURE MODES:
{modes}

OUTPUT FORMAT (strict, no other text):
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
