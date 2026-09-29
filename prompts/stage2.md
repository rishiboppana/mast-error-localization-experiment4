You are given a multi-agent trajectory that failed, along with steps
flagged as showing evidence of specific failure modes, in chronological
order. Identify the single decisive, root-cause step.

RULES:
1. Do not default to the earliest candidate automatically.
2. Do not default to blaming a delegating agent merely because it assigned
   work to the agent that erred — only if its OWN decision was itself
   wrong.
3. Use a counterfactual test: if this step had been done correctly, would
   the outcome plausibly have changed? Among candidates that pass, pick
   the earliest.
4. If genuinely unsure, say so.
A step that is purely a tool/terminal's mechanical execution output (a
stack trace, exit code, "no code to execute") cannot be the decisive step
if an earlier flagged candidate in the same trajectory represents an
actual authored decision or action — the terminal only reports what it
was given, it does not decide anything.
You are given each candidate's mode(s) and evidence quote only — form your
own judgment, not one inherited from how it was described.

OUTPUT FORMAT (strict, no other text):
DECISIVE_STEP: <step number, or UNSURE>
DECISIVE_AGENT: <agent name, or UNSURE>
JUSTIFICATION: <one sentence, using the counterfactual test>
=====USER=====
TASK: {task_description}

CANDIDATE STEPS (chronological):
{candidates}
