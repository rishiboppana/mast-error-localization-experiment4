You are given a multi-agent trajectory that failed, and every step Stage 1
flagged as showing evidence of a failure mode, in chronological order, WITH THE FULL TEXT
OF EACH FLAGGED STEP. Your only job is to decide which one of these candidates is the
single decisive, root-cause step — read each candidate's actual text yourself and judge
it directly. Stage 1's mode label is given for reference only; it is not a verdict, and
its earlier evidence-quote pipeline is gone on purpose — you now have the whole step,
so form your own reading of what actually happened, not a reading inherited from Stage 1.

RULES:
1. Do not default to the earliest candidate automatically.
2. Do not default to blaming a delegating agent merely because it assigned
   work to the agent that erred — only if its OWN decision was itself
   wrong.
3. Use a counterfactual test: if this step had been done correctly, would
   the outcome plausibly have changed? Among candidates that pass, pick
   the earliest.
4. If genuinely unsure, say so.
5. A step that is purely a tool/terminal's mechanical execution output (a
   stack trace, exit code, "no code to execute") cannot be the decisive step
   if an earlier candidate in the same trajectory represents an actual
   authored decision or action — the terminal only reports what it was
   given, it does not decide anything. Check this against the candidate's
   own full text, not against how Stage 1 described it.
6. If a candidate's full text does not actually support the mode Stage 1
   assigned it, discount it accordingly — Stage 1 can be wrong.

OUTPUT FORMAT (strict, no other text):
DECISIVE_STEP: <step number, or UNSURE>
DECISIVE_AGENT: <agent name, or UNSURE>
JUSTIFICATION: <one sentence, using the counterfactual test>
=====USER=====
TASK: {task_description}

CANDIDATE STEPS (chronological, full text):
{candidates}
