You are given a multi-agent trajectory that failed, and every step Stage 1
flagged as showing evidence of a failure mode, WITH THE FULL TEXT OF EACH FLAGGED STEP.
Candidates are grouped by agent (in order of first appearance), and numbered within
their group — e.g. [W1], [W2] are WebSurfer's 1st and 2nd flagged steps, in chronological
order within that group. Your job is to decide which ONE candidate, across all agents, is
the single decisive, root-cause step. Read each candidate's actual text yourself and judge
it directly — Stage 1's mode label is given for reference only, not a verdict.

WORK IN TWO PASSES.

PASS 1 — within each agent's own group, if it has more than one candidate, decide: are
these independent failures, or is this one agent repeating/continuing the same underlying
problem across several steps (a loop, a repeated bad query, the same unverified claim
carried forward)? If it's the latter, the decisive one is normally the EARLIEST step in
that group where the problem was actually introduced — not necessarily the first flagged
step overall, because early candidates in a group are often legitimate exploration that
Stage 1 over-flagged, and the real problem starts partway through. Read the full text: does
the earliest candidate actually contain a mistake, or is it a reasonable action that later
candidates in the same group repeat without correcting? Pick the true within-group origin
point, not just [X1].

PASS 2 — you now have at most one candidate per agent. Compare across agents using the
rules below.

EXAMPLE (illustrative, not from a real graded case):
WebSurfer group: [W1] step 4 — searches "Ted Danson TV series list", gets a results page.
[W2] step 11 — clicks into a "Credits" page, reasonable navigation. [W3] step 15 — scrolls
and OCRs a results page but does not record the season counts the task asked for. [W4]
step 19 — scrolls again, still not recording season counts, now also mis-states one
series' rating. [W5] step 23 — scrolls a third time, same gap.
Correct within-group reasoning: [W1] and [W2] are normal search/navigation, not failures —
searching and clicking into a page are not mistakes on their own. The actual problem
starts at [W3] (step 15): that's where the agent had the page in front of it and failed to
extract the information the task required, and [W4]/[W5] are the SAME unfixed gap
repeating, not new failures. So this group's decisive candidate is step 15, not step 4 —
even though step 4 is both the earliest flagged step overall and the earliest in this
group. Do not pick step 4 just because it came first; pick step 15 because that's where
the actual mistake was introduced.

RULES:
1. Do not default to the earliest candidate automatically — in either pass. Read what the
   candidate actually says before crediting or discounting it.
2. Do not default to blaming a delegating agent merely because it assigned
   work to the agent that erred — only if its OWN decision was itself
   wrong.
3. Use a counterfactual test: if this step had been done correctly, would
   the outcome plausibly have changed? Among candidates that pass, pick
   the earliest TRUE origin point (per PASS 1), not the earliest flagged step.
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
JUSTIFICATION: <one sentence, naming which pass (within-group or cross-agent) decided it>
=====USER=====
TASK: {task_description}

CANDIDATE STEPS (grouped by agent, full text):
{candidates}
