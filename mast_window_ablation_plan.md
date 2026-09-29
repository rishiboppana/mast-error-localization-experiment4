# Plan: MAST Direct-Classification Pipeline — Prefix Window Ablation (0 / 3 / All)

## What this replaces and why

Every prior experiment (v1, Qwen32B, from-scratch taxonomy, remediation) used
either a large per-agent-role component taxonomy or embedding-based
similarity search. Both kept hitting the same wall: register mismatch
between abstract descriptions and concrete action text, and (for the
component approach) taxonomy internal inconsistency.

This pipeline drops components and embeddings entirely. It classifies each
step directly against MAST's 14 fixed failure modes, in one LLM call, no
similarity search. The only open variable is how much trajectory history
("prefix") each Stage 1 call sees — that's what this ablation tests.

**Do not skip straight to comparing all three windows.** Run window=0 to
completion, inspect and report its results, THEN run window=3, THEN
window=all. Each window's results must be shown before the next window's
run starts. This is a hard requirement, not a suggestion — see Phase 5.

---

## Pipeline recap (3 call types)

1. **Summarizer** — one call per step, after that step is processed. Writes
   a neutral, factual one-line summary of the step, generated from the
   step's raw text only (never from a prior summary — append-only, no
   drift). Appended to a per-trajectory `step_log`.
2. **Stage 1** — one call per step. Input: task description + a window of
   prior steps' summaries (0, 3, or all — this ablation's variable) +
   current step's full raw text + the 14 MAST mode definitions. Output:
   which mode(s) apply (or NONE) + an evidence quote. No confidence, no
   rationale beyond the quote.
3. **Stage 2** — one call per trajectory, only if Stage 1 flagged at least
   one step. Input: task description + every flagged step in chronological
   order (step, agent, mode(s), evidence — no Stage 1 narrative). Output:
   the single decisive step + agent.

---

## Data structures

```python
# after Summarizer, per trajectory:
step_log = [
    {"step_id": 1, "agent": "Orchestrator", "summary": "..."},
    {"step_id": 2, "agent": "WebSurfer", "summary": "..."},
    ...
]

# after Stage 1, per step:
stage1_result = {
    "trajectory_id": "...",
    "step_id": 7,
    "agent": "WebSurfer",
    "modes": ["2.3", "3.2"],   # [] if NONE
    "evidence": "..."
}

# after Stage 2, per trajectory:
stage2_result = {
    "trajectory_id": "...",
    "predicted_step": 7,        # None if UNSURE or zero candidates
    "predicted_agent": "WebSurfer",
    "justification": "...",
    "num_candidates": 3
}
```

---

## The 14 MAST modes (reference — paste verbatim into both prompts)

```
SPECIFICATION
1.1 Disobey Task Specification — the action violates an explicit constraint
    or requirement stated in the task instructions.
1.2 Disobey Role Specification — the agent acts outside its assigned role
    or capabilities.
1.3 Step Repetition — the agent redundantly repeats an action already
    completed, with no new justification.
1.4 Loss of Conversation History — the agent acts as if information
    established earlier in the trajectory no longer exists or applies.
1.5 Unaware of Termination Conditions — the agent continues acting after
    the task's completion criteria have already been met.

INTER-AGENT MISALIGNMENT
2.1 Conversation Reset — the interaction restarts from scratch, discarding
    prior progress or context without cause.
2.2 Fail to Ask for Clarification — the agent proceeds under real ambiguity
    it should have raised, instead of guessing.
2.3 Task Derailment — the step drifts away from the task's actual goal.
2.4 Information Withholding — the agent has information relevant to
    another agent's task and fails to share it.
2.5 Ignored Other Agent's Input — the agent disregards or overrides a
    relevant instruction, correction, or finding another agent gave it.
2.6 Reasoning-Action Mismatch — the agent's own stated reasoning does not
    match the action it actually takes.

TASK VERIFICATION
3.1 Premature Termination — the task ends before the goal is achieved or
    fully verified.
3.2 No or Incomplete Verification — a result is not checked, or the check
    is superficial.
3.3 Incorrect Verification — a verification step reaches the wrong
    conclusion.
```

**Before running at scale**: check this wording against MAST's paper
appendix (arXiv:2503.13657) and correct any drift from the original
definitions — this is a working paraphrase, and "clear evidence" only means
what these definitions say it means.

---

## Prompts

### Summarizer

```
SYSTEM:
You are writing a one-line, purely factual summary of a single step in a
multi-agent trajectory, for use as context in a later review.

RULES:
- State only what the agent did — the concrete action, not its outcome,
  quality, or correctness.
- Do NOT evaluate, judge, or flag anything as wrong, suspicious, correct,
  or notable. Language like "failed to," "incorrectly," "without
  verifying" is evaluative — do not use it even if it seems accurate.
- Do not add information not in the step text.
- One line, one sentence, plain and specific (name the actual tool, value,
  or target mentioned).

OUTPUT FORMAT (strict, no other text):
SUMMARY: <one factual sentence>
```

```
USER:
STEP:
agent: {agent_name}
"""
{step_text}
"""
```

### Stage 1

```
SYSTEM:
You are checking ONE specific step in a multi-agent system's trajectory for
evidence of any of 14 known failure modes.

DEFAULT TO NONE. Most steps are normal. Only report a mode if the step
gives clear, direct evidence — not a plausible inference, not speculation.
You must be able to quote the exact text that shows it.

Judge only the CURRENT STEP. Use the prefix only as context for modes that
require history (repetition, lost context, ignored input, reset) — do not
flag a prior step's problem as if it belongs to the current step.

A step may show more than one mode. List every mode with clear evidence,
or NONE.

FAILURE MODES:
{14 mode definitions, pasted from above}

OUTPUT FORMAT (strict, no other text):
MODE: <comma-separated codes> or NONE
EVIDENCE: <exact quote supporting each mode listed, omit if NONE>
```

```
USER:
TASK: {task_description}

RECENT STEPS (context only — judge only the CURRENT STEP below):
{windowed_prefix}

CURRENT STEP TO JUDGE:
step {step_id} | agent: {agent_name}
"""
{step_text}
"""
```

`{windowed_prefix}` is empty string for window=0, the last 3 `step_log`
entries formatted as `step {id} | {agent}: {summary}` for window=3, and
every prior entry for window=all.

### Stage 2

```
SYSTEM:
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

You are given each candidate's mode(s) and evidence quote only — form your
own judgment, not one inherited from how it was described.

OUTPUT FORMAT (strict, no other text):
DECISIVE_STEP: <step number, or UNSURE>
DECISIVE_AGENT: <agent name, or UNSURE>
JUSTIFICATION: <one sentence, using the counterfactual test>
```

```
USER:
TASK: {task_description}

CANDIDATE STEPS (chronological):
- step {step_id} | agent: {agent_name} | mode(s): {modes} | evidence: "{evidence}"
...
```

---

## Worked example — one full trajectory slice, all three calls

Use this exact example when building/testing the code before running at
scale, so the input/output shape is unambiguous.

**Trajectory**: task = "Find Q3 revenue and email a summary to the team."

**Step 6** — `agent: WebSurfer`, raw text:
`"WebSurfer searched 'Q3 revenue figures' and opened the first result, a Q2 report cached from an earlier session."`

Summarizer call:
```
INPUT (user prompt):
STEP:
agent: WebSurfer
"""
WebSurfer searched 'Q3 revenue figures' and opened the first result, a Q2
report cached from an earlier session.
"""

OUTPUT:
SUMMARY: WebSurfer searched "Q3 revenue figures" and opened a cached Q2
report as the first result.
```

Stage 1 call (window=3, so prefix has 3 prior `step_log` entries):
```
INPUT (user prompt):
TASK: Find Q3 revenue and email a summary to the team.

RECENT STEPS (context only — judge only the CURRENT STEP below):
step 3 | Orchestrator: assigned the revenue lookup to WebSurfer
step 4 | WebSurfer: opened the company reporting dashboard
step 5 | WebSurfer: filtered the dashboard to "Q2" instead of "Q3"

CURRENT STEP TO JUDGE:
step 6 | agent: WebSurfer
"""
WebSurfer searched 'Q3 revenue figures' and opened the first result, a Q2
report cached from an earlier session.
"""

OUTPUT:
MODE: 3.2
EVIDENCE: "opened the first result, a Q2 report cached from an earlier
session"
```

(Note: with window=0, Stage 1 would not see step 5's "filtered to Q2
instead of Q3" — it might still catch this step alone, but a mode like 1.4
Loss of Conversation History would be unreachable without that context.
This is exactly the effect the ablation is designed to measure.)

Stage 2 call (assume steps 6 and 11 were both flagged in this trajectory):
```
INPUT (user prompt):
TASK: Find Q3 revenue and email a summary to the team.

CANDIDATE STEPS (chronological):
- step 6 | agent: WebSurfer | mode(s): 3.2 | evidence: "opened the first
  result, a Q2 report cached from an earlier session"
- step 11 | agent: Orchestrator | mode(s): 3.1 | evidence: "sent the email
  summary and marked the task complete"

OUTPUT:
DECISIVE_STEP: 6
DECISIVE_AGENT: WebSurfer
JUSTIFICATION: If WebSurfer had verified the report was for Q3 rather than
a cached Q2 result, the downstream email would have contained the correct
figures — step 11 only propagated an error that was already locked in.
```

---

## Phase 1 — Setup

- [ ] Load the 56 held-out trajectories (or the same 1,356-step held-out
  set used in the taxonomy experiments, for direct comparability).
- [ ] Confirm the model call wrapper enforces the strict output formats
  above (reject and retry once on malformed output; log any step that
  fails twice as `parse_error`, don't silently drop it).
- [ ] Save the 14 mode definitions and both stage prompts as versioned
  files (`prompts/summarizer.md`, `prompts/stage1.md`, `prompts/stage2.md`)
  so every window run uses byte-identical prompts — the only thing that
  changes between windows is `windowed_prefix`.

## Phase 2 — Window = 0 (run this alone first)

- [ ] For every trajectory: run Summarizer + Stage 1 per step with
  `windowed_prefix = ""`, then Stage 2 if any steps flagged.
- [ ] Save raw results to `results_window0.jsonl`.
- [ ] Compute metrics (list below) and generate plots (list below) for
  window=0 only.
- [ ] **Stop here. Report window=0's results before touching window=3.**

## Phase 3 — Window = 3 (only after Phase 2 is reported)

- [ ] Same as Phase 2, `windowed_prefix` = last 3 `step_log` entries.
- [ ] Save to `results_window3.jsonl`, compute the same metrics/plots.
- [ ] **Stop here. Report window=3's results, including a direct
  comparison table against window=0, before touching window=all.**

## Phase 4 — Window = All (only after Phase 3 is reported)

- [ ] Same as Phase 2, `windowed_prefix` = every prior `step_log` entry in
  the trajectory.
- [ ] Save to `results_windowAll.jsonl`, compute the same metrics/plots.
- [ ] Report window=all's results with the full three-way comparison.

## Phase 5 — Sequencing rule (explicit, do not skip)

Run and report Phase 2 completely before starting Phase 3. Run and report
Phase 3 completely before starting Phase 4. Do not batch all three windows
into one run-then-report step — the point is to see each window's result
before deciding whether the next window is even worth running (e.g. if
window=3 already saturates recall on the history-dependent modes, window=all
may not be needed at all, and that's a legitimate reason to stop early —
say so if the data shows it).

---

## Metrics — same per window, so they're directly comparable

- [ ] **Recall of true mistake steps** — of all ground-truth mistake steps
  in the held-out set, what fraction were flagged by Stage 1 at all
  (regardless of mode correctness)? This is the metric most likely to move
  across windows, since several modes are structurally unreachable at
  window=0.
- [ ] **Mode-level flag rate** — how often each of the 14 modes fires,
  split by window. Modes 1.3, 1.4, 2.1, 2.5, 2.6 (the history-dependent
  ones) should show the clearest window effect; the other ~9 should be
  roughly flat across windows if the pipeline is behaving as expected — a
  big unexpected swing in a context-independent mode is worth
  investigating, not just reporting.
- [ ] Agent accuracy, step accuracy, joint accuracy (Stage 2 predictions
  vs. ground truth `mistake_agent`/`mistake_step`).
- [ ] Fresh naive majority-class baseline, computed once (not per window,
  since ground truth doesn't change) and shown next to every window's
  accuracy.
- [ ] Agent-correct/step-wrong breakdown (the check that exposed the
  72.2%-was-hollow finding — apply it here too, every time).
- [ ] Prediction-space collapse check — distribution of predicted agents,
  per window.
- [ ] Zero-candidate rate (fraction of trajectories where Stage 1 flagged
  nothing, so Stage 2 was never called) — per window.
- [ ] Token/latency cost per window (Summarizer + Stage 1 + Stage 2 calls;
  window=all will cost more per Stage 1 call as trajectories get longer —
  quantify this, it's a real tradeoff against window=3 if the recall gain
  is small).

## Plots — per window, plus one combined comparison set

**Per-window (generate for each of the 3 windows, same layout so they're
visually comparable side by side):**
- [ ] Bar chart: flag rate per mode (14 bars), sorted by code.
- [ ] Bar chart: agent accuracy / step accuracy / joint accuracy vs. the
  fresh naive baseline.
- [ ] Histogram: number of candidates flagged per trajectory.
- [ ] Bar chart: predicted-agent distribution (the collapse check, visual
  form).

**Combined, generated only after all three windows have been run and
reported individually:**
- [ ] Grouped bar chart: recall of true mistake steps, window=0 vs. 3 vs.
  all, split into "history-dependent modes" vs. "context-independent
  modes" — this is the chart that actually answers the ablation's
  question.
- [ ] Line or bar chart: token cost per trajectory, window=0 vs. 3 vs. all
  — cost-vs-benefit against the recall chart above.
- [ ] Table: all headline metrics, one row per window, for the final
  writeup.

## Phase 6 — Output

- `results_window0.jsonl`, `results_window3.jsonl`, `results_windowAll.jsonl`
- `window0_report.md` (+ plots) — produced and shown before Phase 3 starts
- `window3_report.md` (+ plots, incl. vs. window0 comparison) — produced
  and shown before Phase 4 starts
- `windowAll_report.md` (+ plots, incl. full three-way comparison)
- `prompts/` — the three versioned prompt files actually used

## Notes

- Keep the Summarizer, Stage 1, and Stage 2 prompts byte-identical across
  all three windows — the only variable under test is `windowed_prefix`
  construction. Any other change between window runs invalidates the
  comparison.
- Log `parse_error` steps separately and report their rate per window —
  don't silently coerce a malformed output into NONE, that would quietly
  suppress recall and bias the comparison.
- Apply the same skepticism pass to every window's headline numbers before
  reporting them (naive baseline comparison, agent-correct/step-wrong,
  prediction collapse) — this has caught a false-positive result (Exp 2's
  72.2%) before and should be routine now, not optional.
