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
=====USER=====
STEP:
agent: {agent_name}
"""
{step_text}
"""
