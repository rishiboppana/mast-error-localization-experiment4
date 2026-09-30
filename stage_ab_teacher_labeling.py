"""
Stage A + B for qwen-lora-finetuning-plan.md:
  A. Label the 128 non-held-out trajectories with an OpenAI teacher model,
     using the SAME windowed-prefix design as the prompted pipeline (window=3),
     but with a stricter/routing-metadata-aware teacher prompt.
  B. Convert those labels into train.jsonl / val.jsonl SFT examples using the
     UNMODIFIED, exact-inference-time STAGE1_SYS/STAGE1_USER template (from
     prompts/stage1.md) -- so the student never sees a prompt shape it won't
     see again at inference.

Run locally (uses your existing MLX Qwen3-8B for the cheap summarizer pass,
and the OpenAI API only for the actual Stage 1 teacher labels). Not for Colab.

    python stage_ab_teacher_labeling.py --help
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = PROJECT_DIR / "prompts"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency): sets os.environ for KEY=VALUE
    lines that aren't already set in the environment. Never logs the values."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv(PROJECT_DIR / ".env")

from openai import OpenAI  # noqa: E402 -- must come after load_dotenv() sets OPENAI_API_KEY
MAX_STEP_CHARS = 8000
WINDOW = 3

# ---------------------------------------------------------------------------
# 1. Exact same trajectory loading as mast_window_ablation.ipynb, so step_id /
#    agent naming is identical between teacher labeling and eval.
# ---------------------------------------------------------------------------

def base_role(raw: str) -> str:
    return re.split(r"\s*\(", (raw or "").strip(), maxsplit=1)[0].strip()


def clip(text: str, n: int = MAX_STEP_CHARS) -> str:
    if len(text) <= n:
        return text
    h = n // 2
    return text[:h] + f"\n[... {len(text) - n} chars truncated ...]\n" + text[-h:]


def load_trajectory(raw: dict) -> dict:
    steps, task = [], raw.get("question", "")
    for i, item in enumerate(raw.get("history", [])):
        agent = base_role(item.get("name") or item.get("role", ""))
        content = item.get("content", "")
        content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if agent.lower() == "human":
            if not task:
                task = content.strip()
            continue
        if not agent:
            continue
        steps.append({"step_id": i, "agent": agent, "text": clip(content)})
    try:
        gt_step = int(raw.get("mistake_step"))
    except (TypeError, ValueError):
        gt_step = None
    return {
        "trajectory_id": raw["question_ID"], "task": task.strip(), "steps": steps,
        "gt_agent": base_role(raw.get("mistake_agent", "")), "gt_step": gt_step,
    }


def load_pool(pool_paths: list[Path], held_out_ids: set[str], limit: int | None) -> list[dict]:
    """Loads every trajectory from pool_paths (JSON list files, same schema as
    held_out_test_set.json), de-duplicated by question_ID, EXCLUDING held_out_ids."""
    seen, out = set(), []
    for p in pool_paths:
        if not p.exists():
            print(f"  (skipping missing pool file: {p})")
            continue
        for raw in json.loads(p.read_text()):
            qid = raw["question_ID"]
            if qid in seen or qid in held_out_ids:
                continue
            seen.add(qid)
            out.append(raw)
    if limit:
        out = out[:limit]
    return [load_trajectory(r) for r in out]


# ---------------------------------------------------------------------------
# 2. Prompts. Two variants of Stage 1:
#      - TEACHER_*  : modified per plan A2, used ONLY for OpenAI labeling calls
#      - STUDENT_*  : the untouched prompts/stage1.md, used to build the SFT
#                      examples so training exactly matches inference.
#    Summaries use the existing summarizer.md unmodified (window-independent,
#    same as the prompted pipeline).
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> tuple[str, str]:
    system, user = (PROMPT_DIR / name).read_text().split("=====USER=====\n", 1)
    return system.rstrip("\n"), user.rstrip("\n")


def fill(template: str, **kw) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), template)


MODES_TEXT = (PROMPT_DIR / "modes.md").read_text().rstrip("\n")
VALID_MODES = sorted(re.findall(r"^(\d\.\d)\s", MODES_TEXT, flags=re.M))
assert len(VALID_MODES) == 14, VALID_MODES

SUMMARIZER_SYS, SUMMARIZER_USER = load_prompt("summarizer.md")
STUDENT_STAGE1_SYS, STUDENT_STAGE1_USER = load_prompt("stage1.md")
STUDENT_STAGE1_SYS = STUDENT_STAGE1_SYS.replace("{modes}", MODES_TEXT)

# Plan A2: same base prompt, plus (1) explicit routing-metadata warning and
# (2) a stronger DEFAULT-TO-NONE push, since training-data false positives are
# worse than eval-run false positives.
_TEACHER_ADDENDUM = """

ROUTING METADATA: Lines like "Next speaker <Agent>" are routing metadata
inserted by the multi-agent framework, not a statement by any agent. Never
cite them as evidence for any mode, especially 2.1 (Conversation Reset).

STRICT DEFAULT TO NONE: This label becomes permanent training data. A false
positive here is worse than a false positive in a single eval run, because a
model will be trained to reproduce it systematically. If you are not
confident enough to defend the flag to a skeptical reviewer quoting your own
evidence back at you, output NONE instead."""

TEACHER_STAGE1_SYS = STUDENT_STAGE1_SYS + _TEACHER_ADDENDUM
TEACHER_STAGE1_USER = STUDENT_STAGE1_USER  # user turn is unchanged


# ---------------------------------------------------------------------------
# 3. Cheap local summarizer pass (MLX Qwen3-8B) -- window=3 prefixes are built
#    from these, exactly like the prompted pipeline. No OpenAI cost here.
# ---------------------------------------------------------------------------

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def append_jsonl(p: Path, rec: dict) -> None:
    with open(p, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_summary(t: str) -> dict | None:
    m = re.search(r"^SUMMARY:\s*(.+)$", t, flags=re.M)
    return {"summary": " ".join(m.group(1).split())} if m and m.group(1).strip() else None


def build_summaries(trajectories: list[dict], cache_path: Path) -> dict:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    print("Loading mlx-community/Qwen3-8B-4bit for the summarizer pass...")
    model, tok = mlx_load("mlx-community/Qwen3-8B-4bit")

    def gen(system, user, max_tokens, sample=False):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text)
        sampler = make_sampler(temp=0.3, top_p=0.9) if sample else make_sampler(temp=0.0)
        return mlx_generate(model, tok, prompt=ids, max_tokens=max_tokens, sampler=sampler, verbose=False).strip()

    cache = {(r["trajectory_id"], r["step_id"]): r for r in read_jsonl(cache_path)}
    todo = [(t, s) for t in trajectories for s in t["steps"] if (t["trajectory_id"], s["step_id"]) not in cache]
    for t, s in tqdm(todo, desc="summarizer"):
        user = fill(SUMMARIZER_USER, agent_name=s["agent"], step_text=s["text"])
        parsed = None
        for attempt in range(2):
            raw_text = gen(SUMMARIZER_SYS, user, 120, sample=(attempt == 1))
            parsed = parse_summary(raw_text)
            if parsed:
                break
        rec = {"trajectory_id": t["trajectory_id"], "step_id": s["step_id"], "agent": s["agent"],
               "summary": parsed["summary"] if parsed else "(no summary available)"}
        append_jsonl(cache_path, rec)
        cache[(rec["trajectory_id"], rec["step_id"])] = rec
    return cache


def build_prefix(step_log: list[dict], window: int) -> str:
    entries = step_log[-window:] if window else []
    return "\n".join(f"step {e['step_id']} | {e['agent']}: {e['summary']}" for e in entries)


# ---------------------------------------------------------------------------
# 4. OpenAI teacher labeling with structured outputs (JSON schema, strict).
# ---------------------------------------------------------------------------

STAGE1_SCHEMA = {
    "name": "stage1_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "modes": {"type": "array", "items": {"type": "string", "enum": VALID_MODES}},
            "evidence": {"type": "string", "description": "Exact quote(s) supporting each listed mode. Empty string if modes is empty."},
        },
        "required": ["modes", "evidence"],
        "additionalProperties": False,
    },
}


def label_one(client: OpenAI, model: str, system: str, user: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": STAGE1_SCHEMA},
            )
            data = json.loads(resp.choices[0].message.content)
            modes = sorted(set(data["modes"]))
            evidence = data["evidence"].strip()
            if modes and not evidence:
                raise ValueError("modes present but evidence empty")
            return {"modes": modes, "evidence": evidence, "parse_error": False}
        except Exception as e:  # noqa: BLE001 -- log and retry/backoff, this is a labeling loop
            wait = min(2 ** attempt, 30)
            if attempt == retries - 1:
                return {"modes": [], "evidence": "", "parse_error": True, "error": str(e)}
            time.sleep(wait)


def format_assistant(modes: list[str], evidence: str) -> str:
    if not modes:
        return "MODE: NONE"
    return f"MODE: {', '.join(modes)}\nEVIDENCE: {evidence}"


def label_pool(trajectories: list[dict], summaries: dict, client: OpenAI, model: str,
                out_path: Path, max_workers: int) -> None:
    done = {(r["trajectory_id"], r["step_id"]) for r in read_jsonl(out_path)}

    jobs = []
    for t in trajectories:
        step_log = []
        for s in t["steps"]:
            key = (t["trajectory_id"], s["step_id"])
            prefix = build_prefix(step_log, WINDOW)
            step_log.append(summaries[key])
            if key in done:
                continue
            teacher_user = fill(TEACHER_STAGE1_USER, task_description=t["task"], windowed_prefix=prefix,
                                 step_id=s["step_id"], agent_name=s["agent"], step_text=s["text"])
            student_user = fill(STUDENT_STAGE1_USER, task_description=t["task"], windowed_prefix=prefix,
                                 step_id=s["step_id"], agent_name=s["agent"], step_text=s["text"])
            jobs.append({"trajectory_id": t["trajectory_id"], "step_id": s["step_id"], "agent": s["agent"],
                         "teacher_user": teacher_user, "student_user": student_user})

    print(f"{len(jobs)} steps to label ({len(done)} already cached in {out_path.name})")

    def work(job):
        result = label_one(client, model, TEACHER_STAGE1_SYS, job["teacher_user"])
        return {**{k: job[k] for k in ("trajectory_id", "step_id", "agent", "student_user")}, **result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rec in tqdm(ex.map(work, jobs), total=len(jobs), desc="teacher labeling"):
            append_jsonl(out_path, rec)


# ---------------------------------------------------------------------------
# 5. Label QA (plan A4) -- run before committing to the full formatted dataset.
# ---------------------------------------------------------------------------

def label_qa(labels: list[dict], trajectories: list[dict]) -> None:
    gt = {t["trajectory_id"]: (t["gt_agent"], t["gt_step"]) for t in trajectories if t["gt_step"] is not None}
    by_key = {(r["trajectory_id"], r["step_id"]): r for r in labels}

    n_parse_err = sum(r["parse_error"] for r in labels)
    print(f"\nparse errors: {n_parse_err}/{len(labels)} ({n_parse_err / max(1, len(labels)):.1%})")

    evaluable = [(tid, by_key.get((tid, gs))) for tid, (_, gs) in gt.items()]
    evaluable = [(tid, r) for tid, r in evaluable if r is not None]
    recall = sum(bool(r["modes"]) for _, r in evaluable) / max(1, len(evaluable))
    print(f"teacher recall against known mistake_step ({len(evaluable)} evaluable trajectories): {recall:.1%}")
    print("  -> if this is poor, FIX THE TEACHER PROMPT before spending the full budget.")

    flag_rate = sum(bool(r["modes"]) for r in labels) / max(1, len(labels))
    print(f"\noverall step flag rate: {flag_rate:.1%}")

    mode_counts = Counter(m for r in labels for m in r["modes"])
    print("label counts per mode:")
    for m in VALID_MODES:
        print(f"  {m}: {mode_counts.get(m, 0)}")

    next_speaker_re = re.compile(r'"?next speaker \w+"?\.?')
    mode21 = [r for r in labels if "2.1" in r["modes"]]
    boilerplate = sum(bool(next_speaker_re.search(r["evidence"].lower())) for r in mode21)
    print(f"\n2.1 flags: {len(mode21)}, of which {boilerplate} still cite 'Next speaker X' routing lines "
          f"(should be ~0 -- this is the exact bug the teacher prompt is meant to avoid)")

    print("\n--- manual spot check: 15 random labeled steps ---")
    random.seed(0)
    flagged = [r for r in labels if r["modes"]]
    sample = random.sample(flagged, min(15, len(flagged)))
    for r in sample:
        print(f"\ntraj={r['trajectory_id'][:12]} step={r['step_id']} agent={r['agent']} modes={r['modes']}")
        print(f"  evidence: {r['evidence'][:200]}")


# ---------------------------------------------------------------------------
# 6. Stage B: split by trajectory, write train.jsonl / val.jsonl
# ---------------------------------------------------------------------------

def write_sft_files(labels: list[dict], trajectories: list[dict], train_path: Path, val_path: Path,
                     val_frac: float, seed: int) -> None:
    clean = [r for r in labels if not r["parse_error"]]
    traj_ids = sorted({t["trajectory_id"] for t in trajectories})
    rng = random.Random(seed)
    rng.shuffle(traj_ids)
    n_val = max(1, round(len(traj_ids) * val_frac))
    val_ids = set(traj_ids[:n_val])

    train_path.write_text("")
    val_path.write_text("")
    n_train = n_val_written = 0
    for r in clean:
        example = {"messages": [
            {"role": "system", "content": STUDENT_STAGE1_SYS},
            {"role": "user", "content": r["student_user"]},
            {"role": "assistant", "content": format_assistant(r["modes"], r["evidence"])},
        ]}
        dest = val_path if r["trajectory_id"] in val_ids else train_path
        append_jsonl(dest, example)
        n_train += dest is train_path
        n_val_written += dest is val_path

    print(f"\nwrote {n_train} train examples ({len(traj_ids) - n_val} trajectories) -> {train_path}")
    print(f"wrote {n_val_written} val examples ({n_val} trajectories) -> {val_path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--held-out", type=Path,
                     default=PROJECT_DIR.parent / "Experiment3" / "whodunit_experiment3" / "held_out_test_set.json",
                     help="The 56-trajectory eval set. NEVER included in the training pool.")
    ap.add_argument("--pool", type=Path, nargs="+", required=True,
                     help="One or more JSON files (same schema as held_out_test_set.json) to draw the "
                          "128 training trajectories from. Trajectories whose question_ID is in --held-out "
                          "are always excluded regardless of what's in these files.")
    ap.add_argument("--limit-trajectories", type=int, default=None, help="Debug: cap pool size.")
    ap.add_argument("--openai-model", type=str, required=True,
                     help="Check platform.openai.com/docs/models for the current strongest reasoning model "
                          "in your account. Do not use a mini/nano tier (see plan Stage A1).")
    ap.add_argument("--max-workers", type=int, default=8, help="Concurrent OpenAI requests.")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction of trajectories held out for val.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-labeling", action="store_true",
                     help="Skip the OpenAI calls and go straight to QA + Stage B using an existing teacher_labels.jsonl.")
    ap.add_argument("--yes", action="store_true",
                     help="Non-interactive: write train.jsonl/val.jsonl automatically after QA, without prompting. "
                          "Use only after you've reviewed label_qa() output (e.g. from a prior --skip-labeling run) "
                          "and are satisfied -- QA is still printed either way.")
    args = ap.parse_args()

    held_out_ids = {r["question_ID"] for r in json.loads(args.held_out.read_text())}
    print(f"excluding {len(held_out_ids)} held-out trajectory IDs from {args.held_out}")

    trajectories = load_pool(args.pool, held_out_ids, args.limit_trajectories)
    n_steps = sum(len(t["steps"]) for t in trajectories)
    print(f"training pool: {len(trajectories)} trajectories, {n_steps} steps")
    assert not ({t["trajectory_id"] for t in trajectories} & held_out_ids), "held-out leak detected -- aborting"

    summary_cache = PROJECT_DIR / "summaries_trainpool.jsonl"
    labels_path = PROJECT_DIR / "teacher_labels.jsonl"

    if not args.skip_labeling:
        summaries = build_summaries(trajectories, summary_cache)
        client = OpenAI()
        label_pool(trajectories, summaries, client, args.openai_model, labels_path, args.max_workers)

    labels = read_jsonl(labels_path)
    label_qa(labels, trajectories)

    print("\n" + "=" * 70)
    if not args.yes:
        if sys.stdin.isatty():
            resp = input("QA looks acceptable -- proceed to write train.jsonl/val.jsonl? [y/N] ")
            proceed = resp.strip().lower() == "y"
        else:
            proceed = False
        if not proceed:
            print("Stopping before Stage B (non-interactive or declined). Review the QA output above, then "
                  "re-run with --skip-labeling --yes once satisfied.")
            return

    write_sft_files(labels, trajectories, PROJECT_DIR / "train.jsonl", PROJECT_DIR / "val.jsonl",
                     args.val_frac, args.seed)
    print("\nUpload train.jsonl and val.jsonl to Colab (or Drive) for Stage C.")


if __name__ == "__main__":
    main()
