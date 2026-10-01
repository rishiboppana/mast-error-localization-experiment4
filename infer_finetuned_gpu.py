"""
Standalone GPU-only eval of the LoRA-fine-tuned Qwen3-8B checkpoint (Stage F of
qwen-lora-finetuning-plan.md) on the untouched 56-trajectory held-out set, for
window=0 and window=3 -- mirrors mast_window_ablation.ipynb's metrics/report
format exactly, but runs the merged HF checkpoint directly via `transformers`
on CUDA (bf16, no quantization) instead of MLX, and writes to separate
*_ft-suffixed files so it never touches your existing baseline results.

This tests the PRE-quantization merged model -- a real sanity check of the
fine-tune itself, but not identical to the 4-bit MLX model you'll eventually
deploy (mlx_lm.convert -q). Re-validate after conversion too; don't treat this
run alone as the final number.

Usage:
    python infer_finetuned_gpu.py --model-path runs/qwen3_8b_lora_v1/merged_fp16
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = PROJECT_DIR / "prompts"
OUT_DIR = PROJECT_DIR
SUFFIX = "_ft"   # keeps every output file separate from the existing baseline results/reports

MAX_STEP_CHARS = 8000
MAX_STAGE2_STEP_CHARS = 1500
MAX_NEW_TOKENS = {"summarizer": 120, "stage1": 450, "stage2": 200}

HELD_OUT_CANDIDATES = [PROJECT_DIR / "held_out_test_set.json",
                       PROJECT_DIR.parent / "Experiment3" / "whodunit_experiment3" / "held_out_test_set.json"]
HELD_OUT_PATH = next((p for p in HELD_OUT_CANDIDATES if p.exists()), None)
assert HELD_OUT_PATH is not None, f"held_out_test_set.json not found in {HELD_OUT_CANDIDATES}"

SUMMARY_CACHE_PATH = PROJECT_DIR / "summaries.jsonl"   # reused read/append, same cache the baseline runs used
WINDOW_TAGS = {0: "window0", 3: "window3"}
HISTORY_MODES = {"1.3", "1.4", "2.1", "2.5", "2.6"}


def results_path(w: int) -> Path:
    return OUT_DIR / f"results_{WINDOW_TAGS[w]}{SUFFIX}.jsonl"


def report_path(w: int) -> Path:
    return OUT_DIR / f"{WINDOW_TAGS[w]}_report{SUFFIX}.md"


# ---------------------------------------------------------------------------
# Trajectory loading -- identical to mast_window_ablation.ipynb
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
    return {"trajectory_id": raw["question_ID"], "task": task.strip(), "steps": steps,
            "gt_agent": base_role(raw.get("mistake_agent", "")), "gt_step": gt_step}


# ---------------------------------------------------------------------------
# Prompts -- loaded verbatim from prompts/*.md, unmodified (must match training exactly)
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> tuple[str, str]:
    system, user = (PROMPT_DIR / name).read_text().split("=====USER=====\n", 1)
    return system.rstrip("\n"), user.rstrip("\n")


def fill(template: str, **kw) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), template)


MODES_TEXT = (PROMPT_DIR / "modes.md").read_text().rstrip("\n")
VALID_MODES = set(re.findall(r"^(\d\.\d)\s", MODES_TEXT, flags=re.M))
assert len(VALID_MODES) == 14, VALID_MODES

SUMMARIZER_SYS, SUMMARIZER_USER = load_prompt("summarizer.md")
STAGE1_SYS, STAGE1_USER = load_prompt("stage1.md")
STAGE1_SYS = STAGE1_SYS.replace("{modes}", MODES_TEXT)
STAGE2_SYS, STAGE2_USER = load_prompt("stage2.md")


# ---------------------------------------------------------------------------
# Model call + parsers -- identical logic to mast_window_ablation.ipynb's CUDA path
# ---------------------------------------------------------------------------

def generate(model, tok, system: str, user: str, max_new_tokens: int, sample: bool = False) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t0 = time.time()
    inputs = tok(text, return_tensors="pt").to(model.device)
    gen_kw = dict(do_sample=True, temperature=0.3, top_p=0.9) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **gen_kw)
    new = out[0, inputs["input_ids"].shape[1]:]
    return {"text": tok.decode(new, skip_special_tokens=True).strip(),
            "in_tok": int(inputs["input_ids"].shape[1]), "out_tok": int(new.shape[0]),
            "latency": time.time() - t0}


def call_with_retry(model, tok, system, user, max_new_tokens, parser):
    usage = {"in_tok": 0, "out_tok": 0, "latency": 0.0, "attempts": 0}
    raw_text = ""
    for attempt in range(2):
        g = generate(model, tok, system, user, max_new_tokens, sample=(attempt == 1))
        for k in ("in_tok", "out_tok", "latency"):
            usage[k] += g[k]
        usage["attempts"] += 1
        raw_text = g["text"]
        parsed = parser(raw_text)
        if parsed is not None:
            return parsed, usage, raw_text
    return None, usage, raw_text


def parse_summary(t: str):
    m = re.search(r"^SUMMARY:\s*(.+)$", t, flags=re.M)
    return {"summary": " ".join(m.group(1).split())} if m and m.group(1).strip() else None


def parse_stage1(t: str):
    m = re.search(r"^MODE:\s*(.+)$", t, flags=re.M)
    if not m:
        return None
    val = m.group(1).strip().strip("*` ")
    if val.upper().startswith("NONE"):
        return {"modes": [], "evidence": ""}
    codes = [c.strip() for c in re.split(r"[,\s]+", val) if c.strip()]
    if not codes or any(c not in VALID_MODES for c in codes):
        return None
    ev = re.search(r"^EVIDENCE:\s*(.+)", t, flags=re.M | re.S)
    evidence = " ".join(ev.group(1).split()) if ev else ""
    if not evidence:
        return None
    return {"modes": sorted(set(codes)), "evidence": evidence}


def parse_stage2(t: str):
    s = re.search(r"^DECISIVE_STEP:\s*(.+)$", t, flags=re.M)
    a = re.search(r"^DECISIVE_AGENT:\s*(.+)$", t, flags=re.M)
    j = re.search(r"^JUSTIFICATION:\s*(.+)", t, flags=re.M | re.S)
    if not (s and a and j):
        return None
    sv = s.group(1).strip().strip("*` ")
    if sv.upper().startswith("UNSURE"):
        step = None
    else:
        n = re.match(r"\d+", sv)
        if not n:
            return None
        step = int(n.group(0))
    av = a.group(1).strip().strip("*` ")
    agent = None if av.upper().startswith("UNSURE") else base_role(av)
    return {"predicted_step": step, "predicted_agent": agent, "justification": " ".join(j.group(1).split())}


# ---------------------------------------------------------------------------
# Summaries -- reused from the existing cache (window-independent); only
# computes a step here if it's genuinely missing from that cache.
# ---------------------------------------------------------------------------

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def append_jsonl(p: Path, rec: dict) -> None:
    with open(p, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_summaries(model, tok, trajectories: list[dict]) -> dict:
    cache = {(r["trajectory_id"], r["step_id"]): r for r in read_jsonl(SUMMARY_CACHE_PATH)}
    todo = [(t, s) for t in trajectories for s in t["steps"] if (t["trajectory_id"], s["step_id"]) not in cache]
    if todo:
        print(f"{len(todo)} steps missing from summaries.jsonl cache -- computing with the fine-tuned model")
    for t, s in tqdm(todo, desc="summarizer (cache miss)"):
        parsed, usage, raw_text = call_with_retry(
            model, tok, SUMMARIZER_SYS, fill(SUMMARIZER_USER, agent_name=s["agent"], step_text=s["text"]),
            MAX_NEW_TOKENS["summarizer"], parse_summary)
        rec = {"trajectory_id": t["trajectory_id"], "step_id": s["step_id"], "agent": s["agent"],
               "summary": parsed["summary"] if parsed else None, "parse_error": parsed is None,
               "raw": raw_text if parsed is None else None, **usage}
        append_jsonl(SUMMARY_CACHE_PATH, rec)
        cache[(rec["trajectory_id"], rec["step_id"])] = rec
    return cache


def build_prefix(step_log: list[dict], window: int) -> str:
    entries = step_log[-window:] if window else []
    return "\n".join(f"step {e['step_id']} | {e['agent']}: {e['summary']}" for e in entries)


def format_candidates(cands: list[dict], steps_by_id: dict) -> str:
    blocks = []
    for c in cands:
        step = steps_by_id.get((c["trajectory_id"], c["step_id"]))
        text = clip(step["text"], MAX_STAGE2_STEP_CHARS) if step else "<step text unavailable>"
        blocks.append(
            f'- step {c["step_id"]} | agent: {c["agent"]} | stage1 mode(s): {", ".join(c["modes"])}\n'
            f'  FULL STEP TEXT:\n  """\n  {text}\n  """'
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Main eval loop -- same design as mast_window_ablation.ipynb's run_window()
# ---------------------------------------------------------------------------

def run_window(model, tok, window: int, trajectories: list[dict], summaries: dict, steps_by_id: dict) -> list[dict]:
    path = results_path(window)
    done = read_jsonl(path)
    s1_done = {(r["trajectory_id"], r["step_id"]) for r in done if r["kind"] == "stage1"}
    s2_done = {r["trajectory_id"] for r in done if r["kind"] == "stage2"}

    for t in tqdm(trajectories, desc=f"window={window}"):
        tid = t["trajectory_id"]
        step_log = []
        for s in t["steps"]:
            key = (tid, s["step_id"])
            if key not in s1_done:
                prefix = build_prefix(step_log, window)
                parsed, usage, raw_text = call_with_retry(
                    model, tok, STAGE1_SYS,
                    fill(STAGE1_USER, task_description=t["task"], windowed_prefix=prefix,
                         step_id=s["step_id"], agent_name=s["agent"], step_text=s["text"]),
                    MAX_NEW_TOKENS["stage1"], parse_stage1)
                rec = {"kind": "stage1", "window": window, "trajectory_id": tid, "step_id": s["step_id"],
                       "agent": s["agent"], "modes": parsed["modes"] if parsed else [],
                       "evidence": parsed["evidence"] if parsed else "", "parse_error": parsed is None,
                       "raw": raw_text if parsed is None else None, **usage}
                append_jsonl(path, rec)
                done.append(rec)
                s1_done.add(key)
            sm = summaries[key]
            step_log.append({"step_id": s["step_id"], "agent": s["agent"],
                             "summary": sm["summary"] or "(summary unavailable)"})

        if tid not in s2_done:
            cands = sorted([r for r in done if r["kind"] == "stage1" and r["trajectory_id"] == tid and r["modes"]],
                           key=lambda r: r["step_id"])
            if cands:
                parsed, usage, raw_text = call_with_retry(
                    model, tok, STAGE2_SYS,
                    fill(STAGE2_USER, task_description=t["task"], candidates=format_candidates(cands, steps_by_id)),
                    MAX_NEW_TOKENS["stage2"], parse_stage2)
                rec = {"kind": "stage2", "window": window, "trajectory_id": tid,
                       "predicted_step": parsed["predicted_step"] if parsed else None,
                       "predicted_agent": parsed["predicted_agent"] if parsed else None,
                       "justification": parsed["justification"] if parsed else "",
                       "num_candidates": len(cands), "parse_error": parsed is None,
                       "raw": raw_text if parsed is None else None, **usage}
            else:
                rec = {"kind": "stage2", "window": window, "trajectory_id": tid, "predicted_step": None,
                       "predicted_agent": None, "justification": "", "num_candidates": 0,
                       "parse_error": False, "raw": None, "in_tok": 0, "out_tok": 0, "latency": 0.0, "attempts": 0}
            append_jsonl(path, rec)
            done.append(rec)
            s2_done.add(tid)
    return done


# ---------------------------------------------------------------------------
# Metrics + report -- identical format to mast_window_ablation.ipynb, so the
# markdown output is directly comparable to window0_report.md / window3_report.md
# ---------------------------------------------------------------------------

def naive_baselines(gt: dict) -> dict:
    agents = Counter(a for a, _ in gt.values())
    steps = Counter(s for _, s in gt.values())
    joint = Counter(gt.values())
    n = len(gt)
    return {"agent": agents.most_common(1)[0][1] / n, "agent_label": agents.most_common(1)[0][0],
            "step": steps.most_common(1)[0][1] / n, "step_label": steps.most_common(1)[0][0],
            "joint": joint.most_common(1)[0][1] / n, "joint_label": joint.most_common(1)[0][0]}


def compute_metrics(window: int, trajectories: list[dict], summaries: dict, gt: dict, all_modes: list[str]) -> dict:
    recs = read_jsonl(results_path(window))
    s1 = [r for r in recs if r["kind"] == "stage1"]
    s2 = {r["trajectory_id"]: r for r in recs if r["kind"] == "stage2"}
    n_traj = len(trajectories)
    m = {"window": window, "n_traj": n_traj, "n_steps": len(s1)}

    m["parse_error_rate_stage1"] = sum(r["parse_error"] for r in s1) / max(1, len(s1))
    m["parse_error_rate_stage2"] = sum(r["parse_error"] for r in s2.values()) / max(1, sum(r["num_candidates"] > 0 for r in s2.values()))
    m["parse_error_rate_summarizer"] = sum(r["parse_error"] for r in summaries.values()) / max(1, len(summaries))
    m["step_flag_rate"] = sum(bool(r["modes"]) for r in s1) / max(1, len(s1))

    mode_counts = Counter(c for r in s1 for c in r["modes"])
    m["mode_flag_rate"] = {c: mode_counts.get(c, 0) / max(1, len(s1)) for c in all_modes}
    m["mode_counts"] = {c: mode_counts.get(c, 0) for c in all_modes}

    by_step = {(r["trajectory_id"], r["step_id"]): r for r in s1}
    gt_recs = [(tid, by_step.get((tid, st))) for tid, (_, st) in gt.items() if st is not None]
    gt_recs = [(tid, r) for tid, r in gt_recs if r is not None]
    m["n_gt_steps_evaluable"] = len(gt_recs)
    flagged = [r for _, r in gt_recs if r["modes"]]
    m["recall"] = len(flagged) / max(1, len(gt_recs))
    m["recall_history_modes"] = sum(any(c in HISTORY_MODES for c in r["modes"]) for _, r in gt_recs) / max(1, len(gt_recs))
    m["recall_context_indep_modes"] = sum(any(c not in HISTORY_MODES for c in r["modes"]) for _, r in gt_recs) / max(1, len(gt_recs))
    m["gt_step_parse_errors"] = sum(r["parse_error"] for _, r in gt_recs)

    ag_ok = st_ok = jt_ok = ag_ok_st_bad = 0
    for tid, (ga, gs) in gt.items():
        r = s2.get(tid)
        pa, ps = (r["predicted_agent"], r["predicted_step"]) if r else (None, None)
        a, s = (pa == ga), (ps is not None and ps == gs)
        ag_ok += a
        st_ok += s
        jt_ok += a and s
        ag_ok_st_bad += a and not s
    m.update({"agent_acc": ag_ok / n_traj, "step_acc": st_ok / n_traj, "joint_acc": jt_ok / n_traj,
              "agent_correct_step_wrong": ag_ok_st_bad,
              "agent_correct_step_wrong_share": ag_ok_st_bad / max(1, ag_ok)})
    m["baseline"] = naive_baselines(gt)

    m["pred_agent_dist"] = dict(Counter((r["predicted_agent"] or "None/UNSURE") for r in s2.values() if r["num_candidates"] > 0))
    m["gt_agent_dist"] = dict(Counter(a for a, _ in gt.values()))
    m["zero_candidate_rate"] = sum(r["num_candidates"] == 0 for r in s2.values()) / n_traj
    m["cands_per_traj"] = [r["num_candidates"] for r in s2.values()]
    m["unsure_rate"] = sum(r["num_candidates"] > 0 and r["predicted_step"] is None for r in s2.values()) / max(1, sum(r["num_candidates"] > 0 for r in s2.values()))

    tok_s1 = sum(r["in_tok"] + r["out_tok"] for r in s1)
    tok_s2 = sum(r["in_tok"] + r["out_tok"] for r in s2.values())
    tok_sm = sum(r["in_tok"] + r["out_tok"] for r in summaries.values())
    lat = sum(r["latency"] for r in s1) + sum(r["latency"] for r in s2.values()) + sum(r["latency"] for r in summaries.values())
    m.update({"tokens_summarizer": tok_sm, "tokens_stage1": tok_s1, "tokens_stage2": tok_s2,
              "tokens_total": tok_sm + tok_s1 + tok_s2, "tokens_per_traj": (tok_sm + tok_s1 + tok_s2) / n_traj,
              "stage1_in_tokens_per_step": sum(r["in_tok"] for r in s1) / max(1, len(s1)),
              "latency_total_s": lat, "latency_per_traj_s": lat / n_traj})
    return m


def headline_row(m: dict) -> dict:
    b = m["baseline"]
    return {"window": m["window"], "recall": m["recall"], "recall_hist_modes": m["recall_history_modes"],
            "recall_ctx_indep_modes": m["recall_context_indep_modes"], "step_flag_rate": m["step_flag_rate"],
            "agent_acc": m["agent_acc"], "step_acc": m["step_acc"], "joint_acc": m["joint_acc"],
            "naive_agent": b["agent"], "naive_step": b["step"], "naive_joint": b["joint"],
            "zero_cand_rate": m["zero_candidate_rate"], "unsure_rate": m["unsure_rate"],
            "parse_err_s1": m["parse_error_rate_stage1"], "tokens/traj": round(m["tokens_per_traj"]),
            "stage1_in_tok/step": round(m["stage1_in_tokens_per_step"]), "latency/traj_s": round(m["latency_per_traj_s"], 1)}


def write_report(m: dict, model_path: str, all_modes: list[str], prior: list[dict] | None = None) -> str:
    w = m["window"]
    b = m["baseline"]
    L = []
    L.append(f"# MAST pipeline (LoRA fine-tuned, GPU/bf16, pre-quantization) — window={w} report\n")
    L.append(f"{m['n_traj']} trajectories, {m['n_steps']} steps. Model: {model_path} (bf16, CUDA).\n")
    L.append("## Headline\n")
    L.append(f"- **Recall of true mistake steps (flagged by Stage 1):** {m['recall']:.1%} of {m['n_gt_steps_evaluable']} "
             f"(via history-dependent modes: {m['recall_history_modes']:.1%}; via context-independent modes: {m['recall_context_indep_modes']:.1%})")
    L.append(f"- Overall step flag rate: {m['step_flag_rate']:.1%}  |  zero-candidate trajectories: {m['zero_candidate_rate']:.1%}  |  Stage 2 UNSURE rate: {m['unsure_rate']:.1%}")
    L.append(f"- **Agent / step / joint accuracy:** {m['agent_acc']:.1%} / {m['step_acc']:.1%} / {m['joint_acc']:.1%}")
    L.append(f"- Naive majority baseline: agent {b['agent']:.1%} (always '{b['agent_label']}'), step {b['step']:.1%} (always {b['step_label']}), joint {b['joint']:.1%} (always {b['joint_label']})")
    L.append(f"- Agent-correct / step-wrong: {m['agent_correct_step_wrong']} trajectories ({m['agent_correct_step_wrong_share']:.1%} of agent-correct predictions)")
    L.append(f"- Predicted-agent distribution: {m['pred_agent_dist']}  (ground truth: {m['gt_agent_dist']})")
    L.append(f"- parse_error rates — Summarizer {m['parse_error_rate_summarizer']:.2%}, Stage 1 {m['parse_error_rate_stage1']:.2%}, Stage 2 {m['parse_error_rate_stage2']:.2%}; parse errors on ground-truth steps: {m['gt_step_parse_errors']}")
    L.append(f"- Cost: {m['tokens_per_traj']:.0f} tokens/trajectory ({m['tokens_total']} total; Stage 1 avg input {m['stage1_in_tokens_per_step']:.0f} tokens/step), {m['latency_per_traj_s']:.1f}s/trajectory\n")
    L.append("## Mode flag counts\n")
    L.append(pd.DataFrame({"count": m["mode_counts"], "rate": {c: f"{v:.2%}" for c, v in m["mode_flag_rate"].items()},
                           "history_dependent": {c: c in HISTORY_MODES for c in all_modes}}).to_markdown() + "\n")
    if prior:
        L.append("## Comparison against other windows (this fine-tuned run)\n")
        L.append(pd.DataFrame([headline_row(p) for p in prior] + [headline_row(m)]).set_index("window").T.to_markdown() + "\n")
    L.append("## Skepticism checklist\n")
    L.append("- Compare each accuracy to the naive baseline above (a number near the baseline is not a result).")
    L.append("- Agent-correct/step-wrong share: a high share means agent accuracy is partly hollow.")
    L.append("- Predicted-agent distribution vs ground truth: one agent dominating = prediction-space collapse.")
    L.append("- **This is the pre-quantization bf16 checkpoint** -- re-check after `mlx_lm.convert -q` before treating this as the final number.")
    report_path(w).write_text("\n".join(L))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=Path, required=True,
                     help="Path to the merged fp16 HF checkpoint from Stage E "
                          "(e.g. runs/qwen3_8b_lora_v1/merged_fp16).")
    ap.add_argument("--limit-trajectories", type=int, default=None, help="Debug: cap held-out set size.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA GPU visible -- this script is GPU-only"
    print(torch.cuda.get_device_name(0), "-", round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GB")

    print(f"loading {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    raw = json.loads(HELD_OUT_PATH.read_text())
    trajectories = [load_trajectory(r) for r in raw]
    if args.limit_trajectories:
        trajectories = trajectories[:args.limit_trajectories]
    gt = {t["trajectory_id"]: (t["gt_agent"], t["gt_step"]) for t in trajectories}
    steps_by_id = {(t["trajectory_id"], s["step_id"]): s for t in trajectories for s in t["steps"]}
    all_modes = sorted(VALID_MODES)
    print(f"{len(trajectories)} held-out trajectories, {sum(len(t['steps']) for t in trajectories)} steps "
          f"(from {HELD_OUT_PATH})")

    summaries = get_summaries(model, tok, trajectories)

    prior_metrics = []
    for window in (0, 3):
        run_window(model, tok, window, trajectories, summaries, steps_by_id)
        m = compute_metrics(window, trajectories, summaries, gt, all_modes)
        write_report(m, str(args.model_path), all_modes, prior=prior_metrics)
        prior_metrics.append(m)
        print(f"\n===== window={window} =====")
        print(f"recall={m['recall']:.1%}  step_flag_rate={m['step_flag_rate']:.1%}  "
              f"agent/step/joint acc={m['agent_acc']:.1%}/{m['step_acc']:.1%}/{m['joint_acc']:.1%}")
        print(f"saved: {results_path(window).name}, {report_path(window).name}")


if __name__ == "__main__":
    main()
