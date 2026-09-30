# Fine-tuning Plan: Qwen3-8B-4bit + LoRA for MAST Stage 1

**Goal:** replace the prompted Stage 1 classifier with a LoRA-adapted version of the same base model (`mlx-community/Qwen3-8B-4bit`), trained on GPT-labeled step-level MAST annotations from the 128 non-held-out trajectories, to fix the precision problem (69.9% flag rate against a ~4% true mistake rate) without touching the 56-trajectory eval set.

**Hard constraint, repeated because it's the one mistake that invalidates everything:** the 56-trajectory set (window=0/window=3 reports) is the eval set. It must never appear in teacher labeling, training, or validation. Only the 128 remaining trajectories are fair game for this pipeline.

---

## Stage A — Teacher labeling (OpenAI API, on the 128 trajectories)

### A1. Model choice
Pick the strongest general-purpose reasoning-capable model available in your OpenAI account at run time (check `platform.openai.com/docs/models` — I could not get a reliable current model list/pricing via search just now, so don't trust a name I'd give you here). Avoid the cheapest "mini/nano" tier for this — MAST classification requires reading nuanced multi-agent context and quoting exact evidence, which is exactly where cheap models get sloppy. This is a one-time labeling cost, not a recurring inference cost, so it's worth paying for quality.

### A2. Teacher prompt — reuse Stage 1's prompt, with two changes
Don't write a new prompt from scratch — reuse `STAGE1_MD` almost verbatim so the label distribution the student learns matches what it'll be asked to reproduce at inference. Two required changes:

1. **Explicitly instruct the teacher to ignore multi-agent framework routing metadata** (e.g. `"Next speaker <Agent>"` lines) as evidence for any mode, especially 2.1 (Conversation Reset). This is the single most important addition — if the teacher makes the same mistake the current small model makes, you've spent API budget teaching the student the same bug more durably. Add a line like:
   > "Lines like 'Next speaker X' are routing metadata inserted by the agent framework, not a statement by any agent. Never cite them as evidence for any mode."
2. **Ask for higher-confidence, stricter evidence.** Since this is training data (errors get baked into weights, not just one inference run), bias the teacher prompt toward the DEFAULT TO NONE instruction even more strongly than the current version — false positives in training data are worse than false positives in one eval run, because the student will learn to reproduce them systematically.

Use **structured output / JSON schema / function-calling mode** (whatever your chosen model's API offers) rather than free-text parsing — you want zero parse errors in training data, and OpenAI's structured output guarantees schema-valid responses.

### A3. Labeling loop
- Same per-step, windowed-prefix design as the current pipeline (pick one window size — window=3, since that's what you've been evaluating against; keep it consistent so the student learns the same task shape it'll run at inference).
- Output schema per step, matching your current Stage 1 JSONL format so you can reuse existing tooling:
  ```json
  {"trajectory_id": "...", "step_id": N, "agent": "...", "modes": ["2.3"], "evidence": "..."}
  ```
- 128 trajectories at your current ~24 steps/trajectory average ≈ **~3,000 step-level API calls**. Budget accordingly once you know current pricing; batch API mode (if your model offers it) is worth using here since this isn't latency-sensitive.
- Add basic retry/backoff and log parse failures — with structured outputs these should be near zero, but don't assume.

### A4. Label QA before training — do not skip this
- **Automatic sanity check:** if you have `mistake_step`/`mistake_agent` ground truth for any of the 128 (even without full per-step mode labels), check that the teacher's flagged steps include the true mistake step at a reasonable rate. If the teacher's own recall against known mistake steps is poor, fix the teacher prompt before generating the full 3,000-call dataset — don't discover this after paying for all of it.
- **Manual spot check:** pull ~30-50 random labeled steps across different trajectories and read them yourself. Look specifically for: is it still citing routing lines anywhere (the exact bug you're trying to distill away)? Is NONE actually being used, or is the teacher also over-flagging?
- **Class balance check:** count labels per mode. Some of the 14 modes (1.2, 2.6 in particular, based on your own reports) will have very few positive examples even across 128 trajectories. Don't try to fix this with synthetic oversampling — just know going in that the fine-tuned model likely won't learn much about the rarest modes, and say so honestly in your results.

---

## Stage B — Data formatting for fine-tuning

Convert teacher labels into chat-format SFT examples using the **exact same system/user template as `STAGE1_MD`** (fill `{modes}`, `{task_description}`, `{windowed_prefix}`, `{step_id}`, `{agent_name}`, `{step_text}` exactly as your inference-time code does), with the assistant turn being the teacher's `MODE:`/`EVIDENCE:` output:

```json
{"messages": [
  {"role": "system", "content": "<STAGE1_SYS with {modes} filled>"},
  {"role": "user", "content": "<STAGE1_USER filled for this step>"},
  {"role": "assistant", "content": "MODE: 2.3\nEVIDENCE: \"...\""}
]}
```

This matters more than it sounds like: if the training format diverges at all from the inference-time prompt (different field order, different whitespace, different instructions), the LoRA adapter partially overfits to a prompt shape it'll never actually see at inference, and you lose some of the gain.

**Split by trajectory, not by step**, into train/val (e.g. ~110/~18 trajectories) — splitting by step would leak steps from the same trajectory across train and val, inflating your validation numbers without telling you anything about generalization.

---

## Stage C — Colab environment

- **Recommended stack: [Unsloth](https://github.com/unslothai/unsloth)** — purpose-built for fast, memory-efficient QLoRA fine-tuning and has explicit Qwen3 support; typically 2x+ faster and much lower VRAM than plain HF `transformers`+`peft`+`bitsandbytes` on the same Colab GPU. Verify Qwen3-8B is in its supported model list at the time you run this (support tables change).
- Fallback if Unsloth doesn't fit: plain `transformers` + `peft` (LoRA) + `bitsandbytes` (4-bit quant) + `trl`'s `SFTTrainer` — slower and more VRAM-hungry, but more standard debugging surface if something goes wrong.
- **GPU:** an 8B model in 4-bit with LoRA (not full fine-tuning) should fit on a free-tier T4 (16GB) with small per-device batch size + gradient accumulation, though it'll be slow. If you have Colab Pro / A100 access, use it — meaningfully faster iteration, and you'll likely want 2-3 training runs (first pass, fix issues, final run) not one.

---

## Stage D — LoRA config (starting point, tune from here)

```python
lora_config = dict(
    r=16,                      # 32 if you have headroom and want more capacity
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    bias="none",
)

training_args = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch 16
    learning_rate=1e-4,              # try 2e-4 if underfitting after 1st run
    num_train_epochs=3,              # watch val loss — ~3k examples overfits fast
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=4096,             # cover windowed prefix + step text comfortably
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    logging_steps=10,
)
```

With only ~3,000 step-examples from 110ish trajectories, **overfitting is the main risk, not underfitting** — watch validation loss closely and stop before it diverges from training loss. This is a small fine-tuning dataset; don't expect it to need many epochs.

---

## Stage E — Save, convert, deploy back into your MLX pipeline

This is the step most likely to get skipped and cause confusion later: **you're training on CUDA/HF in Colab, but your inference pipeline runs on MLX on Apple Silicon.** You need an explicit conversion step, not just "download the adapter":

1. In Colab: `merged_model = model.merge_and_unload()` (PEFT) to fuse the LoRA weights into the base model, then save as a standard HF checkpoint (`save_pretrained`).
2. Transfer the merged HF checkpoint to the Mac (or wherever you run `mlx-lm`).
3. Convert to MLX format with quantization: `mlx_lm.convert --hf-path <merged_checkpoint> --mlx-path <output_dir> -q --q-bits 4` — this produces a drop-in replacement directory you can point your existing pipeline at instead of `mlx-community/Qwen3-8B-4bit`.
4. Sanity-check a handful of steps by hand before running the full eval — confirm the fine-tuned model still produces well-formed `MODE:`/`EVIDENCE:` output (LoRA fine-tuning can occasionally destabilize output formatting if the learning rate is too high).

---

## Stage F — Evaluation

Re-run your existing window=0 and window=3 reports with the fine-tuned model swapped in for Stage 1, **on the untouched 56-trajectory set**, and compare directly against your current baseline on:
- Step flag rate (target: much closer to the true ~4% mistake rate, not 70%)
- Recall (watch that it doesn't collapse — a model trained to stop over-flagging could overcorrect toward under-flagging)
- Whether the 2.1 "Next speaker X" artifact is gone (spot-check evidence text for any 2.1 flags, same way I checked it on your current data)
- Downstream joint accuracy through the (already-improved) Stage 2, to see the fine-tune's effect on the number that actually matters

If you held out a sample of the public MAST corpus per the earlier discussion, this is also where you'd report cross-framework generalization, separately from the in-domain 56.

---

## Checklist before you start

- [ ] Confirm exactly which of the 128 trajectories have any existing ground truth (even partial) — needed for the label-QA sanity check in A4
- [ ] Confirm current OpenAI model + pricing at `platform.openai.com/docs/models` (not from this plan)
- [ ] Confirm Unsloth's current Qwen3-8B support before committing to it over plain HF/PEFT
- [ ] Triple-check the 56 held-out trajectory IDs are excluded from every step of Stage A–D
- [ ] Keep the current prompted pipeline's window=3 report as the baseline to beat — that's your comparison point in Stage F
