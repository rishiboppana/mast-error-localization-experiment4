"""
Reproduces the exact 184 -> 128 train / 56 held-out trajectory split used by
Experiment3/build_from_scratch_taxonomy.ipynb (cell 16: sklearn train_test_split,
test_size=0.30, random_state=SEED=0, stratified by subset), from the raw
trajectory files already downloaded to
Experiment3/whodunit_experiment3_v2/raw_trajectories/{Hand-Crafted,Algorithm-Generated}/*.json

Verifies the reproduced held-out set matches both saved held_out_test_set.json
files exactly (by question_ID) before writing anything -- if this check fails,
STOP, because it means the split assumptions here (order/seed/stratify) drifted
from the actual saved split, and using this file's "train" partition could leak
held-out trajectories into fine-tuning.

IMPORTANT -- question_ID is NOT unique across the two subsets: 45 question_IDs
appear in both Hand-Crafted and Algorithm-Generated (Algorithm-Generated
re-runs the same underlying task/question through a different agent
trajectory). 21 of those land one copy in train and one copy in held-out. A
model fine-tuned on the train-side copy would have seen the exact same
task/question as a held-out eval item, just with a different trajectory --
that's task-level contamination, the same category of mistake the plan's
"hard constraint" warns about, so those 21 question_IDs are dropped from the
training pool entirely (not just deduplicated). Final pool: 128 - 21 = 107
trajectories.

Writes train_pool.json in this folder: a JSON list of the training
trajectories, in the same schema stage_ab_teacher_labeling.py expects
(question_ID, question, history, mistake_agent, mistake_step, mistake_reason).
"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

EXPERIMENT3 = Path(__file__).resolve().parent.parent / "Experiment3"
RAW_DIR = EXPERIMENT3 / "whodunit_experiment3_v2" / "raw_trajectories"
HELD_OUT_CHECK_PATHS = [
    EXPERIMENT3 / "whodunit_experiment3_v2" / "held_out_test_set.json",
    EXPERIMENT3 / "whodunit_experiment3" / "held_out_test_set.json",
]
TRAJECTORY_SETS = ["Hand-Crafted", "Algorithm-Generated"]
SEED = 0
TEST_FRACTION = 0.30
OUT_PATH = Path(__file__).resolve().parent / "train_pool.json"


def sort_key(p: Path):
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def main():
    trajectories = []
    for subset in TRAJECTORY_SETS:
        subset_dir = RAW_DIR / subset
        assert subset_dir.exists(), f"missing {subset_dir} -- is whodunit_experiment3_v2/raw_trajectories present?"
        for p in sorted(subset_dir.glob("*.json"), key=sort_key):
            raw = json.loads(p.read_text())
            raw["subset"] = subset
            raw["file"] = p.name
            trajectories.append(raw)
    print(f"loaded {len(trajectories)} raw trajectories from {RAW_DIR}")

    idx = list(range(len(trajectories)))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_FRACTION, random_state=SEED,
        stratify=[t["subset"] for t in trajectories],
    )
    train = [trajectories[i] for i in sorted(train_idx)]
    held_out = [trajectories[i] for i in sorted(test_idx)]
    print(f"reproduced split: train={len(train)}  held_out={len(held_out)}")

    reproduced_ids = {t["question_ID"] for t in held_out}
    for check_path in HELD_OUT_CHECK_PATHS:
        if not check_path.exists():
            print(f"  (skip check, not found: {check_path})")
            continue
        saved_ids = {t["question_ID"] for t in json.loads(check_path.read_text())}
        match = saved_ids == reproduced_ids
        print(f"  matches {check_path.relative_to(EXPERIMENT3.parent)}: {match}")
        assert match, (
            f"Reproduced held-out set does NOT match {check_path}. "
            "Do not use this script's train partition until this is fixed -- "
            "it may leak held-out trajectories into the fine-tuning pool."
        )

    # question_ID is not unique across subsets (Algorithm-Generated reuses Hand-Crafted's
    # underlying questions) -- drop any train trajectory whose question_ID also appears
    # anywhere in either saved held-out file, to avoid task-level contamination.
    all_held_out_ids = set(reproduced_ids)
    for check_path in HELD_OUT_CHECK_PATHS:
        if check_path.exists():
            all_held_out_ids |= {t["question_ID"] for t in json.loads(check_path.read_text())}

    contaminated = [t for t in train if t["question_ID"] in all_held_out_ids]
    clean_train = [t for t in train if t["question_ID"] not in all_held_out_ids]
    print(f"\ndropping {len(contaminated)} train trajectories whose question_ID also appears in held-out "
          f"(same underlying task, different agent trajectory)")
    for t in contaminated:
        print(f"  dropped: subset={t['subset']:20s} file={t['file']:12s} question_ID={t['question_ID'][:16]}...")

    clean_ids = {t["question_ID"] for t in clean_train}
    assert not (clean_ids & all_held_out_ids), "contamination check failed -- aborting"

    OUT_PATH.write_text(json.dumps(clean_train, indent=2))
    print(f"\nwrote {len(clean_train)} training trajectories -> {OUT_PATH}")
    print("Use this as --pool for stage_ab_teacher_labeling.py:")
    print(f"  python stage_ab_teacher_labeling.py --pool {OUT_PATH.name} --openai-model <model>")


if __name__ == "__main__":
    main()
