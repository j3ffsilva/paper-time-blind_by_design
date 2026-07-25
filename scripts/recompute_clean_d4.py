"""
Recomputa D4 (continuation) com um probe ajustado apenas em t0-t7, corrigindo
o vazamento de periodo identificado na auditoria: Evaluator._build_splits()
ajusta o probe em TODO o split 'test' (763 linhas), que inclui 148 linhas de
t8/t9 (19.4%) -- o mesmo periodo que D4 afirma ser "unseen".

Reaproveita os 31+31 checkpoints ja treinados (outputs/runs/, oracle_runs/);
NAO retreina nada. So corrige o probe e reavalia continuation.

Uso:
  python scripts/recompute_clean_d4.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
import torch

from src.timeformer.dataset import (
    load_corpus, MLMDataset, make_continuation_split,
)
from src.timeformer.models import build_model
from src.timeformer.train import load_checkpoint
from src.timeformer.probe import LinearProbe, extract_reps
from src.timeformer.run import RunManager

CORPUS_PATH = Path("data/corpus.tsv")
RAW_PATH = Path("outputs/multiseed/multiseed_raw.json")
OUT_PATH = Path("outputs/multiseed/clean_d4_stats.json")

MODELS = ["Static", "Additive", "Joint", "Timeformer"]


def mean_ci(values: list[float]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    return {"n": n, "mean": mean, "ci95_lo": mean - 1.96 * se, "ci95_hi": mean + 1.96 * se}


def paired_ci(diffs: list[float]) -> dict:
    arr = np.array(diffs, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    if n > 1:
        t_stat, p = stats.ttest_1samp(arr, 0.0)
    else:
        t_stat, p = float("nan"), float("nan")
    return {"n": n, "mean": mean, "ci95_lo": mean - 1.96 * se, "ci95_hi": mean + 1.96 * se,
            "t_stat": float(t_stat), "p": float(p)}


def main() -> None:
    rows = load_corpus(CORPUS_PATH)
    _, cont_rows = make_continuation_split(rows)
    test_rows_clean = [
        r for r in rows if r["split"] == "test" and r["epoch_idx"] not in (8, 9)
    ]
    print(f"clean probe fit set (t0-t7 'test'): {len(test_rows_clean)} rows")
    print(f"continuation eval set (t8-t9, disjoint from 'train'): {len(cont_rows)} rows")

    clean_ds = MLMDataset(test_rows_clean)
    cont_ds = MLMDataset(cont_rows)

    raw = json.loads(RAW_PATH.read_text())

    per_model: dict[str, list[float]] = {m: [] for m in MODELS}
    per_model["Timeformer_oracle"] = []
    joint_vals: list[float] = []
    timeformer_vals: list[float] = []
    oracle_vals: list[float] = []

    for i, record in enumerate(raw):
        run_id = record["run_id"]
        seed = record["seed"]
        run = RunManager.load(run_id)
        print(f"[{i+1}/{len(raw)}] {run_id} (seed={seed})")

        seed_results = {}
        for name in MODELS:
            model = build_model(name)
            load_checkpoint(model, f"outputs/runs/{run_id}/{name}/best.pt")
            model.eval()
            memory = run.load_memory(name) if name == "Timeformer" else None

            clean_reps = extract_reps(model, clean_ds, memory, 256, "cpu")
            cont_reps = extract_reps(model, cont_ds, memory, 256, "cpu")
            probe = LinearProbe().fit(clean_reps["h_subj"], clean_reps["true_context"])
            acc = probe.evaluate(cont_reps["h_subj"], cont_reps["true_context"])["accuracy"]
            per_model[name].append(acc)
            seed_results[name] = acc

        joint_vals.append(seed_results["Joint"])
        timeformer_vals.append(seed_results["Timeformer"])

        # Timeformer-oracle: mesma memoria deterministica de sempre
        oracle_ckpt = Path(f"outputs/multiseed/oracle_runs/seed_{seed:04d}/best.pt")
        if oracle_ckpt.exists():
            import sys
            sys.path.insert(0, "scripts")
            from run_multiseed_oracle import get_oracle_memory
            from src.timeformer.models import DEFAULT_HPARAMS

            oracle_mem = get_oracle_memory(DEFAULT_HPARAMS["d_model"], "cpu")
            oracle_model = build_model("Timeformer")
            load_checkpoint(oracle_model, oracle_ckpt)
            oracle_model.eval()

            clean_reps_o = extract_reps(oracle_model, clean_ds, oracle_mem, 256, "cpu")
            cont_reps_o = extract_reps(oracle_model, cont_ds, oracle_mem, 256, "cpu")
            probe_o = LinearProbe().fit(clean_reps_o["h_subj"], clean_reps_o["true_context"])
            acc_o = probe_o.evaluate(cont_reps_o["h_subj"], cont_reps_o["true_context"])["accuracy"]
            per_model["Timeformer_oracle"].append(acc_o)
            oracle_vals.append(acc_o)
        else:
            print(f"  aviso: sem checkpoint oracle para seed {seed}")

    summary = {name: mean_ci(vals) for name, vals in per_model.items()}
    summary["delta_timeformer_vs_joint"] = paired_ci(
        [t - j for t, j in zip(timeformer_vals, joint_vals)]
    )
    if oracle_vals:
        summary["delta_oracle_vs_joint"] = paired_ci(
            [o - j for o, j in zip(oracle_vals, joint_vals[: len(oracle_vals)])]
        )

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nEscrito em {OUT_PATH}\n")
    for name, s in summary.items():
        if "mean" in s:
            hw = (s["ci95_hi"] - s["ci95_lo"]) / 2
            print(f"  {name:24s} {s['mean']:.4f} [±{hw:.4f}]  (n={s['n']})")
        else:
            print(f"  {name:24s} diff={s['mean']:+.4f} [{s['ci95_lo']:+.4f},{s['ci95_hi']:+.4f}] p={s['p']:.4g}")


if __name__ == "__main__":
    main()
