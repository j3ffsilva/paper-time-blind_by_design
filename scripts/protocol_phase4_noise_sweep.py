#!/usr/bin/env python3
"""
Phase 4A of the restructured Timeformer evaluation protocol.

Noise/fidelity sweep for Additive vs Token-Time.

This script is intentionally self-contained and writes all generated corpora,
checkpoints, and metrics under outputs/protocol/phase4_noise_sweep so it does
not overwrite data/corpus.tsv or the 31-seed baseline runs.

Default run is a modest pilot: fidelities {0.75, 0.625, 0.50} and 3 seeds.
Increase --seeds/--fidelities for the camera-ready experiment.

Outputs:
  outputs/protocol/phase4_noise_sweep_results.csv
  outputs/protocol/phase4_noise_sweep_summary.csv
  outputs/protocol/phase4_noise_sweep.json
  tmp/protocol_phase4_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from src.corpus_generator import generate_ambiguous_eval, generate_corpus_v2
from src.timeformer.dataset import MLMDataset, load_corpus, make_continuation_split
from src.timeformer.eval import Evaluator
from src.timeformer.models import build_model
from src.timeformer.train import MLMTrainer

from scripts.neighbor_analysis import context_drift_score_by_class
from scripts.protocol_phase1_layerwise_cka import DISPLAY, extract_layerwise_subject_reps
from scripts.protocol_phase2_output_diagnostics import evaluate_model_reps, load_planted


OUT_ROOT = Path("outputs/protocol/phase4_noise_sweep")
SUMMARY_MD = Path("tmp/protocol_phase4_summary.md")
RESULTS_CSV = Path("outputs/protocol/phase4_noise_sweep_results.csv")
SUMMARY_CSV = Path("outputs/protocol/phase4_noise_sweep_summary.csv")
JSON_PATH = Path("outputs/protocol/phase4_noise_sweep.json")
MODELS = ("Additive", "Joint")


def parse_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def mean_ci(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = sd / math.sqrt(len(arr)) if len(arr) else float("nan")
    return {
        "n": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def generate_paths(fidelity: float, seed: int) -> dict[str, Path]:
    f_tag = f"f{fidelity:.3f}".replace(".", "p")
    run_dir = OUT_ROOT / f_tag / f"seed_{seed:04d}"
    data_dir = run_dir / "data"
    return {
        "run_dir": run_dir,
        "data_dir": data_dir,
        "corpus": data_dir / "corpus.tsv",
        "ambiguous": data_dir / "corpus_ambiguous.tsv",
        "contrastive": data_dir / "contrastive_missing.tsv",
    }


def prepare_corpus(fidelity: float, seed: int, force: bool = False) -> dict[str, Path]:
    paths = generate_paths(fidelity, seed)
    if force and paths["run_dir"].exists():
        shutil.rmtree(paths["run_dir"])
    paths["data_dir"].mkdir(parents=True, exist_ok=True)

    if not paths["corpus"].exists():
        _, fractions, _ = generate_corpus_v2(
            output_path=paths["corpus"],
            p_canon=fidelity,
            seed=seed,
        )
        generate_ambiguous_eval(
            output_path=paths["ambiguous"],
            fractions=fractions,
            seed=seed,
        )
    return paths


def train_one_model(
    model_name: str,
    corpus_path: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    force: bool = False,
) -> torch.nn.Module:
    ckpt = output_dir / "best.pt"
    model = build_model(model_name)
    if ckpt.exists() and not force:
        from src.timeformer.train import load_checkpoint
        load_checkpoint(model, ckpt)
        model.to(torch.device(device))
        model.eval()
        return model

    rows = load_corpus(corpus_path)
    train_rows, _ = make_continuation_split(rows)
    val_rows = [r for r in rows if r["split"] == "test"]
    train_ds = MLMDataset(train_rows, seed=seed)
    val_ds = MLMDataset(val_rows, seed=seed)

    trainer = MLMTrainer(model, output_dir=output_dir, device=device)
    trainer.train(
        train_ds,
        val_ds,
        memory=None,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    model.eval()
    return model


def drift_delta(model: torch.nn.Module, dataset: MLMDataset, device: str) -> float:
    from src.timeformer.probe import extract_reps

    reps = extract_reps(model, dataset, memory=None, batch_size=256, device=device)
    drift_by_class = context_drift_score_by_class(reps, k=10)
    drift = drift_by_class["drift"]
    return float(drift[9] - drift[0])


def phase2_metrics_for_model(
    run_key: str,
    model_name: str,
    model: torch.nn.Module,
    dataset: MLMDataset,
    params_path: Path,
    device: str,
) -> dict[str, float]:
    reps = extract_layerwise_subject_reps(
        model,
        dataset,
        memory=None,
        batch_size=256,
        device=torch.device(device),
    )
    rows = evaluate_model_reps(
        run_id=run_key,
        model_name=model_name,
        reps=reps,
        dataset=dataset,
        planted=load_planted(params_path),
    )
    return {row["metric"]: float(row["value"]) for row in rows}


def evaluate_run(
    fidelity: float,
    seed: int,
    paths: dict[str, Path],
    models: dict[str, torch.nn.Module],
    device: str,
) -> list[dict]:
    evaluator = Evaluator(
        corpus_path=paths["corpus"],
        ambiguous_path=paths["ambiguous"],
        contrastive_path=paths["contrastive"],
        device=device,
        batch_size=256,
    )
    test_rows = [r for r in load_corpus(paths["corpus"]) if r["split"] == "test"]
    test_ds = MLMDataset(test_rows, seed=42)
    params_path = paths["corpus"].with_suffix(".params.json")

    rows: list[dict] = []
    for model_name, model in models.items():
        full = evaluator.evaluate(model, memory=None)
        d2 = drift_delta(model, test_ds, device)
        phase2 = phase2_metrics_for_model(
            run_key=f"f{fidelity:.3f}_seed{seed}",
            model_name=model_name,
            model=model,
            dataset=test_ds,
            params_path=params_path,
            device=device,
        )
        rows.append({
            "fidelity": fidelity,
            "noise": 1.0 - fidelity,
            "seed": seed,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "drift_delta": d2,
            "ambiguous_accuracy": full["ambiguous_test"]["probe_subj"]["accuracy"],
            "test_accuracy": full["test"]["probe_subj"]["accuracy"],
            "path_contrast": phase2["path_contrast_drift_minus_stable"],
            "directed_contrast": phase2["directed_contrast_drift_minus_stable"],
            "spearman_drift": phase2["spearman_drift"],
            "spearman_bifurc": phase2["spearman_bifurc"],
        })
    return rows


def paired_gap_rows(rows: list[dict]) -> list[dict]:
    metrics = [
        "drift_delta",
        "ambiguous_accuracy",
        "test_accuracy",
        "path_contrast",
        "directed_contrast",
        "spearman_drift",
        "spearman_bifurc",
    ]
    by_key = {(r["fidelity"], r["seed"], r["model"]): r for r in rows}
    gaps = []
    for fidelity, seed in sorted({(r["fidelity"], r["seed"]) for r in rows}):
        add = by_key.get((fidelity, seed, "Additive"))
        joint = by_key.get((fidelity, seed, "Joint"))
        if not add or not joint:
            continue
        for metric in metrics:
            gaps.append({
                "fidelity": fidelity,
                "noise": 1.0 - fidelity,
                "seed": seed,
                "comparison": "Token-Time - Additive",
                "metric": metric,
                "value": float(joint[metric] - add[metric]),
            })
    return gaps


def summarize_gaps(gaps: list[dict]) -> tuple[list[dict], list[dict]]:
    by_level: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in gaps:
        by_level[(row["fidelity"], row["metric"])].append(row["value"])

    summary = []
    for (fidelity, metric), values in sorted(by_level.items()):
        summary.append({
            "fidelity": fidelity,
            "noise": 1.0 - fidelity,
            "metric": metric,
            **mean_ci(values),
        })

    trend_rows = []
    for metric in sorted({g["metric"] for g in gaps}):
        metric_gaps = [g for g in gaps if g["metric"] == metric]

        # Primary inference: one slope per seed (regressed across its own
        # fidelity levels), then a paired one-sample t-test on those slopes.
        # Pooling all seed×fidelity rows into a single OLS (as before)
        # pseudo-replicates repeated measures on the same seed and inflates
        # degrees of freedom — each seed appears at every fidelity level, so
        # observations are not independent.
        by_seed: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for g in metric_gaps:
            by_seed[g["seed"]].append((g["noise"], g["value"]))

        seed_slopes = []
        for seed, pts in sorted(by_seed.items()):
            pts.sort()
            xs_s = np.array([p[0] for p in pts], dtype=float)
            ys_s = np.array([p[1] for p in pts], dtype=float)
            if len(np.unique(xs_s)) >= 2:
                s_seed, *_ = stats.linregress(xs_s, ys_s)
                seed_slopes.append(s_seed)
        seed_slopes = np.array(seed_slopes, dtype=float)

        n = len(seed_slopes)
        if n >= 2:
            mean_slope = float(seed_slopes.mean())
            sd = float(seed_slopes.std(ddof=1))
            se = sd / math.sqrt(n)
            t_stat, p_paired = stats.ttest_1samp(seed_slopes, 0.0)
            ci_low = mean_slope - stats.t.ppf(0.975, n - 1) * se
            ci_high = mean_slope + stats.t.ppf(0.975, n - 1) * se
            dz = mean_slope / sd if sd > 0 else float("nan")
        else:
            mean_slope = sd = se = ci_low = ci_high = dz = float("nan")
            t_stat, p_paired = float("nan"), float("nan")

        # Pooled OLS kept for reference only (not the primary inference —
        # see note above); point estimate matches the paired mean slope
        # under this balanced design, but its p-value is not valid.
        xs = np.array([g["noise"] for g in metric_gaps], dtype=float)
        ys = np.array([g["value"] for g in metric_gaps], dtype=float)
        if len(np.unique(xs)) >= 2:
            pooled_slope, intercept, r, p_pooled, pooled_se = stats.linregress(xs, ys)
        else:
            pooled_slope = intercept = r = p_pooled = pooled_se = float("nan")

        trend_rows.append({
            "metric": metric,
            "n_seeds": n,
            "slope_vs_noise": mean_slope,
            "slope_ci95_low": float(ci_low),
            "slope_ci95_high": float(ci_high),
            "slope_sd": sd,
            "cohens_dz": float(dz),
            "t_stat": float(t_stat),
            "p_paired": float(p_paired),
            "pooled_slope_vs_noise": float(pooled_slope),
            "pooled_p_two_sided": float(p_pooled),
        })
    return summary, trend_rows


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: list[dict], trends: list[dict], path: Path) -> None:
    focus = ["ambiguous_accuracy", "directed_contrast", "drift_delta", "spearman_drift"]
    lines = [
        "# Fase 4A — Varredura de fidelidade",
        "",
        "Gap = Token-Time − Additive. Slope positivo indica gap aumentando conforme o ruído cresce.",
        "",
    ]
    for metric in focus:
        lines.append(f"## {metric}")
        for row in [r for r in summary if r["metric"] == metric]:
            lines.append(
                f"- fidelity={row['fidelity']:.3f}: {row['mean']:+.4f} "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )
        trend = next((t for t in trends if t["metric"] == metric), None)
        if trend:
            lines.append(
                f"- paired slope vs noise (n={trend['n_seeds']} seeds): "
                f"{trend['slope_vs_noise']:+.4f} "
                f"[{trend['slope_ci95_low']:+.4f}, {trend['slope_ci95_high']:+.4f}], "
                f"dz={trend['cohens_dz']:.3f}, p={trend['p_paired']:.4g}"
            )
            lines.append(
                f"- (pooled OLS, reference only — pseudo-replicated: "
                f"slope={trend['pooled_slope_vs_noise']:+.4f}, "
                f"p={trend['pooled_p_two_sided']:.4g})"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fidelities", default="0.75,0.625,0.50")
    parser.add_argument("--seeds", default="1000,1001,1002")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    fidelities = parse_floats(args.fidelities)
    seeds = parse_ints(args.seeds)

    all_rows: list[dict] = []
    for fidelity in fidelities:
        for seed in seeds:
            print(f"\n=== fidelity={fidelity:.3f} seed={seed} ===")
            paths = prepare_corpus(fidelity, seed, force=args.force)
            models = {}
            for model_name in MODELS:
                model_dir = paths["run_dir"] / model_name
                if args.eval_only and not (model_dir / "best.pt").exists():
                    raise FileNotFoundError(f"Missing checkpoint for eval-only: {model_dir / 'best.pt'}")
                print(f"--- {DISPLAY[model_name]} ---")
                models[model_name] = train_one_model(
                    model_name=model_name,
                    corpus_path=paths["corpus"],
                    output_dir=model_dir,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    force=args.force and not args.eval_only,
                )
            all_rows.extend(evaluate_run(fidelity, seed, paths, models, args.device))

    gaps = paired_gap_rows(all_rows)
    summary, trends = summarize_gaps(gaps)

    write_csv(
        all_rows,
        RESULTS_CSV,
        ["fidelity", "noise", "seed", "model", "model_label", "drift_delta",
         "ambiguous_accuracy", "test_accuracy", "path_contrast",
         "directed_contrast", "spearman_drift", "spearman_bifurc"],
    )
    write_csv(
        summary,
        SUMMARY_CSV,
        ["fidelity", "noise", "metric", "n", "mean", "sd", "ci95_low", "ci95_high"],
    )
    JSON_PATH.write_text(json.dumps({
        "fidelities": fidelities,
        "seeds": seeds,
        "epochs": args.epochs,
        "rows": all_rows,
        "gaps": gaps,
        "summary": summary,
        "trends": trends,
    }, indent=2), encoding="utf-8")
    write_markdown(summary, trends, SUMMARY_MD)

    print(f"\nWrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
