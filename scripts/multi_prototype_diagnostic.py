"""
Multi-Prototype Memory Diagnostic — item 2 do Review 1 (IBERAMIA camera-ready).

Testa uma alternativa ao mean pooling do Prototype Memory: em vez de
m(S, t) = mean_pool(h(S) nas frases de S na época t), guarda até n_modes=2
protótipos por (sujeito, época) via k-means (MultiPrototypeMemory).

Motivação: o paper atribui a falha do Memory-Augmented em sujeitos
bifurcating à agregação por média, que colapsa dois sentidos coexistentes
num centroide que não representa nenhum. Este diagnóstico testa se preservar
dois modos por época resolve isso, sem mudar o mecanismo de leitura (a
cross-attention temporal já existente, que não é mean pooling).

Compara, nos mesmos seeds pareados:
  Joint              (Token-Time, sem memória)
  Timeformer + PrototypeMemory       (Mem-Aug mean — reprodução do paper)
  Timeformer + MultiPrototypeMemory  (Mem-Aug multi-proto k=2 — novo)

em D3 (contrastive sign-flip) e D4 (continuation probe accuracy).

Uso:
  python scripts/multi_prototype_diagnostic.py --seeds 2000,2001,2002
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from src.timeformer.dataset import (
    MLMDataset, TimeformerDataset, ContrastiveDataset,
    load_corpus, make_continuation_split,
)
from src.timeformer.models import build_model, DEFAULT_HPARAMS
from src.timeformer.memory import PrototypeMemory, MultiPrototypeMemory
from src.timeformer.train import MLMTrainer
from src.timeformer.probe import LinearProbe, extract_reps, evaluate_contrastive

CORPUS_PATH      = Path("data/corpus.tsv")
CONTRASTIVE_PATH = Path("data/contrastive_set.tsv")
OUT_DIR          = Path("outputs/protocol/multi_prototype_diagnostic")
RESULTS_JSON     = OUT_DIR / "results.json"


def parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def mean_ci(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se}


def evaluate_model(model, memory, probe_fit_rows, cont_rows, device, batch_size=256) -> dict:
    """
    D4 (continuation probe accuracy) e D3 (contrastive sign-flip).

    probe_fit_rows deve ser restrito a t0-t7: ajustar em todo o split 'test'
    (que inclui t8-t9) vaza informacao de periodo para o probe antes de
    avalia-lo em 'continuation' (mesmos periodos, sentencas diferentes).
    """
    test_ds = MLMDataset(probe_fit_rows)
    cont_ds = MLMDataset(cont_rows)

    test_reps = extract_reps(model, test_ds, memory, batch_size, device)
    probe = LinearProbe().fit(test_reps["h_subj"], test_reps["true_context"])

    cont_reps = extract_reps(model, cont_ds, memory, batch_size, device)
    cont_metrics = probe.evaluate(cont_reps["h_subj"], cont_reps["true_context"])

    contrastive_ds = ContrastiveDataset(CONTRASTIVE_PATH)
    contrastive_metrics = evaluate_contrastive(model, contrastive_ds, memory, batch_size, device)

    return {
        "continuation_accuracy": cont_metrics["accuracy"],
        "sign_flip_rate": contrastive_metrics["sign_flip_rate"],
    }


def run_seed(seed: int, epochs: int, batch_size: int, lr: float, device: str) -> dict:
    rows = load_corpus(CORPUS_PATH)
    train_rows, _ = make_continuation_split(rows)
    # val_rows (validacao/checkpoint selection) mantem todo o split 'test',
    # igual ao pipeline principal -- nao corrigido nesta rodada (ver nota de
    # limitacao no paper). probe_fit_rows (D4) e restrito a t0-t7.
    val_rows       = [r for r in rows if r["split"] == "test"]
    probe_fit_rows = [r for r in val_rows if r["epoch_idx"] not in (8, 9)]
    _, cont_rows = make_continuation_split(rows)

    d_model = DEFAULT_HPARAMS["d_model"]
    results = {}

    # Reseed imediatamente antes de cada build_model(): sem isso, a
    # inicialização de pesos do 2o/3o modelo depende do estado global do RNG
    # deixado pelo treino anterior, não do `seed` do experimento — quebrando
    # o pareamento entre Timeformer_mean e Timeformer_multi (mesma
    # arquitetura, deveriam começar dos mesmos pesos).

    # ── Joint (Token-Time), sem memória ──────────────────────────────────
    print(f"  [seed {seed}] Joint (Token-Time)")
    torch.manual_seed(seed)
    joint = build_model("Joint")
    joint_ds = MLMDataset(train_rows, seed=seed)
    trainer = MLMTrainer(joint, output_dir=OUT_DIR / f"seed_{seed}" / "Joint", device=device)
    trainer.train(joint_ds, MLMDataset(val_rows, seed=seed), memory=None,
                  n_epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    joint.eval()
    results["Joint"] = evaluate_model(joint, None, probe_fit_rows, cont_rows, device)

    # ── Timeformer + PrototypeMemory (mean) — reprodução do Mem-Aug do paper ──
    print(f"  [seed {seed}] Timeformer + mean prototype")
    torch.manual_seed(seed)
    tf_mean = build_model("Timeformer")
    mem_mean = PrototypeMemory(d_model=d_model, device=device)
    tf_ds = TimeformerDataset(train_rows, seed=seed)
    trainer = MLMTrainer(tf_mean, output_dir=OUT_DIR / f"seed_{seed}" / "Timeformer_mean", device=device)
    trainer.train(tf_ds, MLMDataset(val_rows, seed=seed), memory=mem_mean,
                  n_epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    tf_mean.eval()
    results["Timeformer_mean"] = evaluate_model(tf_mean, mem_mean, probe_fit_rows, cont_rows, device)

    # ── Timeformer + MultiPrototypeMemory (k=2) — alternativa nova ──────────
    print(f"  [seed {seed}] Timeformer + multi-prototype (k=2)")
    torch.manual_seed(seed)
    tf_multi = build_model("Timeformer")
    mem_multi = MultiPrototypeMemory(d_model=d_model, n_modes=2, device=device, kmeans_seed=seed)
    tf_ds2 = TimeformerDataset(train_rows, seed=seed)
    trainer = MLMTrainer(tf_multi, output_dir=OUT_DIR / f"seed_{seed}" / "Timeformer_multi", device=device)
    trainer.train(tf_ds2, MLMDataset(val_rows, seed=seed), memory=mem_multi,
                  n_epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    tf_multi.eval()
    results["Timeformer_multi"] = evaluate_model(tf_multi, mem_multi, probe_fit_rows, cont_rows, device)

    return results


def paired_ttest(a: list[float], b: list[float]) -> dict:
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se_d = sd_d / math.sqrt(n) if n else float("nan")
    if n > 1:
        t_stat, p = stats.ttest_1samp(diffs, 0.0)
        ci_low = mean_d - stats.t.ppf(0.975, n - 1) * se_d
        ci_high = mean_d + stats.t.ppf(0.975, n - 1) * se_d
    else:
        t_stat = p = ci_low = ci_high = float("nan")
    return {"n": n, "mean_diff": mean_d, "ci95_low": ci_low, "ci95_high": ci_high,
            "t_stat": float(t_stat), "p": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2000,2001,2002")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_seed = {}
    for seed in seeds:
        print(f"\n=== seed={seed} ===")
        by_seed[seed] = run_seed(seed, args.epochs, args.batch_size, args.lr, args.device)

    models = ["Joint", "Timeformer_mean", "Timeformer_multi"]
    metrics = ["continuation_accuracy", "sign_flip_rate"]

    summary = {}
    for m in models:
        summary[m] = {}
        for metric in metrics:
            vals = [by_seed[s][m][metric] for s in seeds]
            summary[m][metric] = mean_ci(vals)

    comparisons = {}
    for metric in metrics:
        multi_vals = [by_seed[s]["Timeformer_multi"][metric] for s in seeds]
        mean_vals  = [by_seed[s]["Timeformer_mean"][metric] for s in seeds]
        joint_vals = [by_seed[s]["Joint"][metric] for s in seeds]
        comparisons[f"{metric}__multi_minus_mean"]  = paired_ttest(multi_vals, mean_vals)
        comparisons[f"{metric}__multi_minus_joint"] = paired_ttest(multi_vals, joint_vals)

    out = {"seeds": seeds, "epochs": args.epochs, "by_seed": by_seed,
           "summary": summary, "comparisons": comparisons}
    RESULTS_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Resumo ===")
    for m in models:
        print(f"  {m}:")
        for metric in metrics:
            s = summary[m][metric]
            print(f"    {metric}: {s['mean']:.4f} [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}] (n={s['n']})")

    print("\n=== Comparações pareadas ===")
    for key, c in comparisons.items():
        print(f"  {key}: diff={c['mean_diff']:+.4f} [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p={c['p']:.4g}")

    print(f"\nEscrito em {RESULTS_JSON}")


if __name__ == "__main__":
    main()
