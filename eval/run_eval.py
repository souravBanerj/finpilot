"""Self-evaluation. Run:  python -m eval.run_eval [--seeds 20]
Writes eval/results.md and eval/results.json. Numbers in the README come from here.
LLM layer is only evaluated if MESH_API_KEY is set; otherwise the offline (rules+recurrence) result is reported.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from agents.analyzer import analyze
from agents.categorizer import CATEGORIES, categorize
from agents.forecaster import backtest
from eval.generate_data import generate_persona

OUT = Path(__file__).parent


def prf(y_true, y_pred, labels):
    rows = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rows[c] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
    return rows


def eval_categorization(personas, use_llm):
    yt, yp, meth = [], [], []
    for df, _ in personas:
        out = categorize(df[["id", "date", "description", "amount"]], use_llm=use_llm)
        yt += df["true_category"].tolist(); yp += out["category"].tolist(); meth += out["method"].tolist()
    acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
    labels = [c for c in CATEGORIES if c in set(yt)]
    per = prf(yt, yp, labels)
    macro = float(np.mean([v["f1"] for v in per.values()]))
    by_method = {}
    for m in sorted(set(meth)):
        idx = [i for i, x in enumerate(meth) if x == m]
        by_method[m] = {"share": len(idx) / len(meth), "accuracy": float(np.mean([yt[i] == yp[i] for i in idx]))}
    errs = defaultdict(int)
    for t, p in zip(yt, yp):
        if t != p:
            errs[f"{t} -> {p}"] += 1
    return {"n": len(yt), "accuracy": acc, "macro_f1": macro, "per_class": per, "by_method": by_method,
            "top_errors": dict(sorted(errs.items(), key=lambda kv: -kv[1])[:8])}


def eval_anomalies(personas):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for df, truth in personas:
        out = categorize(df[["id", "date", "description", "amount"]], use_llm=False)
        out["id"] = df["id"].values
        res = analyze(out)
        F = res["findings"]
        # outliers
        det = {f["txn_id"] for f in F if f["type"] == "txn_outlier"}
        gt = set(truth["outlier_ids"])
        tp["txn_outlier"] += len(det & gt); fp["txn_outlier"] += len(det - gt); fn["txn_outlier"] += len(gt - det)
        # duplicates: a finding is a TP if it names a transaction from a ground-truth pair
        dets = [set(f["evidence"]["txn_ids"]) for f in F if f["type"] == "duplicate_charge"]
        pairs = [set(p) for p in truth["duplicate_pairs"]]
        hit_pairs = {i for i, p in enumerate(pairs) if any(d & p for d in dets)}
        good_dets = sum(1 for d in dets if any(d & p for p in pairs))
        tp["duplicate_charge"] += len(hit_pairs); fn["duplicate_charge"] += len(pairs) - len(hit_pairs)
        fp["duplicate_charge"] += len(dets) - good_dets
        # category spike (latest month only; truth = Food Delivery)
        det = {(f["category"], f["month"]) for f in F if f["type"] == "category_spike"}
        gt = {(truth["spike"]["category"], truth["spike"]["month"])}
        tp["category_spike"] += len(det & gt); fp["category_spike"] += len(det - gt); fn["category_spike"] += len(gt - det)
        # subscription overlap groups
        det = {f["group"] for f in F if f["type"] == "subscription_overlap"}
        gt = set(truth["overlap_groups"])
        tp["subscription_overlap"] += len(det & gt); fp["subscription_overlap"] += len(det - gt); fn["subscription_overlap"] += len(gt - det)
        # price hike (Netflix hike is always injected inside the last 6 charges)
        det = {f["merchant"] for f in F if f["type"] == "subscription_price_hike"}
        gt = {truth["price_hike_merchant"]}
        tp["subscription_price_hike"] += len(det & gt); fp["subscription_price_hike"] += len(det - gt); fn["subscription_price_hike"] += len(gt - det)
        # hidden (unknown-name) subscriptions found via recurrence
        found = set(out[(out["category"] == "Subscriptions")]["merchant"])
        gt = {"PIXELCLOUD", "ZENITH LEARN"}
        got = {m for m in found if any(h in m for h in ("PIXELCLOUD", "ZENITH"))}
        tp["hidden_subscription_caught"] += len(got); fn["hidden_subscription_caught"] += len(gt) - len(got)
    out = {}
    for k in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else None
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else None
        out[k] = {"tp": tp[k], "fp": fp[k], "fn": fn[k], "precision": p, "recall": r}
    return out


def spike_sweep(personas, sigmas=(0.0, 1.0, 1.5, 2.0, 2.5)):
    """Precision/recall of the Food-Delivery spike detector as the volatility threshold changes."""
    from app import config
    orig, rows = config.SPIKE_SIGMA, {}
    cats = [categorize(df[["id", "date", "description", "amount"]], use_llm=False).assign(id=df["id"].values)
            for df, _ in personas]
    for sg in sigmas:
        config.SPIKE_SIGMA = sg
        tp = fp = fn = 0
        for o, (_, t) in zip(cats, personas):
            det = {(f["category"], f["month"]) for f in analyze(o)["findings"] if f["type"] == "category_spike"}
            gt = {(t["spike"]["category"], t["spike"]["month"])}
            tp += len(det & gt); fp += len(det - gt); fn += len(gt - det)
        p, r = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
        rows[sg] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0}
    config.SPIKE_SIGMA = orig
    return rows


def eval_forecast(personas):
    keys = ["total_expense_mape_model", "total_expense_mape_naive_last", "total_expense_mape_mean3",
            "income_mape", "interval_80_coverage"]
    acc = {k: [] for k in keys}
    cat_m, cat_n = defaultdict(list), defaultdict(list)
    for df, _ in personas:
        out = categorize(df[["id", "date", "description", "amount"]], use_llm=False)
        bt = backtest(out)
        for k in keys:
            acc[k].append(bt[k])
        for c, v in bt["per_category_mape_model"].items():
            cat_m[c].append(v)
        for c, v in bt["per_category_mape_naive"].items():
            cat_n[c].append(v)
    res = {k: round(float(np.mean(v)), 2) for k, v in acc.items()}
    res["per_category_mape_model"] = {c: round(float(np.mean(v)), 1) for c, v in cat_m.items()}
    res["per_category_mape_naive"] = {c: round(float(np.mean(v)), 1) for c, v in cat_n.items()}
    return res


def fmt(x):
    return "n/a" if x is None else f"{x*100:.0f}%"


def main(n_seeds: int):
    from app import llm
    personas = [generate_persona(s) for s in range(n_seeds)]
    cat_off = eval_categorization(personas, use_llm=False)
    cat_llm = eval_categorization(personas[:5], use_llm=True) if llm.available() else None
    anom = eval_anomalies(personas)
    fc = eval_forecast(personas)
    sweep = spike_sweep(personas)
    json.dump({"categorization_offline": cat_off, "categorization_llm": cat_llm, "anomalies": anom, "forecast": fc,
               "spike_sensitivity": sweep},
              open(OUT / "results.json", "w", encoding="utf-8"), indent=2, default=float)

    L = [f"# FinPilot self-evaluation\n",
         f"Synthetic data: **{n_seeds} seeded personas × 12 months** ({cat_off['n']} transactions) with ground-truth "
         f"labels (`eval/generate_data.py`). Regenerate with `python -m eval.run_eval`.\n",
         "## 1. Categorization (offline: memory-free rules + recurrence, no LLM)\n",
         f"- **Accuracy: {cat_off['accuracy']*100:.1f}%**   **Macro-F1: {cat_off['macro_f1']:.3f}**\n",
         "| Layer | Share of txns | Accuracy |\n|---|---|---|"]
    for m, v in cat_off["by_method"].items():
        L.append(f"| {m} | {v['share']*100:.1f}% | {v['accuracy']*100:.1f}% |")
    L += ["\n| Category | Precision | Recall | F1 | Support |\n|---|---|---|---|---|"]
    for c, v in cat_off["per_class"].items():
        L.append(f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
    L.append("\nTop confusions (honest failure modes): " + "; ".join(f"`{k}` ×{v}" for k, v in cat_off["top_errors"].items()))
    if cat_llm:
        L.append(f"\nWith the Mesh LLM fallback (5 personas): accuracy **{cat_llm['accuracy']*100:.1f}%**, "
                 f"macro-F1 **{cat_llm['macro_f1']:.3f}**.")
    else:
        L.append("\n_LLM fallback not evaluated in this run (no MESH_API_KEY). Unknown merchants land in `Other` here; "
                 "set the key and re-run to measure the LLM layer._")
    L += ["\n## 2. Anomaly / pattern detection (injected ground truth)\n",
          "| Detector | TP | FP | FN | Precision | Recall |\n|---|---|---|---|---|---|"]
    for k, v in anom.items():
        L.append(f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {fmt(v['precision'])} | {fmt(v['recall'])} |")
    from app import config
    L += [f"\n**Spike-detector sensitivity** (threshold = std-devs above the category's own history; production default "
          f"= {config.SPIKE_SIGMA}). Monthly order counts are noisy (~35%), so this is a genuine precision/recall trade-off:\n",
          "| σ threshold | Precision | Recall | F1 |\n|---|---|---|---|"]
    for sg, v in sweep.items():
        L.append(f"| {sg} | {v['precision']*100:.0f}% | {v['recall']*100:.0f}% | {v['f1']:.2f} |")
    L += ["\n## 3. Forecast backtest (rolling origin, train on months 1..t-1, predict month t)\n",
          "| Method (total monthly expenses) | MAPE |\n|---|---|",
          f"| **FinPilot model** | **{fc['total_expense_mape_model']}%** |",
          f"| Naive (last month) | {fc['total_expense_mape_naive_last']}% |",
          f"| Mean of last 3 months | {fc['total_expense_mape_mean3']}% |",
          f"\n- Income MAPE: {fc['income_mape']}%",
          f"- 80% prediction-band coverage: **{fc['interval_80_coverage']}%** (target 80%)\n",
          "| Category | Model MAPE | Naive MAPE |\n|---|---|---|"]
    for c, v in sorted(fc["per_category_mape_model"].items()):
        L.append(f"| {c} | {v}% | {fc['per_category_mape_naive'].get(c, 'n/a')}% |")
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 and cannot print σ / ₹
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    main(ap.parse_args().seeds)
