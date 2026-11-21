"""Train, evaluate and save the RouteWise delay model.

Usage (repository root):
    python scripts/train.py                       # 6,000 synthetic shipments, seed 42
    python scripts/train.py --n 10000 --seed 7 --out artifacts
Writes artifacts/routewise_model.joblib (model + metadata) and artifacts/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from routewise.core import evaluate, generate_data, save_model  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts')
    a = ap.parse_args(argv)
    ev = evaluate(generate_data(a.n, a.seed))
    save_model(ev, a.out / 'routewise_model.joblib')
    summary = {'model': ev.model_name, 'threshold': ev.meta['threshold'],
               'holdout': {k: v for k, v in ev.metrics.items() if k != 'report'}, 'tuned_threshold': ev.tuned,
               'random_split': ev.random_split, 'leakage_check': ev.leakage_check,
               'cv_comparison': ev.comparison.round(4).to_dict('records'), 'test_period': ev.meta['test_period'],
               'data': {'n': a.n, 'seed': a.seed}, 'note': 'Synthetic data; not real-world accuracy.'}
    (a.out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(ev.comparison.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ('model', 'threshold', 'test_period')}, indent=2))
    print(f"Holdout ROC-AUC {ev.metrics['roc_auc']:.3f} · PR-AUC {ev.metrics['average_precision']:.3f} -> saved to {a.out}")
    return summary


if __name__ == '__main__':
    main()
