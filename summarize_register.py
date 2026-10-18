"""Regenerate the register-versus-phenology CSV; --check-only never writes a file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from verify_jstars import BASE, KEYS, ROOT, expected_pairs, per_setting, require, validate_runs

COLUMNS = ["ds", "alpha", "B", "reg_gain", "reg_wins", "n", "reg_mgrc", "reg_hsc", "pheno_gain", "pheno_mgrc"]
INDEX = ["ds", "alpha", "B"]


def summarize(root):
    rows = []
    for ds in ("breizh", "eurocrops"):
        summaries = {}
        pairs = expected_pairs(root, ds, "main")
        for prefix, subdir, name, alphas in [
            ("reg", "v2_register", f"register_pac__{ds}.parquet", [0.02, 0.1]),
            ("pheno", "v2", f"main_pac__{ds}.parquet", [0.02, 0.05, 0.1, 0.2]),
        ]:
            d = pd.read_parquet(root / "results" / "exp" / subdir / name)
            validate_runs(d, pairs, alphas, name)
            s = per_setting(d)
            s = s[s.alpha.isin([0.02, 0.1])]
            p = s.pivot(index=KEYS, columns="method", values="hcov")
            # Best code-only method is selected separately for every setting/configuration.
            gain = p.planned_info - p[BASE].max(axis=1)
            detail = pd.DataFrame({"gain": gain, "mgrc": p.planned_info, "hsc": p.hsc_climb}).reset_index()
            summaries[prefix] = {(a, B): x for (a, B), x in detail.groupby(["alpha", "B"])}
        require(set(summaries["reg"]) == set(summaries["pheno"]), f"{ds}: unpaired configurations")
        for (a, B), x in sorted(summaries["reg"].items()):
            y = summaries["pheno"][(a, B)]
            require(len(x) == len(y) == len(pairs), f"{ds}: missing settings at {(a, B)}")
            rows.append({"ds": ds, "alpha": a, "B": B, "reg_gain": float(x.gain.mean()),
                         "reg_wins": int((x.gain > 1e-9).sum()), "n": len(x),
                         "reg_mgrc": float(x.mgrc.mean()), "reg_hsc": float(x.hsc.mean()),
                         "pheno_gain": float(y.gain.mean()), "pheno_mgrc": float(y.mgrc.mean())})
    return pd.DataFrame(rows, columns=COLUMNS)


def compare_csv(actual, path, atol=1e-12):
    expected = pd.read_csv(path)
    require(list(expected.columns) == COLUMNS, "register reference: columns/order differ")
    require(not expected.duplicated(INDEX).any(), "register reference: duplicate keys")
    a, b = [x.set_index(INDEX).sort_index() for x in (actual, expected)]
    require(a.index.equals(b.index), "register reference: row keys differ")
    require(bool(np.isfinite(a.to_numpy()).all() and np.isfinite(b.to_numpy()).all()),
            "register comparison: non-finite values")
    delta = float(np.abs(a.to_numpy() - b.to_numpy()).max())
    require(bool(np.allclose(a.to_numpy(), b.to_numpy(), atol=atol, rtol=0)),
            f"register comparison failed: max absolute difference {delta:g}, tolerance {atol:g}")
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path, help="Archived original CSV; compared before any write")
    parser.add_argument("--output", type=Path, help="Default: ROOT/results/exp/v2_register/register_vs_pheno.csv")
    parser.add_argument("--check-only", action="store_true", help="Recompute/compare in memory without writing")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute numeric tolerance; relative tolerance is zero")
    args = parser.parse_args()
    require(np.isfinite(args.atol) and args.atol >= 0, "atol must be finite and nonnegative")
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "results/exp/v2_register/register_vs_pheno.csv"
    # Never overwrite a reference without first checking it. A fresh rerun may have no reference.
    candidates = [root / "reference/register_vs_pheno.csv", root / "results/exp/v2_register/register_vs_pheno.csv"]
    reference = args.reference.resolve() if args.reference else next((p for p in candidates if p.is_file()), None)
    require(not args.check_only or reference is not None, "--check-only needs an existing reference CSV")
    result = summarize(root)
    require(len(result) == 20, "expected 20 register/phenology comparison rows")
    if reference is not None:
        delta = compare_csv(result, reference, args.atol)
        print(f"PASS: 20 rows, {len(COLUMNS)} columns; max absolute difference={delta:.17g}; atol={args.atol:g}; reference={reference}")
    else:
        print("No reference present: generated from fresh runs; no archival-equality claim.")
    if args.check_only:
        print("Read-only check: no CSV written.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"WROTE {output}")


if __name__ == "__main__":
    main()
