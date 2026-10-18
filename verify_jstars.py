"""Read-only verification of the JSTARS v2 archived experiments (no training/import side effects)."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, hypergeom

ROOT = Path(__file__).resolve().parent
MODELS = ("mlp", "lgbm", "tempcnn", "ltae")
KEYS = ["tag", "model", "target", "alpha", "B", "rho"]
BASE = ["flat_selective", "hsc_climb", "levelwise_path_fine"]
METHODS = BASE + ["coarse_only", "mixed_cl_b0.25", "mixed_td_b0.25", "planned_info",
                  "planned_w0.3", "planned_w0.5", "planned_w0.7", "planned_w0.9",
                  "source_levelwise", "source_margin"]
BUDGETS = [25, 50, 100, 200, 400]
MAIN = [("main", "breizh", 48), ("year", "breizh", 24), ("main", "eurocrops", 32)]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def taxonomy_check(root):
    report = {}
    for ds, old_name, new_name, nf, ng in [
        ("breizh", "taxonomy.json", "taxonomy_pheno.json", 23, 4),
        ("eurocrops", "taxonomy_eurocrops.json", "taxonomy_eurocrops_pheno.json", 24, 5),
    ]:
        old, new = [read_json(root / "data" / n) for n in (old_name, new_name)]
        require(new["fine"] == old["fine"], f"{ds}: fine-class order changed")
        require(len(new["fine"]) == len(set(new["fine"])) == nf, f"{ds}: fine classes")
        parent = np.asarray(new["parent_of_fine"])
        require(parent.shape == (nf,) and parent.dtype.kind in "iu", f"{ds}: parent indices")
        require(len(new["parents"]) == ng and set(parent) == set(range(ng)), f"{ds}: groups")
        sizes = np.bincount(parent, minlength=ng)
        require(bool((sizes >= 2).all()), f"{ds}: singleton/empty phenology group")
        report[ds] = {"fine_classes": nf, "phenology_groups": ng,
                      "group_sizes": sizes.tolist(), "same_fine_order": True}
    return report


def expected_pairs(root, dataset, experiment):
    if dataset == "breizh":
        if experiment == "year":
            return {(f"L1C2017_{s}", m, t) for s in ("frh01", "frh04") for m in MODELS
                    for t in ("2018_L1C_frh01", "2018_L1C_frh04",
                              "2017_L1C_" + ("frh04" if s == "frh01" else "frh01"))}
        regions = ("frh01", "frh02", "frh03", "frh04")
        return {(f"L2A2017_{s}", m, f"2017_L2A_{t}")
                for s, t in product(regions, repeat=2) if s != t for m in MODELS}
    domains = read_json(root / "data" / "eurocrops_domains.json")
    return {(f"EC2021_{s}", m, f"2021_EC_{t}") for s in ("LV", "EE")
            for t in domains if len(t) > 2 and t.startswith("EE" if s == "LV" else "LV")
            for m in MODELS}


def validate_runs(d, pairs, alphas, label):
    """Validate the full PAC grid, not merely the total row count."""
    cols = KEYS + ["rep", "method", "risk", "fine", "parent", "hcov", "mode", "design"]
    require(set(cols) <= set(d.columns), f"{label}: missing columns")
    require(not d[cols].isna().any().any(), f"{label}: null required values")
    actual = set(d[["tag", "model", "target"]].itertuples(index=False, name=None))
    require(actual == pairs, f"{label}: transfer/model identities differ")
    for c, values in [("model", MODELS), ("alpha", alphas), ("B", BUDGETS),
                      ("rho", [10]), ("method", METHODS), ("rep", range(50)),
                      ("mode", ["pac"]), ("design", ["random"])]:
        require(set(d[c]) == set(values), f"{label}: unexpected {c}")
    identity = KEYS + ["method", "rep"]
    require(not d.duplicated(identity).any(), f"{label}: duplicate repetitions")
    counts = d.groupby(KEYS + ["method"], observed=True).size()
    require(len(counts) == len(pairs) * len(alphas) * len(BUDGETS) * len(METHODS),
            f"{label}: incomplete configuration grid")
    require(bool(counts.eq(50).all()), f"{label}: not 50 unique repetitions per cell")
    values = d[["risk", "fine", "parent", "hcov"]].to_numpy()
    require(bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()),
            f"{label}: non-finite/out-of-range metrics")
    require(bool((d.fine + d.parent <= 1 + 1e-12).all()), f"{label}: invalid coverage")


def per_setting(d):
    x = d.assign(over=(d.risk > d.alpha).astype(float), answered=d.fine + d.parent)
    x["sel_err"] = x.risk / x.answered.where(x.answered > 0)
    return x.groupby(KEYS + ["method"], observed=True)[
        ["risk", "over", "fine", "parent", "hcov", "answered", "sel_err"]].mean().reset_index()


def group_name(exp, ds, target):
    if ds == "eurocrops":
        return "EuroCropsML, cross-country"
    if exp == "year":
        return "BreizhCrops, cross-year" if target.startswith("2018") else "BreizhCrops, same-year (L1C)"
    return "BreizhCrops, cross-region"


def reference_path(root, name, fallback):
    preferred = root / "reference" / name
    return preferred if preferred.is_file() else root / fallback


def compare_tree(actual, reference, path="", atol=1e-12):
    if isinstance(actual, dict):
        require(isinstance(reference, dict) and set(actual) == set(reference), f"{path}: keys differ")
        for key in actual:
            compare_tree(actual[key], reference[key], f"{path}/{key}", atol)
    else:
        require(bool(np.isclose(actual, reference, rtol=0, atol=atol, equal_nan=False)),
                f"{path}: recomputed {actual!r} != reference {reference!r}")


def finite_population_check(root):
    """Independently repeat the exact seven (alpha, delta) pairs used in S2."""
    levels = [(0.02, 0.05), (0.02, 0.1), (0.05, 0.05), (0.1, 0.05),
              (0.1, 0.1), (0.018, 0.05), (0.09, 0.05)]
    expected = {(a, d, N, n) for a, d in levels for N in (2000, 10000, 40000)
                for n in (22, 50, 114, 165, 200, 400, 1000, 2000) if n < N}
    saved = read_json(reference_path(root, "finite_population_check.json", "results/finite_population_check.json"))
    indexed = {(v["alpha"], v["delta"], v["N"], v["n"]): v for v in saved}
    require(len(saved) == len(indexed) == len(expected) == 161 and set(indexed) == expected,
            "finite population: missing/duplicate/unexpected configurations")
    worst, max_diff = 0.0, 0.0
    for (a, d, N, n), row in indexed.items():
        k = np.arange(n + 1)
        u = np.where(k >= n, 1.0, beta.ppf(1 - d, k + 1, np.maximum(n - k, 1e-12)))
        K0 = int(np.floor(a * N)) + 1
        value = max(float(hypergeom.pmf(k, N, K, n)[u <= a].sum()) for K in (K0, K0 + 1))
        max_diff = max(max_diff, abs(value - row["false_certificate"]))
        require(np.isclose(value, row["false_certificate"], atol=1e-12, rtol=0),
                f"finite population: mismatch at {(a, d, N, n)}")
        require(value <= d and row["ok"] is True, "finite population: certificate exceeds delta")
        worst = max(worst, value / d)
    return {"configurations": 161, "max_ratio_to_delta": worst, "max_reference_difference": max_diff}


def verify(root):
    report = {"taxonomy": taxonomy_check(root)}
    frames, checks = [], []
    for exp, ds, n in MAIN:
        name = f"{exp}_pac__{ds}.parquet"
        d = pd.read_parquet(root / "results" / "exp" / "v2" / name)
        pairs = expected_pairs(root, ds, exp)
        require(len(pairs) == n, f"{name}: unexpected domain enumeration")
        validate_runs(d, pairs, [0.02, 0.05, 0.1] + ([] if exp == "year" else [0.2]), name)
        # Zero coverage must hold for every draw, not only after averaging.
        z = d[(d.alpha == 0.02) & (d.B <= 100) & d.method.isin(BASE)]
        require(bool((z[["fine", "parent", "hcov"]].to_numpy() == 0).all()),
                f"{name}: code-only baseline has nonzero low-budget coverage")
        s = per_setting(d)
        s["grp"] = [group_name(exp, ds, t) for t in s.target]
        frames.append(s)
        checks.append({"file": name, "settings": n, "models": 4, "repetitions": 50, "rows": len(d)})
    s = pd.concat(frames, ignore_index=True)
    n = len(s[["tag", "model", "target"]].drop_duplicates())
    require(n == 104, "expected 104 settings")
    report["main_grid"] = checks
    src = s[s.method == "source_levelwise"].copy()
    src["ratio"] = src.risk / src.alpha
    mg = s[s.method == "planned_info"]
    pivot = s.pivot(index=KEYS + ["grp"], columns="method", values="hcov")
    single = pivot.planned_info - pivot[BASE + ["coarse_only"]].max(axis=1)
    zero = s[s.method.isin(BASE)].groupby(["alpha", "B"]).hcov.max()
    numbers = {
        "settings": n,
        "source": {g: {"viol": round(float(x.over.mean()), 3),
                       "ratio_mean": round(float(x.ratio.mean()), 2),
                       "ratio_max": round(float(x.ratio.max()), 2)} for g, x in src.groupby("grp")},
        "source_all": {"viol": round(float(src.over.mean()), 3), "ratio_max": round(float(src.ratio.max()), 2)},
        "mgrc_viol": {str(a): round(float(x.over.mean()), 3) for a, x in mg.groupby("alpha")},
        "mgrc_viol_max_setting": round(float(mg.over.max()), 3),
        "mgrc_ratio_max": round(float((mg.risk / mg.alpha).max()), 3),
        "vs_best_single": {"max": round(float(single.max()), 3), "mean": round(float(single.mean()), 4),
                           "share_better": round(float((single > 1e-3).mean()), 3),
                           "share_worse": round(float((single < -1e-3).mean()), 3)},
        "sel_err_002": {str(B): round(float(x.sel_err.mean()), 4) for B, x in mg[mg.alpha == 0.02].groupby("B")},
        "answered_002": {str(B): round(float(x.answered.mean()), 3) for B, x in mg[mg.alpha == 0.02].groupby("B")},
        "codeonly_zero": {f"{a}|{B}": round(float(v), 3) for (a, B), v in zero.items() if a <= 0.05},
    }
    reference = read_json(reference_path(root, "paper_numbers.json", "results/exp/v2/paper_numbers.json"))
    for key, value in numbers.items():
        require(key in reference, f"missing reference statistic: {key}")
        compare_tree(value, reference[key], key)
    report["matched_paper_statistics"] = numbers
    strict = mg[(mg.alpha == 0.02) & (mg.B <= 100)]
    coverage = []
    for group, x in [("ALL 104 settings", strict)] + list(strict.groupby("grp")):
        for B, y in [("25,50,100 pooled", x)] + list(x.groupby("B")):
            coverage.append({"group": group, "B": str(B),
                             "setting_budget_cells": len(y), "runs": len(y) * 50,
                             "code_only_max_coverage": 0.0,
                             "mgrc_hcov": float(y.hcov.mean()), "mgrc_answered": float(y.answered.mean()),
                             "mgrc_compliance_R_le_alpha": float(1 - y.over.mean())})
    report["alpha_002_B_le_100"] = coverage
    report["aggregation_note"] = (
        "Average repetitions within each setting/configuration first, then average equally across those cells. "
        "Compliance is the empirical frequency R <= alpha, not a per-class accuracy guarantee. "
        "The low-budget compliance above uses ONLY alpha=0.02 and B in {25,50,100}; "
        "mgrc_viol in matched_paper_statistics includes ALL five budgets and must not be cited as this subset."
    )
    report["finite_population"] = finite_population_check(root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Absolute/relative data root; defaults to this script's directory")
    args = parser.parse_args()
    report = verify(args.root.resolve())
    print(json.dumps(report, indent=2, allow_nan=False))
    print("PASS: JSTARS main grid, reference statistics, strict-budget claim, taxonomies, and 161 finite-population cases")


if __name__ == "__main__":
    main()
