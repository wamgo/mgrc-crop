"""Plan (default) or explicitly execute JSTARS calibration in a fresh standalone copy.

No training, download, plotting or publication step is included. All script/data paths
are absolute and every child receives explicit MGRC_DATASET and MGRC_TAXON values.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = ("mlp", "lgbm", "tempcnn", "ltae")

# The child preloads the helper from this root, never from its parent directory.
# run_path preserves the target's __main__ guard and multiprocessing spawn entry point.
BOOTSTRAP = r"""
import importlib.util, json, os, runpy, sys
from pathlib import Path
root, dataset, taxon, filename, *arguments = sys.argv[1:]
root = Path(root).resolve()
def check(ok, message):
    if not ok:
        raise AssertionError(message)
check(Path.cwd().resolve() == root, 'working directory mismatch')
check(os.environ.get('MGRC_DATASET') == dataset, 'dataset environment mismatch')
check(os.environ.get('MGRC_TAXON') == taxon, 'taxonomy environment mismatch')
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('pilot_common', root / 'pilot_common.py')
helper = importlib.util.module_from_spec(spec)
sys.modules['pilot_common'] = helper
spec.loader.exec_module(helper)
check(Path(helper.__file__).resolve() == root / 'pilot_common.py', 'nonlocal helper')
check(Path(helper.ROOT).resolve() == root, 'helper root mismatch')
# Inspect setup with mkdir disabled first: an old helper must not create a nested root.
from unittest.mock import patch
with patch.object(Path, 'mkdir', lambda *a, **k: None):
    check(Path(helper.setup('direction_2_crop_hierarchy')).resolve() == root,
          'helper.setup must use the flat release root, not root/direction_2_crop_hierarchy')
import experiments as E
check(Path(E.__file__).resolve() == root / 'experiments.py', 'nonlocal experiments module')
check(Path(E.ROOT).resolve() == root, 'experiments root mismatch')
check(E.DATASET == dataset, 'experiments dataset mismatch')
name = ('taxonomy_eurocrops' if dataset == 'eurocrops' else 'taxonomy')
name += '_pheno.json' if taxon == 'pheno' else '.json'
expected = json.loads((root / 'data' / name).read_text(encoding='utf-8-sig'))
check(E.TAX == expected, 'loaded taxonomy differs from explicitly selected file')
check(E.PARENT.tolist() == expected['parent_of_fine'], 'parent mapping mismatch')
check(Path(E.PROBS).resolve() == root / 'results' / 'probs', 'probability root mismatch')
if taxon == 'pheno':
    counts = {i: expected['parent_of_fine'].count(i) for i in range(len(expected['parents']))}
    check(all(n >= 2 for n in counts.values()), 'phenology taxonomy contains singleton/empty group')
if filename == 'run_v2.py':
    import run_v2 as R
    check(arguments[0] == dataset, 'command dataset mismatch')
    expected_out = root / 'results' / 'exp' / ('v2' if taxon == 'pheno' else 'v2_register')
    check(Path(R.OUT).resolve() == expected_out, 'run_v2 output root mismatch')
    check((arguments[1] == 'register') == (taxon == 'register'), 'register/main taxonomy mix')
else:
    check(taxon == 'pheno', 'auxiliary scripts must never run with register taxonomy')
if filename == 'margin_hindsight.py':
    check(arguments == [dataset], 'margin dataset mismatch')
if filename in ('pilot_path.py', 'map_example.py'):
    check(dataset == 'breizh', 'Breizh-only auxiliary script with EuroCrops taxonomy')
sys.argv = [str(root / filename)] + arguments
runpy.run_path(str(root / filename), run_name='__main__')
"""


def plan(workers):
    jobs = []
    def add(ds, script, *args, taxon="pheno"):
        jobs.append((ds, taxon, script, list(args)))
    # Include Breizh main PAC, which the historical pipeline_v2.sh omitted.
    add("breizh", "run_v2.py", "breizh", "main", "pac", str(workers))
    add("breizh", "run_v2.py", "breizh", "year", "pac", str(workers))
    add("eurocrops", "run_v2.py", "eurocrops", "main", "pac", str(workers))
    for ds in ("breizh", "eurocrops"):
        for exp, mode in [("rho", "pac"), ("spatial", "pac"), ("main", "crc"), ("noise", "pac")]:
            add(ds, "run_v2.py", ds, exp, mode, str(workers))
        add(ds, "margin_hindsight.py", ds)
    for script in ("pilot_path.py", "map_example.py", "accuracy_all.py", "check_finite_population.py"):
        add("breizh", script)
    for ds in ("breizh", "eurocrops"):
        add(ds, "run_v2.py", ds, "register", "pac", str(workers), taxon="register")
    add("breizh", "summarize_register.py", "--root", "{ROOT}")
    return jobs


def prerequisites(root, jobs):
    scripts = {j[2] for j in jobs} | {"pilot_common.py", "experiments.py", "noise_exp.py", "hcrc.py", "verify_jstars.py"}
    required = [root / name for name in sorted(scripts)]
    required += [root / "data" / name for name in ["taxonomy.json", "taxonomy_pheno.json",
                 "taxonomy_eurocrops.json", "taxonomy_eurocrops_pheno.json", "eurocrops_domains.json"]]
    tags = ([f"L2A2017_frh0{i}" for i in range(1, 5)] +
            ["L1C2017_frh01", "L1C2017_frh04", "EC2021_LV", "EC2021_EE"])
    required += [root / "results" / "probs" / f"{tag}_{model}.npz" for tag in tags for model in MODELS]
    return [str(p) for p in required if not p.is_file()]


def existing_outputs(root):
    found = []
    for relative in ("results/exp/v2", "results/exp/v2_register"):
        path = root / relative
        if path.exists() and any(path.iterdir()):
            found.append(str(path))
    for relative in ("results/tables/accuracy_all_pheno.csv", "results/finite_population_check.json"):
        if (root / relative).exists():
            found.append(str(root / relative))
    return found


def check_probability_members(root):
    from verify_jstars import expected_pairs, taxonomy_check
    taxonomy_check(root)
    archives = {}
    for ds, exp in (("breizh", "main"), ("breizh", "year"), ("eurocrops", "main")):
        for tag, model, target in expected_pairs(root, ds, exp):
            archives.setdefault(f"{tag}_{model}.npz", {"source_hold"}).add(target)
    if len(archives) != 32:
        raise AssertionError("Expected exactly 32 probability archives")
    for name, domains in archives.items():
        with zipfile.ZipFile(root / "results" / "probs" / name) as archive:
            keys = set(archive.namelist())
        missing = {f"{domain}__{field}.npy" for domain in domains for field in ("p", "y", "xy", "ids")} - keys
        if missing:
            raise AssertionError(f"{name}: missing members {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Fresh standalone code/data copy; default is this script's directory")
    parser.add_argument("--workers", type=int, default=12, help="run_v2 worker count only; pilot_path currently uses its fixed 14 workers")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print plan without importing experiment modules or writing files (default)")
    mode.add_argument("--execute", action="store_true", help="Explicitly run the entire calibration plan; no model training")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.root.resolve()
    jobs = plan(args.workers)
    missing = prerequisites(root, jobs)
    occupied = existing_outputs(root)
    print(f"{'EXECUTE' if args.execute else 'DRY RUN (default)'}; absolute root: {root}", flush=True)
    print("No training. Every entry runs in a fresh process with the shown environment and root assertions.", flush=True)
    for i, (ds, taxon, script, arguments) in enumerate(jobs, 1):
        arguments = [str(root) if a == "{ROOT}" else a for a in arguments]
        command = [sys.executable, "-B", str(root / script), *arguments]
        print(f"{i:02d}. MGRC_DATASET={ds} MGRC_TAXON={taxon} " + subprocess.list2cmdline(command), flush=True)
    print("pilot_path.py retains 14 workers; map_example produces a locally generated parcel-level NPZ that is not redistributed.", flush=True)
    if missing:
        print("Missing prerequisites (32 probability archives must be regenerated or supplied from matching local runs):", flush=True)
        for path in missing:
            print("  " + path, flush=True)
    if occupied:
        print("Existing outputs: execution is refused here; use a NEW independent directory:", flush=True)
        for path in occupied:
            print("  " + path, flush=True)
    if not args.execute:
        print(f"DRY RUN complete: {len(jobs)} steps; nothing executed or written.")
        return
    if occupied or missing:
        parser.error("Execution requires complete inputs and a fresh output tree; nothing was executed")
    # No checkpoint/result mixing: there is intentionally no overwrite or resume flag.
    sys.path.insert(0, str(root))
    check_probability_members(root)
    for i, (ds, taxon, script, arguments) in enumerate(jobs, 1):
        arguments = [str(root) if a == "{ROOT}" else a for a in arguments]
        env = dict(os.environ, MGRC_DATASET=ds, MGRC_TAXON=taxon,
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root))
        print(f"RUN {i}/{len(jobs)}: {script} {' '.join(arguments)}", flush=True)
        subprocess.run([sys.executable, "-B", "-c", BOOTSTRAP, str(root), ds, taxon, script, *arguments],
                       cwd=root, env=env, check=True)
    print("Calibration complete. Outputs are locally generated; publication preparation is a separate step.")


if __name__ == "__main__":
    main()
