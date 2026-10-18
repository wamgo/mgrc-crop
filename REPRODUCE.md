# Reproducing the JSTARS v2 results

This release uses `results/exp/v2` and the phenology hierarchies: 23 crop codes in
4 groups for BreizhCrops, and 24 codes in 5 groups for EuroCropsML. No phenology
group contains a single code. Register-group results belong in
`results/exp/v2_register`, not in the main results.

Choose one route:

- **Quick reproduction:** validate selected archived results and regenerate
  aggregate tables and plots. No training, probability archives or parcel coordinates
  are needed. Figure 1 and the released-map illustration are reused as PDFs.
- **Calibration reproduction:** regenerate calibration experiments from 32 locally
  generated probability archives that are not redistributed. This can take substantial time,
  but does not train a classifier.
- **Full reproduction:** acquire the upstream data, extract features, train all
  four classifiers, then use the calibration route.

Use Python 3.13 and an isolated environment. Install `requirements.txt` for the
quick route and `requirements-train.txt` for acquisition, features and training.
The latter includes PyTorch, LightGBM, scikit-learn, h5py, pyshp and pyproj.
The calibration route needs the quick-route packages, not PyTorch.

## 1. Quick route from the release archive

Run the following from the **new JSTARS release root**, not the old
`publication/mgrc-crop` package. If you received only `results_bundle.zip`, first
run `python -B prepare_artifacts.py` after the final `MANIFEST.json` is present.
That command verifies and extracts the bundle; it is not read-only. Do not use an
old release's manifest or extraction script with this bundle.

These checks are read-only:

```text
python -B -m unittest test_hcrc test_hcrc_multi
python -B verify_jstars.py
python -B summarize_register.py --check-only
```

Both verification commands also accept `--root /absolute/path/to/release`.
They resolve data relative to that root and do not import `experiments.py` or a
parent-directory helper. The verifier prefers immutable `reference/` copies of
`paper_numbers.json` and `finite_population_check.json`; it falls back to the
corresponding files under `results/` when the reference copies are absent.

The verification covers the three main PAC tables: 48 cross-region settings,
24 L1C settings, and 32 cross-country settings. The 24 L1C settings comprise
16 cross-year and 8 same-year cross-region settings. It checks the actual
transfer/model identities, all four models, the full alpha/budget/method grid,
and 50 distinct repetitions per cell. It checks taxonomy order and group sizes,
compares selected recomputed paper statistics with the archive, and independently
repeats all 161 finite-population configurations. Specifically, `verify_jstars.py`
does not read or validate the rho, spatial, CRC, noise, hindsight-margin or
source-path-pilot Parquet tables, the accuracy CSV, or parcel-level map inputs.
`summarize_register.py --check-only` separately checks the two register PAC
tables against the corresponding main phenology tables and compares their
aggregate CSV with the reference. Running the plotting commands consumes
additional auxiliary results but is not a complete independent validation of
those experiments. Neither check certifies a fresh training run.

### Interpreting the low-budget claim

The report's `alpha_002_B_le_100` section uses **only alpha = 0.02 and budgets
25, 50 and 100**. It reports information-weighted coverage (`hcov`), answered
fraction (`fine + parent`), and the frequency of `R <= alpha` separately.
The code-only methods are `flat_selective`, `hsc_climb`, and
`levelwise_path_fine`; this claim does not include the source-only baselines.
The verifier checks zero coverage for each individual code-only run in that
subset, not just a rounded mean.

For the archived results, the pooled MGRC compliance is 0.9576923 over 15,600
runs, with mean `hcov` 0.3020797 and answered fraction 0.6952628. The per-budget
compliance is 0.9519231, 0.9628846 and 0.9582692. Average repetitions within each
setting/configuration first, then average those cells equally. These empirical
frequencies are not per-class accuracy guarantees. Do not attach the rounded
"96%" statement to an all-budget summary: the archive's alpha = 0.02
`mgrc_viol` statistic includes budgets 200 and 400 as well.

### Regenerate aggregate artifacts without parcel-level inputs

The release plotting route uses the existing `--skip-map` option and archived
class counts. These commands write generated artifacts, unlike the checks above:

```text
python -B paper_v2_numbers.py --skip-map
python -B supp_tables_v2.py
python -B summarize_register.py
python -B supp_register_table.py
```

`paper_v2_numbers.py --skip-map` computes the non-map statistics and Figure 2
from the released results. It copies map statistics from
`reference/paper_numbers.json`; it does not independently recompute the map.
`supp_tables_v2.py` now reads the released aggregate counts in
`data/supplementary_class_counts.json` (the `breizh` and `eurocrops` arrays).
When that file exists, the taxonomy tables require no probability NPZs;
the script falls back to counting labels in the three probability archives only
when the aggregate-count file is absent. The counts follow the corresponding
taxonomy `fine` order. Preserve the reference PDFs
`reference_figures/fig_study_area.pdf` and `reference_figures/fig_map.pdf` in
`paper_jstars/figures/` when preparing the manuscript. For example, in PowerShell:

```powershell
$Root = (Get-Location).Path  # Run this block in the release root.
$Figures = Join-Path $Root 'paper_jstars/figures'
New-Item -ItemType Directory -Force -Path $Figures | Out-Null
Copy-Item -LiteralPath (Join-Path $Root 'reference_figures/fig_study_area.pdf') -Destination $Figures
Copy-Item -LiteralPath (Join-Path $Root 'reference_figures/fig_map.pdf') -Destination $Figures
```

**Figure 1 is reused, not regenerated by the quick route.** The map PDF is also
reused. The public release does not include `map_example.npz`, which contains
parcel-level coordinates and labels. Do not run `map_example.py` or
`paper_v2_numbers.py` without `--skip-map` expecting the public bundle to contain
those inputs. If manuscript sources are included, compile from `paper_jstars`
with `latexmk -pdf main.tex` and `latexmk -pdf supplementary.tex`.

## 2. Calibration only, in a new independent directory

Keep the release archive intact. Create a separate working directory containing
copies of the release Python files and its **local portable `pilot_common.py`**.
Copy the four taxonomy JSON files and `data/eurocrops_domains.json`. Supply only
`results/probs/` under `results/` at this stage. Do not copy archived `v2`,
`v2_register`, their `ck` directories, `accuracy_all_pheno.csv`, or
`finite_population_check.json` into the new working directory.

The 32 required archives are `results/probs/<tag>_<model>.npz`:

```text
tag:   L2A2017_frh01, L2A2017_frh02, L2A2017_frh03, L2A2017_frh04,
       L1C2017_frh01, L1C2017_frh04, EC2021_LV, EC2021_EE
model: mlp, lgbm, tempcnn, ltae
```

They are **not distributed in the public aggregate-results bundle**. Regenerate
them using section 3 or supply matching locally generated archives. Each contains
`source_hold__p/__y/__xy/__ids` and the same members for its target domains.
The runner checks the required members before executing. Probability class
columns must follow the supplied taxonomy's `fine` order. Model weight files
are not needed once these probability archives exist.

From the independent directory:

```text
python -B reproduce_calibration.py
python -B reproduce_calibration.py --dry-run --workers 12
python -B reproduce_calibration.py --execute --workers 12
```

The first two commands only print the plan and missing-input/existing-output
warnings. They do not import experiment modules, create directories or run any
experiment. `--execute` is the only execution switch. You may instead call the
script by absolute path and pass `--root /absolute/path/to/independent-copy`.
That target must contain its own code and inputs.

The runner refuses an existing output tree rather than silently accepting old
results or checkpoints. There is no automatic resume/overwrite option. If a run
fails, keep its directory for diagnosis and start another fresh copy when using
this whole-plan runner. Low-level `run_v2.py` supports checkpoints, but resuming
it requires checking the code, taxonomy, domain enumeration and configuration
against the original run; filename existence alone does not establish a match.

The 20-step plan performs:

1. Main PAC experiments for both datasets and the Breizh L1C year experiment.
2. Rho, spatial, main CRC and noisy-parent experiments for both datasets.
3. Both hindsight-margin calculations; the Breizh source-path pilot and locally generated
   map example; accuracy aggregation; the finite-population calculation.
4. Register-group PAC experiments for both datasets, followed by the register
   versus phenology CSV aggregation.

Each child process receives explicit `MGRC_DATASET=breizh|eurocrops` and
`MGRC_TAXON=pheno|register`. Before running a script, the launcher loads the
helper by its absolute local path and checks the helper root, experiment root,
loaded taxonomy, parent mapping, probability path and applicable output path.
Auxiliary scripts cannot run under the register hierarchy, and the map/pilot
steps cannot inherit the EuroCrops hierarchy. The runner does not train, download
or plot. `--workers` controls `run_v2.py`; `pilot_path.py` retains its own fixed
14-worker pool.

The map step creates `results/exp/v2/map_example.npz` locally. Do not add
this file to a public bundle. It is not needed for aggregate-result verification.
The runner also creates the accuracy CSV and finite-population JSON. Plotting
and manuscript preparation remain separate steps.

### Register comparison and references

`summarize_register.py` first averages repetitions within each setting, computes
the best of the three code-only baselines **per setting**, and then averages the
paired gains. The output columns, in order, are:

```text
ds,alpha,B,reg_gain,reg_wins,n,reg_mgrc,reg_hsc,pheno_gain,pheno_mgrc
```

It writes `results/exp/v2_register/register_vs_pheno.csv`. If
`reference/register_vs_pheno.csv` exists, it compares against that file before
writing; otherwise it compares with an existing output CSV if present. Use
`--reference /absolute/path/to/original.csv` to choose a reference explicitly.
The default absolute tolerance is `1e-12`, with zero relative tolerance.
`--check-only` requires a reference and never writes. A fresh run with neither
reference nor output generates the CSV but makes no archival-equality claim.

If you intend to reproduce the archived probabilities exactly, keep reference
files in `reference/` for comparison. Fresh training may differ with hardware or
library versions. In that case, a strict reference comparison can fail without
implying that the aggregation code is broken. Do not replace the archived
reference with new numbers to obtain a pass. Preserve both runs and report the
difference. `verify_jstars.py` is an archived-result verifier, not a claim of
bitwise reproducible training.

## 3. Full acquisition, features and training

Use another fresh independent code/data directory, or the empty directory that
will become the calibration working copy. Install `requirements-train.txt`.
Keep all four released taxonomy files: `taxonomy.json`, `taxonomy_pheno.json`,
`taxonomy_eurocrops.json`, and `taxonomy_eurocrops_pheno.json`. Feature extraction
uses the fine-code order in the register taxonomy; calibration uses the
phenology parent mapping. Do not substitute one parent mapping for the other.

Acquire the upstream data:

```text
python -B acquire.py
python -B acquire_more.py
python -B acquire_eurocropsml.py
```

The Breizh inputs are four 2017 L2A CSV/HDF5 domains, FRH01/FRH04 2017 L1C
CSV/HDF5 domains, FRH01/FRH04 2018 L1C CSV/ZIP domains, and the corresponding
2017/2018 parcel shapefiles with their sidecar files. The EuroCropsML feature
extractor reads `data/eurocropsml/preprocess.zip`; the acquisition script also
downloads `split.zip`. If you rebuild `taxonomy_eurocrops.json` rather than using
the released version, you additionally need `sources/eurocropsml/HCAT3.csv`.
The acquisition script does not fetch that table. Preserve the released
phenology mappings; feature extraction does not generate them.

Extract features in this order:

```text
python -B features.py
python -B features.py 2017:L1C:frh01 2017:L1C:frh04
python -B features.py 2018
python -B seq_features.py
python -B features_ec.py
```

The first four commands produce the eight Breizh feature/sequence pairs.
`features_ec.py` produces ten EuroCropsML pairs and `eurocrops_domains.json`:
LV and EE source domains plus EE004, EE008, EE009, LV003, LV005, LV007, LV008
and LV009 target domains. L-TAE sequence rows follow the existing feature IDs;
do not extract sequences before the corresponding features.

Train all models:

```text
python -B train_models.py region
python -B train_models.py year
python -B train_models.py eurocrops
python -B train_ltae.py region
python -B train_ltae.py year
python -B train_ltae.py eurocrops
```

`train_models.py` trains MLP, LightGBM and TempCNN. `train_ltae.py` uses 60 epochs
by default and accepts `--epochs N`; changing it changes the experiment.
The source split uses the first 40,000 ordered parcels for training, the next
5,000 for development/temperature fitting, and the rest for source holdout.
Feature ordering and training use seed 20261011. Existing probability files
cause the trainers to skip work, so do not train over a directory containing
stale predictions. Preserve feature IDs, taxonomy order and model metadata.

After the 32 probability archives exist, run section 2. The v2 experiment seeds
start at 50000 for Breizh and 60000 for EuroCropsML; `run_v2.py` adds the setting
index, with the additional noise offset defined there. The source-path pilot
and map example have their own fixed seeds. Use the supplied enumeration and
scripts rather than reconstructing an older pipeline.

### Optional geographic regeneration at 600 dpi

The release now includes a standalone `plot_geographic_figures.py` entry point;
you do not need the old TGRS plotting chain. After regenerating the required inputs
locally, use absolute data and output directories:

```text
python -B plot_geographic_figures.py --data-root /absolute/path/to/local-run --output /absolute/path/to/figures --only study-area
python -B plot_geographic_figures.py --data-root /absolute/path/to/local-run --output /absolute/path/to/figures --only map
```

Use `--only all` to generate both `fig_study_area.pdf` and `fig_map.pdf` in one
call. The PDF export uses 600 dpi for rasterized content. The `study-area` branch
needs the four `data/feat_2017_L2A_frh*.npz` feature files, the eight EC target
feature files, and `data/eurocrops_domains.json`. The `map` branch needs
`results/exp/v2/map_example.npz` and `map_example_ua.csv` from the local
calibration run. No training occurs in this plotting command. The map NPZ is not redistributed in this release. Quick reproduction still reuses the supplied
Figure 1/map PDFs and does not require these locally generated inputs.

## 4. Do not mix this release with v1

The old `publication/mgrc-crop` package fixed the hierarchy to register groups
and verified 36/18 settings. Its `paper_figures.py` consumes old result paths,
not `results/exp/v2`. Setting `MGRC_TAXON=pheno` does not retrofit that old code.
The historical `pipeline_v2.sh` also omitted the Breizh main PAC stage and did
not include the auxiliary results. Use the new explicit plan instead.

Register taxonomies have 12 RPG and 13 HCAT groups, including single-code groups.
Treating these as cheap parent labels changes both information value and the
coverage measure. They are a supplementary comparison, not the main JSTARS
phenology experiment.

## WorldCereal supplementary S6 audit

`results/worldcereal_audit.json` contains aggregate counts only. To recompute from locally generated probability archives and the sampled WorldCereal arrays:

```text
python worldcereal_audit.py --output results/worldcereal_audit.json
```

Run acquire_worldcereal.py and wc_sample.py first if the sampled products are absent. These need upstream data and rasterio. The audit checks parcel-ID alignment, excludes source holdouts, and counts 368 no-data product values as disagreements, matching the archived manuscript convention. Packaging independently recomputed 413,011 target parcels, disagreement 0.1240548, and shared-error rates 0.0552818 (LightGBM) and 0.0351637 (L-TAE). This does not rerun the optional noisy-product calibration experiments.
