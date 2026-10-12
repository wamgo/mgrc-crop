# MGRC: mixed-granularity risk control for hierarchical crop maps

Research code and per-run results accompanying **Coarse labels for fine guarantees: mixed-granularity risk control for hierarchical crop-type maps under regional and inter-annual shift**, by Ruiyuan Ding, [ORCID 0009-0003-2366-7203](https://orcid.org/0009-0003-2366-7203).

This is a research artifact, not a claim of journal acceptance. Version: **v1.0.0** (12 October 2026).

## Included material

- Calibration, source-only planning, feature extraction, training and experiment scripts.
- `results_bundle.zip`: nine per-run experiment tables in Parquet format, the 23-code/12-group taxonomy, ordered parcel identifiers for the eight sampled domains, model-training metadata and descriptive summaries.
- `MANIFEST.json`: SHA-256 checksums and sizes of all release files, plus the files inside the data bundle.
- Core calibration tests and an audit that recomputes the headline statistics from the released per-run results.

The bundle contains aggregate measurements per experimental run, not raw satellite time series or parcel geometries. The earlier nine-class pilot, abandoned results, personal files, manuscript drafts and browser automation are excluded. Existing probability archives contain third-party labels and coordinates and are not redistributed here; the scripts below regenerate them from the upstream data. No claim of a legal prohibition on redistribution is intended.

## Quick reproduction from saved results

Use Python 3.13 (3.12 or newer is required by the acquisition scripts). From this repository directory:

```text
python -m venv .venv
```

Activate `.venv` using your operating system's normal command, then:

```text
python -m pip install -r requirements.txt
python prepare_artifacts.py
python -m unittest test_hcrc
python verify_results.py
python paper_figures.py --skip-study-area
python graphical_abstract.py
```

`prepare_artifacts.py` checks the release checksums before unpacking; it refuses to overwrite different existing files. The verification script writes its findings to stdout. Figures and generated table fragments are written under `paper/figures/` and `paper/tables/`. The study-area map needs the upstream parcel geometry and extracted features, so the quick route skips it. The saved results suffice for the budget, noise, price-ratio, error-level and transfer comparisons; no training or raw-data download occurs in this route.

Key checks from the released files:

| Measurement | Value |
|---|---:|
| Source-calibrated violation frequency, 36 Level-2A settings | 0.835556 |
| Source-calibrated violation frequency, 18 Level-1C settings | 0.616667 |
| Information-weighted coverage gain over best fine-only baseline, B=25 | 0.09326 |
| Same, B=50 | 0.08742 |
| Same, B=100 | 0.03234 |

These are aggregates over the specified stored runs. Some individual target-calibrated runs exceed the nominal error level. The Level-1C aggregate contains **12 cross-year settings and 6 same-year cross-region controls**.

## Recompute from upstream data

This route downloads large archives and retrains models. It is not run by the quick commands above. Review storage and runtime requirements before starting; raw data require many gigabytes. Obtain data under their upstream terms, described in `DATA_SOURCES.md`.

```text
python -m pip install -r requirements-train.txt
python -m unittest test_training_resume
python acquire.py
python acquire_more.py
python features.py
python features.py 2017:L1C:frh01 2017:L1C:frh04
python features.py 2018
python train_models.py region
python train_models.py year
python experiments.py main pac
python experiments.py rho pac
python experiments.py alpha pac
python experiments.py spatial pac
python experiments.py year pac
python experiments.py main crc
python experiments.py year crc
python noise_exp.py
python noise_b50.py
python describe.py
python paper_figures.py
python graphical_abstract.py
```

Feature and probability scripts reuse existing outputs. Use a fresh checkout for an independent rerun instead of mixing regenerated and released results. Experiment scripts overwrite their corresponding Parquet outputs. A rerun can vary with numerical-library versions and hardware; the released per-run tables are the exact archived outputs used for the reported aggregates. No full retraining during release preparation is claimed.

## Sampling, seeds and interpretation

`data/splits/*_ids.npz` contains the ordered selected parcel IDs. For each source domain, positions 0–39,999 are training, 40,000–44,999 are development, and the remainder is source holdout. The 2017 L1C samples have 59,998 parcels in FRH01 and 59,994 in FRH04; the other six have 60,000. The old pilot's `splits.json` is not used.

Feature hashing and training use seed 20261011. Main experiment seeds are the setting indices in the deterministic enumeration in `experiments.py`; noise seeds are 200 + setting index at B=100 and 500 + setting index at B=50. Calibration/test indices for individual repeats are regenerated by the experiment scripts and their RNG call order, not stored separately. Labels cost B fine-label units; the implementation uses Python nearest-even `round(beta * B)` with a minimum of one fine label.

The clean expectation result assumes exchangeability; the high-probability result assumes i.i.d. sampling. The noisy-label theorem assumes identically distributed interpreted coarse labels and a known noise-rate bound. The released noise experiments instead pool clean and noisy coarse labels and assess the correction empirically. Annotation costs and noise are simulated, not measured from human interpretation. Spatial blocking is a sensitivity analysis, not proof of independence.

## Citation and rights

Use `CITATION.cff` and cite the original BreizhCrops publication and data providers. The original research code is offered under `LICENSE` (MIT); generated aggregate results under CC BY 4.0 as described in `DATA_SOURCES.md`. Third-party data and software retain their own terms. This repository does not relicense BreizhCrops, IGN/RPG or Sentinel data.

Repository: https://github.com/wamgo/mgrc-crop
