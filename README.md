# MGRC: JSTARS research artifact

Code and per-run aggregate results for **Certifying Transferred Crop-Type Maps With Few Crop-Code Labels: Group Labels for Strict Error Levels**, Ruiyuan Ding.

Prepared version: **v2.0.0**, 18 October 2026. This is a research artifact, not a claim of acceptance or publication. It supersedes the v1 artifact for the present manuscript; v1 remains associated with the earlier register-group experiments.

## What is different from v1?

- Phenology groups: 23 BreizhCrops codes / 4 groups and 24 EuroCropsML codes / 5 groups; no singleton group.
- Four classifiers (MLP, LightGBM, TempCNN, L-TAE), 104 transfer/classifier settings, including 16 cross-year and 8 same-year L1C controls.
- Strict error levels, noise and price sensitivity, spatial calibration, CRC, hindsight source margins, source-path pilot and register-group sensitivity.
- Exact finite-population certificate calculation at 161 configurations.

## Quick reproduction (no downloads or training)

Python 3.13 was used locally. Install dependencies in a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
python prepare_artifacts.py
python -B -m unittest test_hcrc test_hcrc_multi
python -B verify_jstars.py
python -B paper_v2_numbers.py --skip-map
python -B supp_tables_v2.py
python -B summarize_register.py
python -B supp_register_table.py
```

`results_bundle.zip` contains the reviewed numerical artifacts and ordered public-benchmark parcel IDs. `MANIFEST.json` supplies SHA-256 hashes. The unpacker refuses changed existing files. Outputs go to `paper_jstars/figures`, `paper_jstars/tables` and `results/exp/v2`. Run in a fresh extraction so generated output does not conflict with archive checksum checks.

`--skip-map` recomputes the statistical figures/tables, not the geographic map. It explicitly copies only the map summary from `reference/paper_numbers.json`; that field is not independently reproduced by this quick route. The two geographic figures are provided as archived display PDFs under `reference_figures/`. Raw map coordinates, parcel labels, probabilities, images, and trained weights are not redistributed. `map_example.py` regenerates the map input after upstream data/training are available. Supplementary taxonomy counts are released as class-level counts instead of probability archives.

See [REPRODUCE.md](REPRODUCE.md) for training and calibration and [DATA_SOURCES.md](DATA_SOURCES.md) for data boundaries. The full raw-data and model-training route has not been rerun during packaging; quick-result verification is recorded in `VALIDATION.md`.

## Interpretation

The guarantee controls unconditional hierarchical error across parcels, not the conditional error among only labelled parcels. The 96% figure is the empirical share meeting the error limit in the specified low-budget regime, not a theorem and not a per-class accuracy guarantee. Prices and group-label correctness are assumptions/sensitivity parameters, not measured annotation costs. The strict-alpha advantage does not establish a benefit at alpha >= 0.1.

## License and provenance

Original code: MIT (existing project license). Original aggregate results and documentation: CC BY 4.0 to the extent of the author's rights. Upstream data and identifiers remain subject to their source terms. This package grants no license over third-party material and includes no copied third-party loader or dependencies.

No manuscript PDF, cover letter, author photograph, private notes, browser automation, caches, credentials, or raw parcel geometry is part of this code artifact.
