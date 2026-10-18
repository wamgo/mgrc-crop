# MGRC: JSTARS research artifact

This repository contains code and per-run aggregate results for **Certifying Transferred Crop-Type Maps With Few Crop-Code Labels: Group Labels for Strict Error Levels**, by Ruiyuan Ding.

The research artifact was released as **v2.0.0** on 18 October 2026. It accompanies the manuscript prepared for JSTARS. The v1 artifact covers the earlier register-group experiments. The main-branch documentation was revised on 19 October 2026; the versioned v2.0.0 download retains its original files and checksums.

## Changes from v1

- The phenology hierarchies contain 23 BreizhCrops codes in 4 groups and 24 EuroCropsML codes in 5 groups. Each group contains at least two codes.
- The experiments cover four classifiers (MLP, LightGBM, TempCNN and L-TAE) and 104 transfer/classifier settings, including 16 cross-year and 8 same-year L1C controls.
- The evaluation covers strict error levels, noise and price sensitivity, spatial calibration, CRC, hindsight source margins, a source-path pilot and register-group sensitivity.
- The finite-population audit computes the false-certificate probability for 161 configurations.

## Quick reproduction (no downloads or training)

The packaging checks used Python 3.13. The following sequence installs dependencies and reproduces the archived statistics and tables in a virtual environment:

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

`results_bundle.zip` contains the reviewed numerical artifacts and ordered public-benchmark parcel IDs. `MANIFEST.json` supplies SHA-256 hashes. The unpacker refuses changed existing files. Outputs go to `paper_jstars/figures`, `paper_jstars/tables` and `results/exp/v2`. A fresh extraction separates generated outputs from the archived files used in checksum checks.

`--skip-map` recomputes the statistical figures/tables, not the geographic map. It explicitly copies only the map summary from `reference/paper_numbers.json`; that field is not independently reproduced by this quick route. The two geographic figures are provided as archived display PDFs under `reference_figures/`. Raw map coordinates, parcel labels, probabilities, images, and trained weights are not redistributed. `map_example.py` regenerates the map input after upstream data/training are available. Supplementary taxonomy counts are released as class-level counts instead of probability archives.

[REPRODUCE.md](REPRODUCE.md) describes training and calibration. [DATA_SOURCES.md](DATA_SOURCES.md) describes the included and excluded data. The full raw-data and model-training route has not been rerun during packaging; quick-result verification is recorded in `VALIDATION.md`.

## Interpretation

The guarantee controls unconditional hierarchical error across parcels, not the conditional error among only labelled parcels. The 96% figure is the empirical share meeting the error limit in the specified low-budget regime, not a theorem and not a per-class accuracy guarantee. Prices and group-label correctness are assumptions/sensitivity parameters, not measured annotation costs. The strict-alpha advantage does not establish a benefit at alpha >= 0.1.

## License and provenance

The original code is licensed under MIT. Original aggregate results and documentation are licensed under CC BY 4.0 to the extent of the author's rights. Upstream data and identifiers remain subject to their source terms. This package grants no license over third-party material and includes no copied third-party loader or dependencies.

No manuscript PDF, cover letter, author photograph, private notes, browser automation, caches, credentials, or raw parcel geometry is part of this code artifact.
