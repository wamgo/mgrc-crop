# Validation executed on 18 October 2026

## Passed

- `python -B -m unittest test_hcrc test_hcrc_multi test_training_resume`: 9 tests, OK (26.448 s).
- `python -B verify_jstars.py`: 104 settings, four classifiers, 50 repetitions per main configuration; main archived statistics and all 161 finite-population calculations match.
- Strict alpha=0.02 / B=25,50,100: all individual code-only runs have zero coverage; MGRC answered fractions 0.6235637, 0.6992736, 0.7629511. Pooled empirical compliance R<=alpha = 0.9576923 (15,600 runs).
- `python -B paper_v2_numbers.py --skip-map`: all output JSON fields match archived reference; geographic-map summary is explicitly copied from reference, not independently recomputed.
- Supplementary table generation succeeds from aggregate class counts; 16 table fragments match the manuscript source.
- `python -B summarize_register.py`: 20 rows x 10 columns match the archived reference to 1e-12 (maximum absolute difference 1.67e-16).
- `python -B reproduce_calibration.py --dry-run`: 20 commands planned, no training or calibration executed; refuses overwriting existing experiment output directories.
- All release Python files parse successfully.

## Environment and limits

Python 3.13. Quick dependencies use requirements.txt. The training-resume unit test additionally used torch 2.14.0, scikit-learn 1.6.1, h5py 3.12.1 and LightGBM 4.7.0. These are the packaging-test environment, not a claim that fresh training with these versions reproduces archived model weights bit for bit. The initial training-resume import lacked LightGBM; installing that optional dependency resolved it.

No full model training, raw-data download, or long calibration rerun was performed during packaging. Optional WorldCereal/multilevel scripts are provided as provenance but their entire downstream calculations were not rerun in this packaging check. Archived results are the evidence for paper statistics; numeric tests are not a proof of guarantees outside their tested assumptions.

## Additional packaging checks

- `worldcereal_audit.py` recomputed the supplementary S6 descriptive audit from locally held upstream-derived arrays: 413,011 targets, 12.40548% disagreement; shared errors 5.52818% for LightGBM and 3.51637% for L-TAE. Only aggregate counts are released; 368 product no-data values are counted as disagreements, matching the original convention.
- A fresh-archive test found the plotting script assumed its parent output directory existed. The release now creates parent directories explicitly; this is a packaging-path fix, not an algorithm change.

## Fresh archive verification

A new directory was populated only from the release ZIP. Checksum-verified unpacking, main-result verification, aggregate figure/statistic generation, supplementary taxonomy/path tables, register comparison checks and table generation all passed. Generated statistics and tables matched the manuscript. All 9 unit tests passed again in the clean extraction (27.019 s). Full logs are retained in the local submission audit folder.
