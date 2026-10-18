# Validation executed on 18 October 2026

## Passed

- `python -B -m unittest test_hcrc test_hcrc_multi test_training_resume` passed all 9 tests in 26.448 s.
- `python -B verify_jstars.py` verified 104 settings, four classifiers and 50 repetitions per main configuration. The main archived statistics and all 161 finite-population calculations matched.
- At alpha=0.02 and B=25,50,100, all individual code-only runs had zero coverage. MGRC answered fractions were 0.6235637, 0.6992736 and 0.7629511. Pooled empirical compliance R<=alpha was 0.9576923 over 15,600 runs.
- `python -B paper_v2_numbers.py --skip-map` produced JSON fields matching the archived reference. The geographic-map summary was copied from the reference rather than independently recomputed.
- Supplementary table generation succeeds from aggregate class counts; 16 table fragments match the manuscript source.
- `python -B summarize_register.py` produced 20 rows and 10 columns matching the archived reference to 1e-12, with a maximum absolute difference of 1.67e-16.
- `python -B reproduce_calibration.py --dry-run` listed 20 commands without executing training or calibration. The runner rejects existing experiment output directories.
- All release Python files parse successfully.

## Environment and limits

The verification environment used Python 3.13 and the quick-route dependencies in requirements.txt. The training-resume unit test additionally used torch 2.14.0, scikit-learn 1.6.1, h5py 3.12.1 and LightGBM 4.7.0. These are the packaging-test environment, not a claim that fresh training with these versions reproduces archived model weights bit for bit. The initial training-resume import lacked LightGBM; installing that optional dependency resolved it.

No full model training, raw-data download, or long calibration rerun was performed during packaging. Optional WorldCereal/multilevel scripts are provided as provenance but their entire downstream calculations were not rerun in this packaging check. Archived results are the evidence for paper statistics; numeric tests are not a proof of guarantees outside their tested assumptions.

## Additional packaging checks

- `worldcereal_audit.py` recomputed the supplementary S6 descriptive audit from locally held upstream-derived arrays: 413,011 targets, 12.40548% disagreement; shared errors 5.52818% for LightGBM and 3.51637% for L-TAE. Only aggregate counts are released; 368 product no-data values are counted as disagreements, matching the original convention.
- A fresh-archive test found the plotting script assumed its parent output directory existed. The release now creates parent directories explicitly; this is a packaging-path fix, not an algorithm change.

## Fresh archive verification

A new directory was populated only from the release ZIP. Checksum-verified unpacking, main-result verification, aggregate figure/statistic generation, supplementary taxonomy/path tables, register comparison checks and table generation all passed. Generated statistics and tables matched the manuscript. All 9 unit tests passed again in the clean extraction (27.019 s). Full logs are retained in the local submission audit folder.
