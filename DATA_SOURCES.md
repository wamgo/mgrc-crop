# Data provenance and scope

Upstream benchmark: [BreizhCrops](https://github.com/dl4sits/BreizhCrops).

Publication: Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., Körner, M. (2020). BreizhCrops: A Time Series Dataset for Crop Type Mapping. DOI: https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020.

The upstream repository's `readme.md` identifies French IGN Registre Parcellaire Graphique (RPG) as the source of raw crop labels and links to [the government data catalogue](https://www.data.gouv.fr/fr/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/). Acquisition scripts use `https://breizhcrops.s3.eu-central-1.amazonaws.com/`, the benchmark's distribution endpoint. Check current upstream terms when downloading. The repository software LICENSE is GPL-3.0; that file alone is not evidence of a license for every upstream dataset. The copied upstream loader is not part of this release.

## Released bundle

- `results/exp/*.parquet`: numerical measurements per calibration/test repetition, produced by this study; no per-parcel features, labels or geometry.
- `data/taxonomy.json`: the chosen crop-code to RPG-group mapping and grouping-purity checks. This is a study selection from the administrative hierarchy, not a newly invented taxonomy.
- `data/splits/*_ids.npz`: one `ids` array per sampled region/year/product in feature-file order. These are parcel identifiers used to specify the sample, not names, addresses or ownership records.
- `results/probs/*_meta.json`: recorded training summaries, temperatures, accuracies and runtimes. Model probability/label/coordinate arrays are excluded.
- `results/describe.json` and `results/tables/{data_counts,taxonomy,model_accuracy}.csv`: descriptive summaries used to explain the samples and models.

Original aggregate results and release documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Ruiyuan Ding, 2026, to the extent of the author's rights. This grant does not extend to third-party rights in administrative codes or source data. The release's MIT license applies to the original Python code only. The repository contains no vendored third-party source code or dependencies.

## Reproduction limits

The archived result tables reproduce reported statistics without retrieving the raw benchmark. Re-running calibration from probabilities requires regenerating the probability archives from the upstream data; raw imagery, raw label tables, parcel geometries, extracted feature arrays and trained weights are not bundled. They are not described as confidential or prohibited to share; this release avoids duplicating upstream datasets whose licensing must be checked separately.

The acquisition scripts retain original download URLs and SHA-256 receipts locally. `MANIFEST.json` covers the released artifacts. A versioned GitHub snapshot identifies this release; no Zenodo DOI has been minted and no such DOI is claimed.
