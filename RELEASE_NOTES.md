# v2.0.0 - JSTARS artifact (2026-10-18)

Replaces the main artifact with the phenology-group experiments matching the JSTARS manuscript. The v1 tag remains the earlier register-group release. Adds EuroCropsML, L-TAE, strict-alpha and noise experiments, exact finite-population audit, explicit replication instructions, checksummed aggregate results and 18 ordered benchmark-ID arrays.

Packaging-only adjustments: portable repository-root helper, no parent-directory imports, removal of an unused third-party loader download, checksum validation of cached downloads, optional omission of the geographic map when using aggregate-only data, and aggregate class-count input for supplementary tables. Scientific calibration/training calculations are unchanged.

No raw parcel maps, labels/probabilities, weights, manuscript PDFs, or author correspondence are uploaded. Full training was not rerun during packaging. See VALIDATION.md for executed checks.

Also adds an aggregate-only WorldCereal S6 audit script/report and a standalone 600-dpi geographic plotting entry point. The plotting entry point now creates missing parent output directories in a clean extraction.
