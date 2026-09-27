# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions correspond to the
Git tags and [GitHub releases](https://github.com/MD-ALL-SHAHRIA/pulseair/releases).
Dates are the tag/commit dates from the repository history; nothing here is reconstructed
beyond it.

## [Unreleased]

Changes on `main` since v1.0.3, not yet tagged.

### Added
- National Air Quality Management Plan 2024–2030 as a primary source
  (`docs/reference/NAQMP_2024-2030.pdf`), with roadmap framing in the thesis
  Introduction, §4.12 and Discussion.
- A survey of the Department of Environment's publicly published AQI reports
  (`docs/reference/doe_published_reports_survey.md` and the archive indices), reported
  for scope only — nothing integrated into the pipeline.
- A fifth data-integrity signature: the discarded pre-2022 Dhaka series compared
  month-by-month against the DoE's published monthly CAMS averages
  (`reports/dhaka_monthly_fabrication_crosscheck.md`, `src/preprocessing/dhaka_monthly_crosscheck.py`).
- The six extended-validation phases and the fifth fabrication signature integrated into
  the thesis (new §4.15 and figures), with the 8-fold rolling-origin result now the
  headline and the 5-fold kept as the underpowered-design note.
- An explicit fold-independence caveat around the rolling-origin Wilcoxon p-values
  (§3.10, §4.9, §4.11, §6.4).
- A claim ledger (`docs/claim_ledger.md`), a "scope boundary" thesis subsection (§8.2),
  and a precise, evidence-grounded novelty statement (§1.3).
- A single end-to-end reproduction entrypoint (`src/reproduce_all.py`) and a `Makefile`.
- A Zenodo DOI badge and a citable Citation section in the README.
- A draft DoE/CASE data-request email (`docs/outreach/doe_data_request_email.md`).

### Changed
- Corrected one over-strong phrase flagged by the claim ledger: "closes the gap" →
  "narrows the gap" in the Abstract and §1.3 (wording only; no numbers changed).

## [1.0.3] — 2026-09-26

First release whose citation metadata validates. See the
[release notes](https://github.com/MD-ALL-SHAHRIA/pulseair/releases/tag/v1.0.3).

### Fixed
- Both `CITATION.cff` files (root and `pulsebench/`) now pass CFF 1.2.0 schema
  validation; the `references` entries had been missing the required `authors` field.
  v1.0.0–v1.0.2 ship citation metadata that does not validate.

## [1.0.2] — 2026-09-26

Packaging metadata only; see the
[release notes](https://github.com/MD-ALL-SHAHRIA/pulseair/releases/tag/v1.0.2).

### Changed
- Aligned the version declaration across `pyproject.toml`, `pulsebench/__init__.py` and
  both `CITATION.cff` files so `pip` metadata and `import pulsebench` agree.

## [1.0.1] — 2026-09-26

Extended validation; see the
[release notes](https://github.com/MD-ALL-SHAHRIA/pulseair/releases/tag/v1.0.1).

### Added
- Six extended-validation phases: Holm-Bonferroni option in `pulsebench`; 8-fold
  rolling-origin CV; leave-one-station-out CV; sensor-noise robustness; selective
  prediction; and `dataset_audit`, `pulsebench`'s fifth exported function.
- The complete thesis generated as a Word document from committed files.
- A seasonal-naive evaluation baseline (contributed via #8).
- The Phase 1 expanded reference list (61 entries, 22 newly verified).
- Publication figures generated from committed metrics.

### Fixed
- pandas-3 date handling in `dataset_audit`, plus reproducibility gaps closed by a report
  index-completeness check.
- Two `.gitignore` rules that had hidden tracked work.

## [1.0.0] — 2026-09-22

Initial public release: the thesis code, the `pulsebench` toolkit, and the reports and
metrics behind every reported number. See the
[release notes](https://github.com/MD-ALL-SHAHRIA/pulseair/releases/tag/v1.0.0).

[Unreleased]: https://github.com/MD-ALL-SHAHRIA/pulseair/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/MD-ALL-SHAHRIA/pulseair/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/MD-ALL-SHAHRIA/pulseair/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/MD-ALL-SHAHRIA/pulseair/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/MD-ALL-SHAHRIA/pulseair/releases/tag/v1.0.0
