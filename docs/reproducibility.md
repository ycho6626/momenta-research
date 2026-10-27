# Reproducibility and scope

This repository provides runnable numerical methods and verifiable aggregate evidence from a completed study. It deliberately separates two different reproducibility claims.

## What a public checkout can reproduce

`uv sync --locked` installs the Python 3.12 environment. `uv run pytest` executes hand-calculated accounting and regression fixtures, property tests for the mathematical identities, fit-timing and exclusion checks, and synthetic transport and state-recovery tests. All inputs to those tests are generated or constructed in the tests. No credentials or market-data subscription are required.

The initial public release passed **152 public tests in 141.60 seconds**, plus lint and the evidence verifier. The pinned HMM dependency emits 1,062 NumPy deprecation warnings during synthetic recovery; these are recorded separately from test failures. The full historical suite count below refers to a different scope. The [2026-10-28 revision](validation_2026-10-28.md) passes 155 tests, including three new synthetic-example checks.

`uv run momenta verify` checks each published evidence file against its SHA-256 digest, checks samples and row uniqueness, and verifies MSE-improvement, incremental-R² and Holm-adjustment arithmetic. It also checks agreement between the primary and headline tables. It does **not** independently recompute the stored forecast errors, Clark–West standard errors, bootstrap intervals or trading returns from raw observations. A digest establishes file integrity relative to the supplied manifest; it does not establish scientific correctness by itself.

The numerical source is an explicit selection from the completed research implementation. Each retained function's executable syntax tree was compared with the original, excluding docstrings and two documented diagnostic-message changes. Import lists and documentation were adapted to this package. [Code provenance](code_provenance.json) records file and function hashes. The settings file contains the numeric settings used by these methods; it is not an instruction to rerun the unavailable full data pipeline.

The public package is intended for source-checkout use. The tests and numerical examples are the executable interface; historical download, curation and experiment orchestration are not included. Some functions accept complete arrays or frames assembled by a caller, so their documented input assumptions matter.

## Runnable instructional example

`uv run --locked python examples/synthetic_walkforward.py` generates inputs,
constructs trailing features, runs the exported chronological engine and reports
forecast losses. [The example guide](../examples/README.md) gives the assumptions,
expected output and benchmark definitions. It writes only to standard output;
its synthetic results do not enter the historical evidence manifest.

To regenerate the one-page case study, optional document dependencies can be
provided without changing the project lock:

```sh
uv run --with reportlab --with pypdfium2 python tools/render_case_study.py . examples/output/case_study_preview.png
```

The renderer reads the Markdown, embeds clickable links, requires one page and
writes a preview into the ignored instructional-output directory. The dated
[research history](research_history.md) separates internal records from public
publication; it makes no independent public-preregistration claim.

## What was reproduced in the completed study

The original licensed-data pipeline rebuilt curation, portfolio books, features, annual model fits, strategies and bootstrap tables from pinned raw downloads. It matched **25 design, 25 holdout and 37 portfolio artifact hashes**, taking **3,636.307 seconds** on a two-core Intel laptop with 8 GB RAM. This timing excludes downloads and the separate robustness grid. Historical placebo, synthetic-diagnostic and earlier strategy-population records were fixed audit inputs, not rerun as part of that timing.

The original full suite passed **490 tests**. This is the historical full-study count, not the smaller public test-suite count. The sanitized [study receipt](../results/study_receipt.json) preserves timing, counts, vintages and aggregate hashes. It is a historical record; the public verifier cannot recreate the 87 restricted artifacts.

## Evidence included and excluded

The release includes aggregate model losses, hypothesis tests, strategy metrics, placebo summaries, sensitivity tables and research figures. Original numeric strings are preserved in the CSV exports. Figures retain their original image content. The [manifest](../results/manifest.json) identifies each released artifact.

Security-level prices, volume, actions, fundamentals, identifiers, holdings, feature panels, dated model predictions and downloaded reference series are not distributed. Recreating the full historical experiment requires separately authorized inputs and the original restricted pipeline; this public repository makes no turnkey raw-data replication promise.

The headline preserves the frozen design. Superseded design-period results are labeled explicitly. Post-unblinding sensitivities do not change the primary conclusion, selected models or hypotheses.
