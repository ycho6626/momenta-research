# Public revision validation — 2026-10-28

This revision changes presentation and adds an instructional example. It does not
change the research methods, settings, historical evidence or private repository.

| Check | Result |
|---|---|
| `uv sync --locked` | Passed; dependency lock unchanged |
| `uv run momenta verify` | Passed: 22 artifacts and 16 aggregate tables |
| `uv run pytest` | 155 passed in 25.65 s; command wall time 27.85 s |
| `uv run ruff check` | Passed |
| `uv run --locked python examples/synthetic_walkforward.py` | Passed in 4.79 s; output matches the documented six-decimal values |
| Documentation links | All local Markdown targets resolve |
| Case-study PDF | One page; rendered and visually inspected; repository URI embedded as a clickable link |
| Research preservation | All 43 protected public files match the preceding release byte-for-byte |

The protected files comprise the numerical source, settings, dependency metadata,
result tables and manifest, figures, code-provenance record and original public
validation receipt. No frozen evidence hash was regenerated. The 155 tests include
three new example checks: hand-calculated feature/target timing, completed training
windows, and invariance of prior features, fits and forecasts to future-data damage.
The pinned HMM dependency still emits 1,062 NumPy deprecation warnings; no test failed.

The demo generates synthetic inputs and writes only to standard output. It is not
a new financial experiment, a historical reproduction or investment evidence.
The [original validation receipt](public_validation.json) remains unchanged;
[reproducibility and scope](reproducibility.md) distinguishes these public checks
from the completed licensed-data study.
