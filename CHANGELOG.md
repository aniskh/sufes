# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-29
### Added
- Feature `cycle` documentation expanded in `README.md` (flags, outputs, examples).
- `--cycle-cardinality` option: compute and export canonical cycle cardinalities (CSV/JSON).
- `--special-cycles` option: detect (k,i,j) combinations where all n=1..N produce the same cycle length and export to CSV.

### Changed
- Plots for cycle lengths now prefer integer y-axis ticks (improves readability of cycle-length plots).
- CLI: `--special-cycles` flag added and wired to cycle features.
- README updated with full feature list and examples.

### Fixed
- Minor import/indent issues discovered while adding features (syntax corrected and package recompiled).

### Notes
- `--special-cycles` is currently strict: it only reports combinations where *every* n=1..N resulted in a cycle and the cycle length was identical for all n. If a tolerant mode is desired (e.g. threshold-based), consider adding `--special-cycles-threshold` or `--special-cycles-min-count` in a future patch.

