# sufes — Analysis of the generalized (k, i, j) Syracuse family

This repository provides tools to analyze a family of integer transformations
parameterized by `(k, i, j)`, inspired by the **Syracuse / Collatz** problem.
The CLI `python3 -m sufes` computes trajectories, detects cycles, generates
statistics and visualizations.

## 📄 Published document (Zenodo)

The document associated with this repository is available here:

→ https://zenodo.org/records/20498713

---

## Table of Contents

1. [Iteration rule](#1-iteration-rule)
2. [Mathematical background](#2-mathematical-background)
3. [Installation](#3-installation)
4. [Feature overview](#4-feature-overview)
5. [CLI reference by feature](#5-cli-reference-by-feature)
   - [5.1 divisions](#51-divisions--a-diagnostic)
   - [5.2 residu-distribution](#52-residu-distribution)
   - [5.3 footprint](#53-footprint)
   - [5.4 cycle](#54-cycle)
   - [5.5 proof / proof-persist](#55-proof--proof-persist)
   - [5.6 single-n](#56-single-n)
   - [5.7 single-overall](#57-single-overall)
   - [5.8 spirale](#58-spirale)
   - [5.9 stopping](#59-stopping)
   - [5.10 pearson](#510-pearson)
   - [5.11 altitude](#511-altitude)
   - [5.12 gamma](#512-gamma)
   - [5.13 shannon-entropy](#513-shannon-entropy)
   - [5.14 mixing-property](#514-mixing-property)
   - [5.15 resistance](#515-resistance)
   - [5.16 lyapunov](#516-lyapunov)
   - [5.17 dirichlet](#517-dirichlet)
   - [5.18 hamming](#518-hamming)
   - [5.19 coalescence](#519-coalescence)
   - [5.20 kernel](#520-kernel)
   - [5.21 datalake](#521-datalake)
6. [Global options](#6-global-options)
7. [Output structure](#7-output-structure)
8. [Dependencies](#8-dependencies)

---

## 1. Iteration rule

The rule implemented in `next_term_ji(t, k, j, i)` is:

```
if t % k == 0  →  t ← t // k
else r = t % k  →  t ← (k + i) * t + (j*k - i) * r
```

**Alternated variant** (`--alternated`, `--alt-m m`): the factor `i` is replaced
by `i * (-m)^t`. The code enforces `m < k` and skips invalid combinations.

---

## 2. Mathematical background

### k-adic valuation

For an integer `t` and a prime `k`, the valuation `ν_k(t)` is the largest
exponent `e` such that `k^e` divides `t`.

→ [p-adic valuation (Wikipedia)](https://en.wikipedia.org/wiki/P-adic_valuation)

### Shannon entropy (over non-zero residues)

$$H(X) = -\sum_{r=1}^{k-1} p_r \log_2(p_r), \quad H_{\max} = \log_2(k-1)$$

→ [Shannon entropy (Wikipedia)](https://en.wikipedia.org/wiki/Entropy_(information_theory))

### Skewness

$$\gamma_1 = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3}$$

→ [Skewness (Wikipedia)](https://en.wikipedia.org/wiki/Skewness)

### Mixing property (intuition)

The `mixing_property` feature inspects pairs `(r_t, r_{t+ℓ})` where `r_t = t mod k`
and `ℓ` is a lag. A structureless scatter plot suggests good "mixing" — this is
an exploratory tool, not a formal ergodic property.

---

## 3. Installation

### Development mode (editable install)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Verification:

```bash
python3 -m sufes --help
```

### Fixed install (wheel)

```bash
pip install --upgrade build
python3 -m build
pip install dist/*.whl
```

### Docker

```bash
# Build
docker build -t sufes:local .

# Verify
docker run --rm sufes:local --help

# Example run (output files are retrieved in ./output/)
docker run --rm -v "$(pwd)/output:/app/output" sufes:local \
  --single-overall-n 15367 --single-overall-k 17 --single-overall-i 1 --single-overall-j 0
```

---

## 4. Feature overview

| Feature | Purpose | Main flags |
|---|---|---|
| `divisions` | A₀ statistics (k-adic valuations) | `--divisions-n`, `--divisions-p`, `--divisions-i` |
| `residu-distribution` | Mean distribution of residues `t mod k` | `--residu-distribution-n`, `--residu-distribution-p` |
| `footprint` | Union of nodes visited by all trajectories 1..N | `--footprint-n`, `--footprint-p` |
| `cycle` | Detection and cardinality of canonical cycles | `--cycle-n`, `--cycle-p` |
| `proof` / `proof-persist` | Ascending convergence proof up to N | `--proof`, `--proof-persist`, `--proof-p`, `--proof-max-n` |
| `single-n` | Full diagnostic for a single n | `--single-n`, `--single-k`, `--single-p` |
| `single-overall` | Step-by-step residue detail for a single (n,k,i,j) | `--single-overall-n`, `--single-overall-k` |
| `spirale` | Trajectory in polar coordinates | `--spirale-n`, `--spirale-k` |
| `stopping` | Stopping time | `--stopping-n`, `--stopping-p` |
| `pearson` | Pearson correlation between successive residues | `--pearson-n`, `--pearson-p` |
| `altitude` | Trajectory peaks and distance to a threshold | `--altitude-n`, `--altitude-p` |
| `gamma` | γ metric aggregated over the trajectory | `--gamma-n`, `--gamma-p` |
| `shannon-entropy` | Shannon entropy over non-zero residues | `--shannon-entropy-n`, `--shannon-entropy-p` |
| `mixing-property` | Lag plot of residues | `--mixing-property-n`, `--mixing-property-p` |
| `resistance` | Length before first consecutive D→D operation | `--resistance-n`, `--resistance-p` |
| `lyapunov` | Empirical Lyapunov exponent | `--lyapunov-n`, `--lyapunov-p` |
| `dirichlet` | Dirichlet distribution of residues | `--dirichlet-n`, `--dirichlet-p` |
| `hamming` | Hamming distance between trajectories | `--hamming-n`, `--hamming-p` |
| `coalescence` | Comparison of trajectories of n and n+1 | `--coalescence-n`, `--coalescence-p` |
| `kernel` | Analysis for all values n < k | `--kernel`, `--kernel-k`, `--kernel-p` |
| `datalake` | Resumable JSON export to disk | `--datalake-path`, `--datalake-n` |

---

## 5. CLI reference by feature

### 5.1 `divisions` — A₀ diagnostic

**Purpose**: for a starting value `n` and all primes `k ≤ p`, simulates the
trajectory and computes the frequency of visited nodes whose k-adic valuation is ≥ 1:

$$A_0(k) = \frac{\#\{t \text{ visited} : \nu_k(t) \ge 1\}}{\#\{t \text{ visited}\}}$$

Then compares to the reference $\mathrm{ref}_1(k) = k/(2k-1)$, and reports
$\Delta = A_0 - \mathrm{ref}_1$ and the percentage $100\Delta/A_0$.

> **Note**: the old `--epsilon-*` flags are still accepted as deprecated aliases.
> Use `--divisions-*` flags for new runs.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--divisions-n N` | int | — | Starting value `n` (required) |
| `--divisions-p P` | int | — | Upper bound; considers all primes k ≤ P (required) |
| `--divisions-i I` | int | 1 | Parameter i |
| `--divisions-j J` | int | None | Fixed j; if omitted, j=0 unless `--divisions-j-multi` is active |
| `--divisions-j-multi M` | int | 1 | Loop over j ∈ [0, M·k) and aggregate results |
| `--divisions-all-n` | flag | False | Loop over n₀ = 1..N and compute the aggregated mean A₀ |
| `--divisions-find-best-j` | flag | False | For each k, find the j minimizing \|Δ\| |
| `--divisions-ordre-multiplicatif-j` | flag | False | Compute the multiplicative order of j+1 mod k |
| `--divisions-table` | flag | False | Generate a detailed CSV (k, v, count_ge_m) |

**Outputs** (in `output/divisions_YYYYMMDD_HHMMSS_.../`):

```
divisions_n{N}_p{P}_i{I}.csv
divisions_n{N}_p{P}_i{I}.json
divisions_meanA0_n{N}_p{P}_i{I}.csv   # if --divisions-all-n or --divisions-j-multi > 1
divisions_meanA0_n{N}_p{P}_i{I}.png   # subplot per k (x = j), if matplotlib available
```

**Examples**:

```bash
# Fixed n, j=0
python3 -m sufes --divisions-n 12345678 --divisions-p 1000 --divisions-i 1 --divisions-j 0

# Multi-j over j ∈ [0, 2k) with aggregation over all n
python3 -m sufes --divisions-n 1000 --divisions-p 47 --divisions-i 1 \
  --divisions-j-multi 2 --divisions-all-n

# Find the best j for each k
python3 -m sufes --divisions-n 12345678 --divisions-p 1000 --divisions-find-best-j
```

---

### 5.2 `residu-distribution`

**Purpose**: for each prime `k ≤ p` and each `j ∈ [0, jmult·k)`, computes the
mean of non-zero residues `r_t = t mod k` along the trajectory starting at N.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--residu-distribution-n N` | int | — | Starting value N |
| `--residu-distribution-p P` | int | — | Bound p (primes k ≤ P) |
| `--residu-distribution-i I` | int | 1 | Parameter i |
| `--residu-distribution-j J` | int | None | Fixed j (otherwise loops over j) |
| `--residu-distribution-j-mult M` | int | 2 | j range multiplier (j ∈ [0, M·k)) |
| `--residu-distribution-all-j` | flag | False | Loop over all j = 0..k-1 |
| `--residu-distribution-all-n` | flag | False | Aggregate all values n₀ = 1..N |
| `--residu-distribution-include-zero` | flag | False | Include zero residues in the mean |

**Outputs**:

```
residu_distribution_n{N}_p{P}_jmult{M}.csv
residu_distribution_n{N}_p{P}_jmult{M}.json
residu_distribution_n{N}_p{P}_jmult{M}.png   # if matplotlib available
```

**Examples**:

```bash
python3 -m sufes --residu-distribution-n 1000 --residu-distribution-p 17

python3 -m sufes --residu-distribution-n 10000 --residu-distribution-p 47 \
  --residu-distribution-all-j --residu-distribution-j-mult 2
```

---

### 5.3 `footprint`

**Purpose**: computes the union of nodes visited by starting trajectories `1..N`
and determines `S(N)`, the largest prefix `1..S` fully covered.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--footprint-n N` | int | — | Bound N |
| `--footprint-k K` | int | — | Divisor k |
| `--footprint-p P` | int | — | Loop over primes k ≤ P |
| `--footprint-i I` | int | 1 | Parameter i |
| `--footprint-j J` | int | 0 | Parameter j |
| `--footprint-j-multi M` | int | 1 | Overlay multiple j ∈ [0, M·k) |
| `--footprint-n-multiple-k` | int | — | Set N = value × k (instead of `--footprint-n`) |
| `--footprint-prefixes` | flag | False | Compute S(N') for all 1 ≤ N' ≤ N |
| `--footprint-check-parity` | flag | False | Check the parity rule for S(N) |
| `--footprint-compact` | flag | False | Skip detailed per-(k,j,N) files |
| `--footprint-verbose` | flag | False | Force writing detailed files |
| `--footprint-n-delta` | flag | False | Plot ΔS(N') = S(N') − S(N'−1) |
| `--footprint-total` | flag | False | Plot the cumulative total of covered nodes |

**Parity rules** (`--footprint-check-parity`):

- k = 2: S(N) = N+1 if N is even, otherwise S(N) = N
- k ≥ 3: S(N) = N if N is even, otherwise S(N) = N+1

**Examples**:

```bash
python3 -m sufes --footprint-n 1000 --footprint-k 17 --footprint-i 1 --footprint-j 0

python3 -m sufes --footprint-n 200 --footprint-p 31 --footprint-i 1 \
  --footprint-prefixes --footprint-j-multi 2 --footprint-check-parity

python3 -m sufes --footprint-n-multiple-k 1000 --footprint-p 31 --footprint-i 1 \
  --footprint-prefixes --footprint-compact
```

---

### 5.4 `cycle`

**Purpose**: for each `n ∈ [1, N]` detects the canonical cycle (minimal rotation)
reached by the trajectory and aggregates cycle lengths by `(k,i,j)`.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cycle-n N` | int | — | Bound N |
| `--cycle-k K` | int | — | Divisor k |
| `--cycle-p P` | int | — | Loop over primes k ≤ P |
| `--cycle-i I` | int | 1 | Parameter i |
| `--cycle-j J` | int | 0 | Parameter j |
| `--cycle-all-j` | flag | False | Loop over j = 0..k-1 |
| `--cycle-cardinality` | flag | False | Count occurrences of each canonical cycle |
| `--special-cycles` | flag | False | Report (k,i,j) where all n share the same cycle |
| `--extra-special-cycles` | flag | False | Stricter variant of `--special-cycles` |
| `--fst-appearance` | flag | False | Record the first appearance of each cycle |
| `--cycle-j-multiple M` | int | 1 | Multiplier for the j range |
| `--card-top-cycles` | int | 10 | Number of most frequent cycles to display |

**Examples**:

```bash
python3 -m sufes --cycle-n 1000 --cycle-p 17 --cycle-i 1

python3 -m sufes --cycle-n 1000 --cycle-p 17 --cycle-i 1 \
  --cycle-cardinality --special-cycles
```

---

### 5.5 `proof` / `proof-persist`

**Purpose**: proves convergence for all values `n ≤ N` in ascending order.
For each n, the simulation stops as soon as:

- the trajectory reaches a value `< n` (all values < n already proven), or
- a cycle is detected.

If the trajectory exceeds `--divergence-threshold` or does not reach a cycle before
`--max-iters`, the value n is considered unproven (`max_proved = n − 1`).

`--proof-persist` keeps a progress file `proof_k{k}_i{i}_j{j}_maxproved.txt`
per combination to allow resuming an interrupted run.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--proof` | flag | — | Enable proof mode (single run) |
| `--proof-persist` | flag | — | Enable persistent proof mode (resumable) |
| `--proof-p P` | int | — | Consider all primes k ≤ P |
| `--proof-k K` | int | — | Run only for this k |
| `--proof-max-n N` | int | — | Bound N (required with `--proof` / `--proof-persist`) |
| `--proof-i I` | int | None | Restrict to a single value of i |
| `--proof-j-mult M` | int | 1 | Extend j range to [0, M·k) |
| `--proof-lake` | flag | False | Lake mode: keep visited nodes as an oracle |
| `--plot-proof` | flag | False | Generate heatmaps per k |
| `--workers W` | int | 4 | Workers to parallelize (k,i,j) combinations |

**Outputs**:

```
proof_p{P}_maxn{N}.csv
proof_p{P}_maxn{N}.json
proof_k{k}_i{i}_j{j}_maxproved.txt   # progress files (proof-persist only)
```

**Examples**:

```bash
# Quick test
python3 -m sufes --proof --proof-p 5 --proof-max-n 100

# Persistent mode (resumes after interruption)
python3 -m sufes --proof-persist --proof-p 17 --proof-max-n 5000 \
  --workers 4 --max-iters 500000 --plot-proof

# Large run — adjust --workers to available cores
python3 -m sufes --proof-persist --proof-p 29 --proof-max-n 10000000 --all-i \
  --workers 8 --max-iters 1000000 --divergence-threshold 1e20

# Extended j range
python3 -m sufes --proof-persist --proof-p 47 --proof-max-n 100000000 \
  --all-i --proof-j-mult 2 --workers 16 --max-iters 1000000 \
  --divergence-threshold 1e14 --plot-proof
```

> **Tip**: before a large run, first launch a small pilot (`--proof-max-n 5000`)
> to estimate runtime and memory usage.

---

### 5.6 `single-n`

**Purpose**: full diagnostic for a single value `n` — trajectory, detected cycle,
stopping time, peak.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--single-n N` | int | — | Value n |
| `--single-k K` | int | — | Divisor k |
| `--single-i I` | int | — | Parameter i |
| `--single-j J` | int | — | Parameter j |
| `--single-p P` | int | — | Loop over all primes k ≤ P |

**Outputs**:

```
single_n_{N}_k{K}_i{I}_j{J}.json
single_n_{N}_k{K}_i{I}_j{J}.png              # trajectory
single_n_{N}_p{P}_trajectories.png           # --single-p mode
single_n_{N}_p{P}_steps_perk.png
single_n_{N}_p{P}_peak_perk.png
```

**Examples**:

```bash
python3 -m sufes --single-n 27 --single-k 3 --single-i 1 --single-j 0

python3 -m sufes --single-n 15367 --single-p 17 --single-i 1
```

---

### 5.7 `single-overall`

**Purpose**: detailed residue diagnostic for a single `(n,k,i,j)` — step-by-step
table, full residue distribution, statistics (mean, variance, standard deviation).

The output JSON contains in particular:
- `count_total`, `count_divisible`: total number of steps and number of divisions
- `lambda_divisible`: $\mathbb{E}[\nu_k(t_s)]$, expected k-adic valuation
- `residue_distribution`: full distribution of residues `0..k-1`
- `non_zero_residue_distribution`: distribution restricted to non-zero residues
- `mean_non_zero`, `std_non_zero`, `mean_total`, `std_total`

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--single-overall-n N` | int | — | Value n |
| `--single-overall-k K` | int | — | Divisor k |
| `--single-overall-i I` | int | — | Parameter i |
| `--single-overall-j J` | int | — | Parameter j |

> The legacy flags `--residu-single-overall-*` are still accepted.

**Outputs**:

```
single_overall_n{N}_k{K}_i{I}_j{J}.csv
single_overall_n{N}_k{K}_i{I}_j{J}.json
single_overall_n{N}_k{K}_i{I}_j{J}_trajectory.png
single_overall_n{N}_k{K}_i{I}_j{J}_residues.png
single_overall_n{N}_k{K}_i{I}_j{J}_residue_percentages.png
single_overall_n{N}_k{K}_i{I}_j{J}_residue_percentages_non_zero.png
```

**Example**:

```bash
python3 -m sufes --single-overall-n 15367 --single-overall-k 17 \
  --single-overall-i 1 --single-overall-j 0
```

---

### 5.8 `spirale`

**Purpose**: polar representation of the trajectory (angle based on residue or
step number, radius = log(|t|+1)).

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--spirale-n N` | int | — | Value n |
| `--spirale-k K` | int | — | Divisor k |
| `--spirale-p P` | int | — | Loop over primes k ≤ P |
| `--spirale-all` | flag | False | Include non-prime k (with `--spirale-p`) |
| `--spirale-i I` | int | 1 | Parameter i |
| `--spirale-j J` | int | 0 | Parameter j |
| `--spirale-angle-mode` | str | `residue` | `residue` (angle = 2π·(t%k)/k) or `step` |

**Examples**:

```bash
python3 -m sufes --spirale-n 15367 --spirale-k 17 --spirale-i 1 --spirale-j 0 \
  --spirale-angle-mode residue

python3 -m sufes --spirale-n 15367 --spirale-p 31 --spirale-i 1 --spirale-j 0
```

---

### 5.9 `stopping`

**Purpose**: computes the stopping time (number of steps before reaching a value
strictly less than `n`) for each `n ∈ [1, N]`.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--stopping-n N` | int | — | Bound N |
| `--stopping-k K` | int | — | Divisor k |
| `--stopping-p P` | int | — | Loop over primes k ≤ P |
| `--stopping-i I` | int | — | Parameter i |
| `--stopping-j J` | int | — | Parameter j |
| `--stopping-all-j` | flag | False | Loop over j = 0..k-1 |

**Outputs**:

```
stopping_upto_n{N}_k{K}_i{I}_j{J}_results.json
stopping_upto_n{N}_k{K}_i{I}_j{J}_summary.json
stopping_n{N}_p{P}_stopping_time_by_k.png       # --stopping-p mode
stopping_n{N}_p{P}_mean_stopping_time_by_k.png
```

**Example**:

```bash
python3 -m sufes --stopping-n 100 --stopping-p 7 --stopping-i 1 --stopping-all-j
```

---

### 5.10 `pearson`

**Purpose**: Pearson correlation between successive residues `(r_t, r_{t+1})`
along trajectories for `n' = 1..N`.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--pearson-n N` | int | — | Bound N |
| `--pearson-k K` | int | — | Divisor k |
| `--pearson-p P` | int | — | Loop over primes k ≤ P |
| `--pearson-i I` | int | 1 | Parameter i |
| `--pearson-j J` | int | 0 | Parameter j |
| `--pearson-all-j` | flag | False | Loop over j = 0..k-1 |

**Outputs**:

```
pearson_upto_n{N}_k{K}_i{I}_j{J}_summary.json
pearson_n{N}_p{P}_summaries.json                # --pearson-p mode
pearson_n{N}_p{P}_by_kj.csv
pearson_n{N}_p{P}_pearson_by_k.png
pearson_n{N}_p{P}_mean_by_k.png
```

**Examples**:

```bash
python3 -m sufes --pearson-n 100 --pearson-k 5 --pearson-i 1 --pearson-j 0

python3 -m sufes --pearson-n 100 --pearson-p 7 --pearson-i 1 --pearson-all-j
```

---

### 5.11 `altitude`

**Purpose**: measures the maximum peak reached and the distance to a threshold
(altitude) along trajectories for `n ∈ [1, N]`.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--altitude-n N` | int | — | Bound N |
| `--altitude-k K` | int | — | Divisor k |
| `--altitude-p P` | int | — | Loop over primes k ≤ P |
| `--altitude-i I` | int | 1 | Parameter i |
| `--altitude-j J` | int | 0 | Parameter j |
| `--altitude-partitionning` | flag | False | Enable result partitioning |

**Examples**:

```bash
python3 -m sufes --altitude-n 1000 --altitude-k 17 --altitude-i 1 --altitude-j 0

python3 -m sufes --altitude-n 1000 --altitude-p 31 --altitude-i 1
```

---

### 5.12 `gamma`

**Purpose**: computes a γ metric aggregated over the trajectory for all
primes `k ≤ p`.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--gamma-n N` | int | — | Starting value |
| `--gamma-p P` | int | — | Loop over primes k ≤ P |
| `--gamma-i I` | int | 1 | Parameter i |
| `--gamma-j J` | int | 0 | Parameter j |
| `--gamma-all-i` | flag | False | Loop over i = 1..k-1 |
| `--gamma-all-j` | flag | False | Loop over j = 0..k-1 |
| `--plot-gamma` | flag | False | Generate a summary PNG |

**Examples**:

```bash
python3 -m sufes --gamma-n 15367 --gamma-p 31 --gamma-i 1 --gamma-j 0 --plot-gamma

python3 -m sufes --gamma-n 15367 --gamma-p 31 --gamma-i 1 --gamma-all-j --plot-gamma
```

---

### 5.13 `shannon-entropy`

**Purpose**: Shannon entropy over the distribution of non-zero residues,
compared to the theoretical maximum log₂(k-1).

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--shannon-entropy-n N` | int | — | Starting value |
| `--shannon-entropy-k K` | int | — | Divisor k |
| `--shannon-entropy-p P` | int | — | Loop over primes k ≤ P |
| `--shannon-entropy-i I` | int | 1 | Parameter i |
| `--shannon-entropy-j J` | int | 0 | Parameter j |
| `--shannon-entropy-all-j` | flag | False | Loop over j = 0..k-1 |

**Examples**:

```bash
python3 -m sufes --shannon-entropy-n 15367 --shannon-entropy-k 17 \
  --shannon-entropy-i 1 --shannon-entropy-j 0

python3 -m sufes --shannon-entropy-n 15367 --shannon-entropy-p 31 \
  --shannon-entropy-i 1 --shannon-entropy-all-j
```

---

### 5.14 `mixing-property`

**Purpose**: lag plot of pairs `(r_t, r_{t+ℓ})` to visualize the structure
or lack of structure in residues.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--mixing-property-n N` | int | — | Starting value |
| `--mixing-property-k K` | int | — | Divisor k |
| `--mixing-property-p P` | int | — | Loop over primes k ≤ P |
| `--mixing-property-i I` | int | 1 | Parameter i |
| `--mixing-property-j J` | int | 0 | Parameter j |
| `--mixing-property-all-j` | flag | False | Loop over j = 0..k-1 |
| `--mixing-property-lag L` | int | 1 | Lag ℓ |
| `--mixing-property-max-points M` | int | — | Limit the number of plotted points |

**Examples**:

```bash
python3 -m sufes --mixing-property-n 15367 --mixing-property-k 17 \
  --mixing-property-i 1 --mixing-property-j 0 --mixing-property-lag 1

python3 -m sufes --mixing-property-n 15367 --mixing-property-p 31 \
  --mixing-property-i 1 --mixing-property-all-j
```

---

### 5.15 `resistance`

**Purpose**: length of the operation alternation (M/D/…) until the first
consecutive D→D occurrence.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--resistance-n N` | int | — | Bound N |
| `--resistance-k K` | int | — | Divisor k |
| `--resistance-p P` | int | — | Loop over primes k ≤ P |
| `--resistance-i I` | int | 1 | Parameter i |
| `--resistance-j J` | int | 0 | Parameter j |
| `--resistance-all-j` | flag | False | Loop over j = 0..k-1 |
| `--resistance-all-n` | flag | False | Aggregate over n₀ = 1..N |

**Examples**:

```bash
python3 -m sufes --resistance-n 1000 --resistance-k 17 --resistance-i 1 --resistance-j 0

python3 -m sufes --resistance-n 1000 --resistance-p 31 --resistance-i 1 --resistance-all-j
```

---

### 5.16 `lyapunov`

**Purpose**: empirical Lyapunov exponent along trajectories.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lyapunov-n N` | int | — | Starting value |
| `--lyapunov-k K` | int | — | Divisor k |
| `--lyapunov-p P` | int | — | Loop over primes k ≤ P |
| `--lyapunov-i I` | int | 1 | Parameter i |
| `--lyapunov-j J` | int | 0 | Parameter j |

**Examples**:

```bash
python3 -m sufes --lyapunov-n 15367 --lyapunov-k 17 --lyapunov-i 1 --lyapunov-j 0

python3 -m sufes --lyapunov-n 15367 --lyapunov-p 31 --lyapunov-i 1 --lyapunov-j 0
```

---

### 5.17 `dirichlet`

**Purpose**: Dirichlet distribution of trajectory residues.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dirichlet-n N` | int | — | Bound N |
| `--dirichlet-k K` | int | — | Divisor k |
| `--dirichlet-p P` | int | — | Loop over primes k ≤ P |
| `--dirichlet-i I` | int | 1 | Parameter i |
| `--dirichlet-j J` | int | 0 | Parameter j |
| `--dirichlet-plot-3d` | flag | False | Generate a 3D plot |

**Examples**:

```bash
python3 -m sufes --dirichlet-n 1000 --dirichlet-k 17 --dirichlet-i 1 --dirichlet-j 0

python3 -m sufes --dirichlet-n 1000 --dirichlet-p 31 --dirichlet-i 1 --dirichlet-j 0
```

---

### 5.18 `hamming`

**Purpose**: Hamming distance between residue-encoded trajectories.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--hamming-n N` | int | — | Bound N |
| `--hamming-k K` | int | — | Divisor k |
| `--hamming-p P` | int | — | Loop over primes k ≤ P |
| `--hamming-i I` | int | 1 | Parameter i |
| `--hamming-j J` | int | 0 | Parameter j |
| `--hamming-all-j` | flag | False | Loop over j = 0..k-1 |

**Examples**:

```bash
python3 -m sufes --hamming-n 1000 --hamming-k 17 --hamming-i 1 --hamming-j 0

python3 -m sufes --hamming-n 1000 --hamming-p 31 --hamming-i 1 --hamming-all-j
```

---

### 5.19 `coalescence`

**Purpose**: compares the trajectories of `n` and `n+1` to detect at which step
they merge (coalesce).

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--coalescence-n N` | int | — | Bound N |
| `--coalescence-k K` | int | — | Divisor k |
| `--coalescence-p P` | int | — | Loop over primes k ≤ P |
| `--coalescence-i I` | int | 1 | Parameter i |
| `--coalescence-j J` | int | 0 | Parameter j |
| `--coalescence-j-multi M` | int | 1 | Loop over j ∈ [0, M·k) |
| `--coalescence-verbose` | flag | False | Detailed output per (n, n+1) pair |

**Examples**:

```bash
python3 -m sufes --coalescence-n 1000 --coalescence-k 17 --coalescence-i 1 --coalescence-j 0

python3 -m sufes --coalescence-n 1000 --coalescence-p 31 --coalescence-i 1 --coalescence-j-multi 2
```

---

### 5.20 `kernel`

**Purpose**: analysis for all values `n < k`.

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--kernel` | flag | — | Enable kernel mode (with `--kernel-k`) |
| `--kernel-k K` | int | — | Divisor k |
| `--kernel-p P` | int | — | Loop over primes k ≤ P |
| `--kernel-i I` | int | 1 | Parameter i |
| `--kernel-j J` | int | 0 | Parameter j |

**Examples**:

```bash
python3 -m sufes --kernel --kernel-k 17 --kernel-i 1 --kernel-j 0

python3 -m sufes --kernel-p 31 --kernel-i 1 --kernel-j 0
```

---

### 5.21 `datalake`

**Purpose**: export results to a stable on-disk directory tree,
with automatic resume after interruption (checkpoint per chunk).

**Layout**:

```
{datalake_path}/k{k}/i{i}/chunk_XXXXXXXX_YYYYYYYY/data.json
{datalake_path}/k{k}/i{i}/chunk_XXXXXXXX_YYYYYYYY/j{J}.json
{datalake_path}/k{k}/i{i}/cycles/j{J}_cycles.json
{datalake_path}/k{k}/i{i}/checkpoint.json
```

**Flags**:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--datalake-path PATH` | str | — | Root directory of the data lake |
| `--datalake-n N` | int | — | Bound N (required) |
| `--datalake-k K` | int | — | Divisor k |
| `--datalake-p P` | int | — | Loop over primes k ≤ P |
| `--datalake-i-max` | int | k | Loop i = 1..i_max |
| `--datalake-j-mult M` | int | 2 | j ∈ [0, M·k) |
| `--datalake-base-chunk` | int | 10,000 | Base chunk size |
| `--datalake-trajectory-limit` | int | 200 | Limit saved nodes per trajectory (0 = none) |
| `--datalake-trajectory-hash` | flag | False | Add a SHA256 hash for each trajectory |

**Examples**:

```bash
python3 -m sufes --datalake-path /data/sufes_dl --datalake-k 47 --datalake-n 10000000 \
  --datalake-base-chunk 10000 --datalake-j-mult 2

# All primes k ≤ 47, parallelized
python3 -m sufes --datalake-path /data/sufes_dl --datalake-p 47 --datalake-n 1000000 \
  --datalake-base-chunk 10000 --datalake-j-mult 2 --workers 4
```

---

## 6. Global options

These options apply to all features.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--start S` | int | 1 | Inclusive start value (range [S, E]) |
| `--end E` | int | 1000 | Inclusive end value |
| `--base B` | int | 3 | Base divisor (if `--k` is not provided) |
| `--k K` | int | None | Explicit divisor k |
| `--j J` | int | None | Parameter j (non-family runs) |
| `--i I` | int | 1 | Parameter i (non-family runs) |
| `--p P` | int | None | Family loop over all primes k ≤ P |
| `--kmax M` | int | None | Family loop for k = 2..M |
| `--family` | flag | — | Enable the full family for the given k |
| `--all-i` | flag | — | Consider all i = 1..k-1 |
| `--compact-json` | flag | — | Compact JSON (without origin lists) |
| `--alternated` | flag | — | Enable the alternated variant |
| `--alt-m M` | int | 1 | Parameter m of the alternated variant (must be < k) |
| `--workers W` | int | 4 | Workers for parallelization |
| `--chunk-size C` | int | 100,000 | Chunk size (with `--workers > 1`) |
| `--max-iters L` | int | 500,000 | Iteration limit per trajectory |
| `--divergence-threshold T` | float | 1e18 | Numerical divergence threshold |
| `--use-gmpy` | flag | — | Use `gmpy2` if available (large integers) |
| `--use-numba` | flag | — | Use `numba` JIT if available (experimental) |
| `--out PATH` | str | None | Additional JSON output file (optional) |

---

## 7. Output structure

At each execution, a timestamped folder is created under `./output/`:

```
output/{prefix}_{YYYYMMDD}_{HHMMSS}_{suffix}/
  run_info.txt          # CLI parameters used
  run.log               # captured stdout/stderr
  *.json                # feature summaries
  *.csv                 # summaries in CSV format
  *.png                 # visualizations (if matplotlib is installed)
```

**Output folder prefixes by feature**:

| Feature | Folder prefix |
|---|---|
| `divisions` (or `--epsilon-*`) | `divisions_` |
| `residu-distribution` | `residu-distribution_` |
| `footprint` | `footprint_` |
| `cycle` | `cycle_` |
| `proof` | `proof_` |
| `proof-persist` | `proof-persist_` |
| `spirale` | `spirale_` |
| `stopping` | `stopping_` |
| `single-n` | `single-n_` |
| `single-overall` | `single-overall_` |
| `altitude` | `altitude_` |
| `gamma` | `gamma_` |
| `shannon-entropy` | `shannon_entropy_` |
| `mixing-property` | `mixing_property_` |
| `coalescence` | `coalescence_` |
| `pearson` | `pearson_` |
| `resistance` | `resistance_` |
| Other / family | `run_` |

---

## 8. Dependencies

| Package | Role | Required |
|---|---|---|
| Python ≥ 3.8 | Runtime | ✅ |
| `matplotlib` | PNG generation | ✅ (for plots) |
| `numpy` | Numerical computations | ✅ (for plots) |
| `pandas` | Post-processing, heatmaps | Optional |
| `seaborn` | Advanced visualizations | Optional |
| `gmpy2` | Large integer arithmetic (`--use-gmpy`) | Optional |
| `numba` | JIT acceleration (`--use-numba`) | Optional |

Installation:

```bash
pip install -r requirements.txt
```

Or minimal installation:

```bash
pip install matplotlib numpy
```
