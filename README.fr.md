# sufes — Analyse de la famille (k, i, j) de Syracuse généralisée

Ce dépôt fournit des outils pour analyser une famille de transformations entières
paramétrée par `(k, i, j)`, inspirée du problème de **Syracuse / Collatz**.
La CLI `python3 -m sufes` calcule des trajectoires, détecte des cycles, génère
des statistiques et des visualisations.

## 📄 Document publié (Zenodo)

Le document associé à ce dépôt est disponible ici :

→ https://zenodo.org/records/20498713

---

## Sommaire

1. [Règle d'itération](#1-règle-ditération)
2. [Notions mathématiques](#2-notions-mathématiques)
3. [Installation](#3-installation)
4. [Vue d'ensemble des features](#4-vue-densemble-des-features)
5. [Référence CLI par feature](#5-référence-cli-par-feature)
   - [5.1 divisions](#51-divisions--diagnostic-a)
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
6. [Options globales](#6-options-globales)
7. [Structure des sorties](#7-structure-des-sorties)
8. [Dépendances](#8-dépendances)

---

## 1. Règle d'itération

La règle implémentée dans `next_term_ji(t, k, j, i)` est :

```
si t % k == 0  →  t ← t // k
sinon r = t % k  →  t ← (k + i) * t + (j*k - i) * r
```

**Variante alternée** (`--alternated`, `--alt-m m`) : le facteur `i` est remplacé
par `i * (-m)^t`. Le code impose `m < k` et ignore les combinaisons invalides.

---

## 2. Notions mathématiques

### Valuation k-adique

Pour un entier `t` et un premier `k`, la valuation `ν_k(t)` est le plus grand
exposant `e` tel que `k^e` divise `t`.

→ [p-adic valuation (Wikipedia)](https://en.wikipedia.org/wiki/P-adic_valuation)

### Entropie de Shannon (sur les résidus non nuls)

$$H(X) = -\sum_{r=1}^{k-1} p_r \log_2(p_r), \quad H_{\max} = \log_2(k-1)$$

→ [Shannon entropy (Wikipedia)](https://en.wikipedia.org/wiki/Entropy_(information_theory))

### Asymétrie (skewness)

$$\gamma_1 = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3}$$

→ [Skewness (Wikipedia)](https://en.wikipedia.org/wiki/Skewness)

### Mixing property (intuition)

La feature `mixing_property` inspecte les paires `(r_t, r_{t+ℓ})` où `r_t = t mod k`
et `ℓ` est un décalage (lag). Un nuage sans structure visible suggère un bon
"mélange" — c'est un outil exploratoire, pas une propriété formelle.

---

## 3. Installation

### Mode développement (editable)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Vérification :

```bash
python3 -m sufes --help
```

### Installation fixe (wheel)

```bash
pip install --upgrade build
python3 -m build
pip install dist/*.whl
```

### Docker

```bash
# Build
docker build -t sufes:local .

# Vérification
docker run --rm sufes:local --help

# Exemple de run (les fichiers générés sont récupérés dans ./output/)
docker run --rm -v "$(pwd)/output:/app/output" sufes:local \
  --single-overall-n 15367 --single-overall-k 17 --single-overall-i 1 --single-overall-j 0
```

---

## 4. Vue d'ensemble des features

| Feature | But principal | Flags principaux |
|---|---|---|
| `divisions` | Statistiques A₀ (valuations k-adiques) | `--divisions-n`, `--divisions-p`, `--divisions-i` |
| `residu-distribution` | Distribution moyenne des résidus `t mod k` | `--residu-distribution-n`, `--residu-distribution-p` |
| `footprint` | Union des nœuds visités par toutes les trajectoires 1..N | `--footprint-n`, `--footprint-p` |
| `cycle` | Détection et cardinalité des cycles canoniques | `--cycle-n`, `--cycle-p` |
| `proof` / `proof-persist` | Preuve de convergence ascendante jusqu'à N | `--proof`, `--proof-persist`, `--proof-p`, `--proof-max-n` |
| `single-n` | Diagnostic complet pour un seul n | `--single-n`, `--single-k`, `--single-p` |
| `single-overall` | Détail résidus pas-à-pas pour un seul (n,k,i,j) | `--single-overall-n`, `--single-overall-k` |
| `spirale` | Trajectoire en coordonnées polaires | `--spirale-n`, `--spirale-k` |
| `stopping` | Temps d'arrêt (stopping time) | `--stopping-n`, `--stopping-p` |
| `pearson` | Corrélation de Pearson entre résidus successifs | `--pearson-n`, `--pearson-p` |
| `altitude` | Pics (peak) et distance à un seuil | `--altitude-n`, `--altitude-p` |
| `gamma` | Métrique γ agrégée sur la trajectoire | `--gamma-n`, `--gamma-p` |
| `shannon-entropy` | Entropie de Shannon sur les résidus non nuls | `--shannon-entropy-n`, `--shannon-entropy-p` |
| `mixing-property` | Lag plot des résidus | `--mixing-property-n`, `--mixing-property-p` |
| `resistance` | Longueur avant la première alternance D→D | `--resistance-n`, `--resistance-p` |
| `lyapunov` | Exposant de Lyapunov empirique | `--lyapunov-n`, `--lyapunov-p` |
| `dirichlet` | Distribution de Dirichlet des résidus | `--dirichlet-n`, `--dirichlet-p` |
| `hamming` | Distance de Hamming entre trajectoires | `--hamming-n`, `--hamming-p` |
| `coalescence` | Comparaison des trajectoires de n et n+1 | `--coalescence-n`, `--coalescence-p` |
| `kernel` | Analyse pour toutes les valeurs n < k | `--kernel`, `--kernel-k`, `--kernel-p` |
| `datalake` | Export JSON resumable sur disque | `--datalake-path`, `--datalake-n` |

---

## 5. Référence CLI par feature

### 5.1 `divisions` — Diagnostic A₀

**But** : pour une valeur de départ `n` et tous les premiers `k ≤ p`, simule la
trajectoire et calcule la fréquence des nœuds dont la valuation k-adique est ≥ 1 :

$$A_0(k) = \frac{\#\{t \text{ visité} : \nu_k(t) \ge 1\}}{\#\{t \text{ visité}\}}$$

Puis compare à la référence $\mathrm{ref}_1(k) = k/(2k-1)$, et rapporte
$\Delta = A_0 - \mathrm{ref}_1$ et le pourcentage $100\Delta/A_0$.

> **Note** : les anciens flags `--epsilon-*` sont encore acceptés comme alias dépréciés.
> Préférez les flags `--divisions-*` pour les nouveaux runs.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--divisions-n N` | int | — | Valeur de départ `n` (obligatoire) |
| `--divisions-p P` | int | — | Borne sup. ; considère tous les k premiers ≤ P (obligatoire) |
| `--divisions-i I` | int | 1 | Paramètre i |
| `--divisions-j J` | int | None | Paramètre j fixe ; si omis, j=0 sauf si `--divisions-j-multi` actif |
| `--divisions-j-multi M` | int | 1 | Boucle sur j ∈ [0, M·k) et agrège les résultats |
| `--divisions-all-n` | flag | False | Boucle sur n₀ = 1..N et calcule la moyenne A₀ agrégée |
| `--divisions-find-best-j` | flag | False | Pour chaque k, cherche le j qui minimise \|Δ\| |
| `--divisions-ordre-multiplicatif-j` | flag | False | Calcule l'ordre multiplicatif de j+1 mod k |
| `--divisions-table` | flag | False | Génère un CSV détaillé (k, v, count_ge_m) |

**Sorties** (dans `output/divisions_YYYYMMDD_HHMMSS_.../`) :

```
divisions_n{N}_p{P}_i{I}.csv
divisions_n{N}_p{P}_i{I}.json
divisions_meanA0_n{N}_p{P}_i{I}.csv   # si --divisions-all-n ou --divisions-j-multi > 1
divisions_meanA0_n{N}_p{P}_i{I}.png   # subplot par k (x = j), si matplotlib
```

**Exemples** :

```bash
# n fixe, j=0
python3 -m sufes --divisions-n 12345678 --divisions-p 1000 --divisions-i 1 --divisions-j 0

# Multi-j sur j ∈ [0, 2k) avec agrégation sur tous les n
python3 -m sufes --divisions-n 1000 --divisions-p 47 --divisions-i 1 \
  --divisions-j-multi 2 --divisions-all-n

# Trouver le meilleur j pour chaque k
python3 -m sufes --divisions-n 12345678 --divisions-p 1000 --divisions-find-best-j
```

---

### 5.2 `residu-distribution`

**But** : pour chaque premier `k ≤ p` et chaque `j ∈ [0, jmult·k)`, calcule la
moyenne des résidus non nuls `r_t = t mod k` le long de la trajectoire démarrant en N.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--residu-distribution-n N` | int | — | Valeur de départ N |
| `--residu-distribution-p P` | int | — | Borne p (k premiers ≤ P) |
| `--residu-distribution-i I` | int | 1 | Paramètre i |
| `--residu-distribution-j J` | int | None | j fixe (sinon boucle sur j) |
| `--residu-distribution-j-mult M` | int | 2 | Multiplicateur plage de j (j ∈ [0, M·k)) |
| `--residu-distribution-all-j` | flag | False | Boucle sur tous les j = 0..k-1 |
| `--residu-distribution-all-n` | flag | False | Agrège toutes les valeurs n₀ = 1..N |
| `--residu-distribution-include-zero` | flag | False | Inclut les résidus nuls dans la moyenne |

**Sorties** :

```
residu_distribution_n{N}_p{P}_jmult{M}.csv
residu_distribution_n{N}_p{P}_jmult{M}.json
residu_distribution_n{N}_p{P}_jmult{M}.png   # si matplotlib
```

**Exemples** :

```bash
python3 -m sufes --residu-distribution-n 1000 --residu-distribution-p 17

python3 -m sufes --residu-distribution-n 10000 --residu-distribution-p 47 \
  --residu-distribution-all-j --residu-distribution-j-mult 2
```

---

### 5.3 `footprint`

**But** : calcule l'union des nœuds visités par les trajectoires de départ `1..N`
et détermine `S(N)`, le plus grand préfixe `1..S` entièrement couvert.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--footprint-n N` | int | — | Borne N |
| `--footprint-k K` | int | — | Diviseur k |
| `--footprint-p P` | int | — | Boucle sur k premiers ≤ P |
| `--footprint-i I` | int | 1 | Paramètre i |
| `--footprint-j J` | int | 0 | Paramètre j |
| `--footprint-j-multi M` | int | 1 | Superpose plusieurs j ∈ [0, M·k) |
| `--footprint-n-multiple-k` | int | — | Fixe N = valeur × k (au lieu de `--footprint-n`) |
| `--footprint-prefixes` | flag | False | Calcule S(N') pour tout 1 ≤ N' ≤ N |
| `--footprint-check-parity` | flag | False | Vérifie la règle de parité de S(N) |
| `--footprint-compact` | flag | False | Évite les fichiers détaillés par (k,j,N) |
| `--footprint-verbose` | flag | False | Force l'écriture des fichiers détaillés |
| `--footprint-n-delta` | flag | False | Trace ΔS(N') = S(N') − S(N'−1) |
| `--footprint-total` | flag | False | Trace le total cumulé de nœuds couverts |

**Règles de parité** (`--footprint-check-parity`) :

- k = 2 : S(N) = N+1 si N pair, sinon S(N) = N
- k ≥ 3 : S(N) = N si N pair, sinon S(N) = N+1

**Exemples** :

```bash
python3 -m sufes --footprint-n 1000 --footprint-k 17 --footprint-i 1 --footprint-j 0

python3 -m sufes --footprint-n 200 --footprint-p 31 --footprint-i 1 \
  --footprint-prefixes --footprint-j-multi 2 --footprint-check-parity

python3 -m sufes --footprint-n-multiple-k 1000 --footprint-p 31 --footprint-i 1 \
  --footprint-prefixes --footprint-compact
```

---

### 5.4 `cycle`

**But** : pour chaque `n ∈ [1, N]` détecte le cycle canonique (rotation minimale)
atteint par la trajectoire et agrège les longueurs de cycle par `(k,i,j)`.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--cycle-n N` | int | — | Borne N |
| `--cycle-k K` | int | — | Diviseur k |
| `--cycle-p P` | int | — | Boucle sur k premiers ≤ P |
| `--cycle-i I` | int | 1 | Paramètre i |
| `--cycle-j J` | int | 0 | Paramètre j |
| `--cycle-all-j` | flag | False | Boucle sur j = 0..k-1 |
| `--cycle-cardinality` | flag | False | Compte les occurrences de chaque cycle canonique |
| `--special-cycles` | flag | False | Signale les (k,i,j) où tous les n ont le même cycle |
| `--extra-special-cycles` | flag | False | Variante plus stricte de `--special-cycles` |
| `--fst-appearance` | flag | False | Enregistre la première apparition de chaque cycle |
| `--cycle-j-multiple M` | int | 1 | Multiplicateur pour la plage de j |
| `--card-top-cycles` | int | 10 | Nombre de cycles fréquents à afficher |

**Exemples** :

```bash
python3 -m sufes --cycle-n 1000 --cycle-p 17 --cycle-i 1

python3 -m sufes --cycle-n 1000 --cycle-p 17 --cycle-i 1 \
  --cycle-cardinality --special-cycles
```

---

### 5.5 `proof` / `proof-persist`

**But** : prouve la convergence pour toutes les valeurs `n ≤ N` de façon ascendante.
Pour chaque n, la simulation s'arrête dès que :

- la trajectoire atteint une valeur `< n` (toutes les valeurs < n étant déjà prouvées), ou
- un cycle est détecté.

Si la trajectoire dépasse `--divergence-threshold` ou n'atteint pas de cycle avant
`--max-iters`, la valeur n est considérée non prouvée (`max_proved = n − 1`).

`--proof-persist` conserve un fichier de progression `proof_k{k}_i{i}_j{j}_maxproved.txt`
par combinaison pour pouvoir reprendre un run interrompu.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--proof` | flag | — | Active le mode proof (run unique) |
| `--proof-persist` | flag | — | Active le mode proof persistant (résumable) |
| `--proof-p P` | int | — | Considère tous les k premiers ≤ P |
| `--proof-k K` | int | — | Exécute uniquement pour ce k |
| `--proof-max-n N` | int | — | Borne N (obligatoire avec `--proof` / `--proof-persist`) |
| `--proof-i I` | int | None | Restreint à une seule valeur de i |
| `--proof-j-mult M` | int | 1 | Étend j dans [0, M·k) |
| `--proof-lake` | flag | False | Mode lake : conserve les nœuds visités comme oracle |
| `--plot-proof` | flag | False | Génère des heatmaps par k |
| `--workers W` | int | 4 | Workers pour paralléliser les combinaisons (k,i,j) |

**Sorties** :

```
proof_p{P}_maxn{N}.csv
proof_p{P}_maxn{N}.json
proof_k{k}_i{i}_j{j}_maxproved.txt   # fichiers de progression (proof-persist seulement)
```

**Exemples** :

```bash
# Test rapide
python3 -m sufes --proof --proof-p 5 --proof-max-n 100

# Mode persistant (reprend après arrêt)
python3 -m sufes --proof-persist --proof-p 17 --proof-max-n 5000 \
  --workers 4 --max-iters 500000 --plot-proof

# Run large — ajuster --workers selon les cœurs disponibles
python3 -m sufes --proof-persist --proof-p 29 --proof-max-n 10000000 --all-i \
  --workers 8 --max-iters 1000000 --divergence-threshold 1e20

# Avec j étendu
python3 -m sufes --proof-persist --proof-p 47 --proof-max-n 100000000 \
  --all-i --proof-j-mult 2 --workers 16 --max-iters 1000000 \
  --divergence-threshold 1e14 --plot-proof
```

> **Conseil** : avant un gros run, lancez d'abord un petit pilote (`--proof-max-n 5000`)
> pour estimer le temps et la consommation mémoire.

---

### 5.6 `single-n`

**But** : diagnostic complet pour une seule valeur `n` — trajectoire, cycle détecté,
stopping time, peak.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--single-n N` | int | — | Valeur n |
| `--single-k K` | int | — | Diviseur k |
| `--single-i I` | int | — | Paramètre i |
| `--single-j J` | int | — | Paramètre j |
| `--single-p P` | int | — | Boucle sur tous les k premiers ≤ P |

**Sorties** :

```
single_n_{N}_k{K}_i{I}_j{J}.json
single_n_{N}_k{K}_i{I}_j{J}.png              # trajectoire
single_n_{N}_p{P}_trajectories.png           # mode --single-p
single_n_{N}_p{P}_steps_perk.png
single_n_{N}_p{P}_peak_perk.png
```

**Exemples** :

```bash
python3 -m sufes --single-n 27 --single-k 3 --single-i 1 --single-j 0

python3 -m sufes --single-n 15367 --single-p 17 --single-i 1
```

---

### 5.7 `single-overall`

**But** : diagnostic détaillé résidus pour un seul `(n,k,i,j)` — tableau pas-à-pas,
distribution complète des résidus, stats (moyenne, variance, écart-type).

Le JSON de sortie contient notamment :
- `count_total`, `count_divisible` : nombre total d'étapes et nombre de divisions
- `lambda_divisible` : $\mathbb{E}[\nu_k(t_s)]$, espérance de la valuation k-adique
- `residue_distribution` : distribution complète des résidus `0..k-1`
- `non_zero_residue_distribution` : distribution restreinte aux résidus non nuls
- `mean_non_zero`, `std_non_zero`, `mean_total`, `std_total`

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--single-overall-n N` | int | — | Valeur n |
| `--single-overall-k K` | int | — | Diviseur k |
| `--single-overall-i I` | int | — | Paramètre i |
| `--single-overall-j J` | int | — | Paramètre j |

> Les anciens flags `--residu-single-overall-*` sont encore acceptés.

**Sorties** :

```
single_overall_n{N}_k{K}_i{I}_j{J}.csv
single_overall_n{N}_k{K}_i{I}_j{J}.json
single_overall_n{N}_k{K}_i{I}_j{J}_trajectory.png
single_overall_n{N}_k{K}_i{I}_j{J}_residues.png
single_overall_n{N}_k{K}_i{I}_j{J}_residue_percentages.png
single_overall_n{N}_k{K}_i{I}_j{J}_residue_percentages_non_zero.png
```

**Exemple** :

```bash
python3 -m sufes --single-overall-n 15367 --single-overall-k 17 \
  --single-overall-i 1 --single-overall-j 0
```

---

### 5.8 `spirale`

**But** : représentation polaire de la trajectoire (angle basé sur le résidu ou
le numéro d'étape, rayon = log(|t|+1)).

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--spirale-n N` | int | — | Valeur n |
| `--spirale-k K` | int | — | Diviseur k |
| `--spirale-p P` | int | — | Boucle sur k premiers ≤ P |
| `--spirale-all` | flag | False | Inclut les k non premiers (avec `--spirale-p`) |
| `--spirale-i I` | int | 1 | Paramètre i |
| `--spirale-j J` | int | 0 | Paramètre j |
| `--spirale-angle-mode` | str | `residue` | `residue` (angle = 2π·(t%k)/k) ou `step` |

**Exemples** :

```bash
python3 -m sufes --spirale-n 15367 --spirale-k 17 --spirale-i 1 --spirale-j 0 \
  --spirale-angle-mode residue

python3 -m sufes --spirale-n 15367 --spirale-p 31 --spirale-i 1 --spirale-j 0
```

---

### 5.9 `stopping`

**But** : calcule le temps d'arrêt (nombre d'étapes avant d'atteindre une valeur
strictement inférieure à `n`) pour chaque `n ∈ [1, N]`.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--stopping-n N` | int | — | Borne N |
| `--stopping-k K` | int | — | Diviseur k |
| `--stopping-p P` | int | — | Boucle sur k premiers ≤ P |
| `--stopping-i I` | int | — | Paramètre i |
| `--stopping-j J` | int | — | Paramètre j |
| `--stopping-all-j` | flag | False | Boucle sur j = 0..k-1 |

**Sorties** :

```
stopping_upto_n{N}_k{K}_i{I}_j{J}_results.json
stopping_upto_n{N}_k{K}_i{I}_j{J}_summary.json
stopping_n{N}_p{P}_stopping_time_by_k.png       # mode --stopping-p
stopping_n{N}_p{P}_mean_stopping_time_by_k.png
```

**Exemple** :

```bash
python3 -m sufes --stopping-n 100 --stopping-p 7 --stopping-i 1 --stopping-all-j
```

---

### 5.10 `pearson`

**But** : corrélation de Pearson entre résidus successifs `(r_t, r_{t+1})`
le long des trajectoires pour `n' = 1..N`.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--pearson-n N` | int | — | Borne N |
| `--pearson-k K` | int | — | Diviseur k |
| `--pearson-p P` | int | — | Boucle sur k premiers ≤ P |
| `--pearson-i I` | int | 1 | Paramètre i |
| `--pearson-j J` | int | 0 | Paramètre j |
| `--pearson-all-j` | flag | False | Boucle sur j = 0..k-1 |

**Sorties** :

```
pearson_upto_n{N}_k{K}_i{I}_j{J}_summary.json
pearson_n{N}_p{P}_summaries.json                # mode --pearson-p
pearson_n{N}_p{P}_by_kj.csv
pearson_n{N}_p{P}_pearson_by_k.png
pearson_n{N}_p{P}_mean_by_k.png
```

**Exemples** :

```bash
python3 -m sufes --pearson-n 100 --pearson-k 5 --pearson-i 1 --pearson-j 0

python3 -m sufes --pearson-n 100 --pearson-p 7 --pearson-i 1 --pearson-all-j
```

---

### 5.11 `altitude`

**But** : mesure le pic maximal atteint (peak) et la distance à un seuil
(altitude) le long des trajectoires pour `n ∈ [1, N]`.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--altitude-n N` | int | — | Borne N |
| `--altitude-k K` | int | — | Diviseur k |
| `--altitude-p P` | int | — | Boucle sur k premiers ≤ P |
| `--altitude-i I` | int | 1 | Paramètre i |
| `--altitude-j J` | int | 0 | Paramètre j |
| `--altitude-partitionning` | flag | False | Active le partitionnement des résultats |

**Exemples** :

```bash
python3 -m sufes --altitude-n 1000 --altitude-k 17 --altitude-i 1 --altitude-j 0

python3 -m sufes --altitude-n 1000 --altitude-p 31 --altitude-i 1
```

---

### 5.12 `gamma`

**But** : calcule une métrique γ agrégée sur la trajectoire pour tous les
premiers `k ≤ p`.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--gamma-n N` | int | — | Valeur de départ |
| `--gamma-p P` | int | — | Boucle sur k premiers ≤ P |
| `--gamma-i I` | int | 1 | Paramètre i |
| `--gamma-j J` | int | 0 | Paramètre j |
| `--gamma-all-i` | flag | False | Boucle sur i = 1..k-1 |
| `--gamma-all-j` | flag | False | Boucle sur j = 0..k-1 |
| `--plot-gamma` | flag | False | Génère un PNG récapitulatif |

**Exemples** :

```bash
python3 -m sufes --gamma-n 15367 --gamma-p 31 --gamma-i 1 --gamma-j 0 --plot-gamma

python3 -m sufes --gamma-n 15367 --gamma-p 31 --gamma-i 1 --gamma-all-j --plot-gamma
```

---

### 5.13 `shannon-entropy`

**But** : entropie de Shannon sur la distribution des résidus non nuls,
comparée au maximum théorique log₂(k-1).

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--shannon-entropy-n N` | int | — | Valeur de départ |
| `--shannon-entropy-k K` | int | — | Diviseur k |
| `--shannon-entropy-p P` | int | — | Boucle sur k premiers ≤ P |
| `--shannon-entropy-i I` | int | 1 | Paramètre i |
| `--shannon-entropy-j J` | int | 0 | Paramètre j |
| `--shannon-entropy-all-j` | flag | False | Boucle sur j = 0..k-1 |

**Exemples** :

```bash
python3 -m sufes --shannon-entropy-n 15367 --shannon-entropy-k 17 \
  --shannon-entropy-i 1 --shannon-entropy-j 0

python3 -m sufes --shannon-entropy-n 15367 --shannon-entropy-p 31 \
  --shannon-entropy-i 1 --shannon-entropy-all-j
```

---

### 5.14 `mixing-property`

**But** : lag plot des paires `(r_t, r_{t+ℓ})` pour visualiser la structure
ou l'absence de structure des résidus.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--mixing-property-n N` | int | — | Valeur de départ |
| `--mixing-property-k K` | int | — | Diviseur k |
| `--mixing-property-p P` | int | — | Boucle sur k premiers ≤ P |
| `--mixing-property-i I` | int | 1 | Paramètre i |
| `--mixing-property-j J` | int | 0 | Paramètre j |
| `--mixing-property-all-j` | flag | False | Boucle sur j = 0..k-1 |
| `--mixing-property-lag L` | int | 1 | Décalage ℓ |
| `--mixing-property-max-points M` | int | — | Limite le nombre de points tracés |

**Exemples** :

```bash
python3 -m sufes --mixing-property-n 15367 --mixing-property-k 17 \
  --mixing-property-i 1 --mixing-property-j 0 --mixing-property-lag 1

python3 -m sufes --mixing-property-n 15367 --mixing-property-p 31 \
  --mixing-property-i 1 --mixing-property-all-j
```

---

### 5.15 `resistance`

**But** : longueur de l'alternance d'opérations (M/D/…) jusqu'à la première
occurrence consécutive D→D.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--resistance-n N` | int | — | Borne N |
| `--resistance-k K` | int | — | Diviseur k |
| `--resistance-p P` | int | — | Boucle sur k premiers ≤ P |
| `--resistance-i I` | int | 1 | Paramètre i |
| `--resistance-j J` | int | 0 | Paramètre j |
| `--resistance-all-j` | flag | False | Boucle sur j = 0..k-1 |
| `--resistance-all-n` | flag | False | Agrège sur n₀ = 1..N |

**Exemples** :

```bash
python3 -m sufes --resistance-n 1000 --resistance-k 17 --resistance-i 1 --resistance-j 0

python3 -m sufes --resistance-n 1000 --resistance-p 31 --resistance-i 1 --resistance-all-j
```

---

### 5.16 `lyapunov`

**But** : exposant de Lyapunov empirique le long des trajectoires.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--lyapunov-n N` | int | — | Valeur de départ |
| `--lyapunov-k K` | int | — | Diviseur k |
| `--lyapunov-p P` | int | — | Boucle sur k premiers ≤ P |
| `--lyapunov-i I` | int | 1 | Paramètre i |
| `--lyapunov-j J` | int | 0 | Paramètre j |

**Exemples** :

```bash
python3 -m sufes --lyapunov-n 15367 --lyapunov-k 17 --lyapunov-i 1 --lyapunov-j 0

python3 -m sufes --lyapunov-n 15367 --lyapunov-p 31 --lyapunov-i 1 --lyapunov-j 0
```

---

### 5.17 `dirichlet`

**But** : distribution de Dirichlet des résidus de la trajectoire.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--dirichlet-n N` | int | — | Borne N |
| `--dirichlet-k K` | int | — | Diviseur k |
| `--dirichlet-p P` | int | — | Boucle sur k premiers ≤ P |
| `--dirichlet-i I` | int | 1 | Paramètre i |
| `--dirichlet-j J` | int | 0 | Paramètre j |
| `--dirichlet-plot-3d` | flag | False | Génère un graphe 3D |

**Exemples** :

```bash
python3 -m sufes --dirichlet-n 1000 --dirichlet-k 17 --dirichlet-i 1 --dirichlet-j 0

python3 -m sufes --dirichlet-n 1000 --dirichlet-p 31 --dirichlet-i 1 --dirichlet-j 0
```

---

### 5.18 `hamming`

**But** : distance de Hamming entre trajectoires encodées sur les résidus.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--hamming-n N` | int | — | Borne N |
| `--hamming-k K` | int | — | Diviseur k |
| `--hamming-p P` | int | — | Boucle sur k premiers ≤ P |
| `--hamming-i I` | int | 1 | Paramètre i |
| `--hamming-j J` | int | 0 | Paramètre j |
| `--hamming-all-j` | flag | False | Boucle sur j = 0..k-1 |

**Exemples** :

```bash
python3 -m sufes --hamming-n 1000 --hamming-k 17 --hamming-i 1 --hamming-j 0

python3 -m sufes --hamming-n 1000 --hamming-p 31 --hamming-i 1 --hamming-all-j
```

---

### 5.19 `coalescence`

**But** : compare les trajectoires de `n` et `n+1` pour détecter à quel pas elles
se rejoignent (coalescence).

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--coalescence-n N` | int | — | Borne N |
| `--coalescence-k K` | int | — | Diviseur k |
| `--coalescence-p P` | int | — | Boucle sur k premiers ≤ P |
| `--coalescence-i I` | int | 1 | Paramètre i |
| `--coalescence-j J` | int | 0 | Paramètre j |
| `--coalescence-j-multi M` | int | 1 | Boucle sur j ∈ [0, M·k) |
| `--coalescence-verbose` | flag | False | Écriture détaillée par paire (n, n+1) |

**Exemples** :

```bash
python3 -m sufes --coalescence-n 1000 --coalescence-k 17 --coalescence-i 1 --coalescence-j 0

python3 -m sufes --coalescence-n 1000 --coalescence-p 31 --coalescence-i 1 --coalescence-j-multi 2
```

---

### 5.20 `kernel`

**But** : analyse de toutes les valeurs `n < k`.

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--kernel` | flag | — | Active le mode kernel (avec `--kernel-k`) |
| `--kernel-k K` | int | — | Diviseur k |
| `--kernel-p P` | int | — | Boucle sur k premiers ≤ P |
| `--kernel-i I` | int | 1 | Paramètre i |
| `--kernel-j J` | int | 0 | Paramètre j |

**Exemples** :

```bash
python3 -m sufes --kernel --kernel-k 17 --kernel-i 1 --kernel-j 0

python3 -m sufes --kernel-p 31 --kernel-i 1 --kernel-j 0
```

---

### 5.21 `datalake`

**But** : exporter les résultats dans une arborescence stable sur disque,
avec reprise automatique après arrêt (checkpoint par tranche).

**Layout** :

```
{datalake_path}/k{k}/i{i}/chunk_XXXXXXXX_YYYYYYYY/data.json
{datalake_path}/k{k}/i{i}/chunk_XXXXXXXX_YYYYYYYY/j{J}.json
{datalake_path}/k{k}/i{i}/cycles/j{J}_cycles.json
{datalake_path}/k{k}/i{i}/checkpoint.json
```

**Flags** :

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--datalake-path PATH` | str | — | Répertoire racine du lac de données |
| `--datalake-n N` | int | — | Borne N (obligatoire) |
| `--datalake-k K` | int | — | Diviseur k |
| `--datalake-p P` | int | — | Boucle sur k premiers ≤ P |
| `--datalake-i-max` | int | k | Boucle i = 1..i_max |
| `--datalake-j-mult M` | int | 2 | j ∈ [0, M·k) |
| `--datalake-base-chunk` | int | 10 000 | Taille de base des tranches |
| `--datalake-trajectory-limit` | int | 200 | Limite les nœuds sauvegardés par trajectoire (0 = aucun) |
| `--datalake-trajectory-hash` | flag | False | Ajoute un SHA256 de chaque trajectoire |

**Exemples** :

```bash
python3 -m sufes --datalake-path /data/sufes_dl --datalake-k 47 --datalake-n 10000000 \
  --datalake-base-chunk 10000 --datalake-j-mult 2

# Tous les k premiers ≤ 47, parallélisé
python3 -m sufes --datalake-path /data/sufes_dl --datalake-p 47 --datalake-n 1000000 \
  --datalake-base-chunk 10000 --datalake-j-mult 2 --workers 4
```

---

## 6. Options globales

Ces options s'appliquent à toutes les features.

| Flag | Type | Défaut | Description |
|---|---|---|---|
| `--start S` | int | 1 | Valeur de départ incluse (intervalle [S, E]) |
| `--end E` | int | 1000 | Valeur de fin incluse |
| `--base B` | int | 3 | Diviseur de base (si `--k` n'est pas fourni) |
| `--k K` | int | None | Diviseur k explicite |
| `--j J` | int | None | Paramètre j (runs non-famille) |
| `--i I` | int | 1 | Paramètre i (runs non-famille) |
| `--p P` | int | None | Boucle famille pour tous les k premiers ≤ P |
| `--kmax M` | int | None | Boucle famille pour k = 2..M |
| `--family` | flag | — | Active la famille complète pour le k donné |
| `--all-i` | flag | — | Considère tous i = 1..k-1 |
| `--compact-json` | flag | — | JSON compact (sans listes d'origines) |
| `--alternated` | flag | — | Active la variante alternée |
| `--alt-m M` | int | 1 | Paramètre m de la variante alternée (doit être < k) |
| `--workers W` | int | 4 | Workers pour la parallélisation |
| `--chunk-size C` | int | 100 000 | Taille des chunks (avec `--workers > 1`) |
| `--max-iters L` | int | 500 000 | Limite d'itérations par trajectoire |
| `--divergence-threshold T` | float | 1e18 | Seuil de divergence numérique |
| `--use-gmpy` | flag | — | Utilise `gmpy2` si disponible (grands entiers) |
| `--use-numba` | flag | — | Utilise `numba` JIT si disponible (expérimental) |
| `--out PATH` | str | None | Fichier JSON de sortie additionnel (optionnel) |

---

## 7. Structure des sorties

À chaque exécution, un dossier daté est créé sous `./output/` :

```
output/{prefix}_{YYYYMMDD}_{HHMMSS}_{suffix}/
  run_info.txt          # paramètres CLI utilisés
  run.log               # capture stdout/stderr
  *.json                # résumés par feature
  *.csv                 # résumés au format CSV
  *.png                 # visualisations (si matplotlib installé)
```

**Préfixes de dossier par feature** :

| Feature | Préfixe dossier |
|---|---|
| `divisions` (ou `--epsilon-*`) | `divisions_` |
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
| Autres / famille | `run_` |

---

## 8. Dépendances

| Package | Rôle | Obligatoire |
|---|---|---|
| Python ≥ 3.8 | Runtime | ✅ |
| `matplotlib` | Génération des PNG | ✅ (pour les plots) |
| `numpy` | Calculs numériques | ✅ (pour les plots) |
| `pandas` | Post-traitement, heatmaps | Optionnel |
| `seaborn` | Visualisations avancées | Optionnel |
| `gmpy2` | Arithmétique grandes valeurs (`--use-gmpy`) | Optionnel |
| `numba` | Accélération JIT (`--use-numba`) | Optionnel |

Installation :

```bash
pip install -r requirements.txt
```

Ou installation minimale :

```bash
pip install matplotlib numpy
```
