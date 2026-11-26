Corpus: 22,457 `.py` files (295 MB), 1,257 byte-identical copies removed, 21,191 documents with ≥ 5 tokens, 44,238,836 distinct 5-gram shingles in total.

Exact join (prefix filter, J ≥ 0.5): 93.3 s, 162,991 candidates verified. Pairs with J ≥ 0.8: **42,908**; keeping one document per connected cluster removes 1,242.

K = 128. (b, r) from `optimal_params(0.8, 128, fp_weight, fn_weight)`: fn_weight 0.5 → 9 × 13, fn_weight 0.9 → 14 × 9.

| hasher | b × r | signatures (s) | candidates | recall | precision | recall, est ≥ t | precision, est ≥ t | removed, LSH clusters | removed, verified (exact: 1,242) |
|---|---|---|---|---|---|---|---|---|---|
| MinHash | 9 × 13 | 46.2 | 34,867 | 0.801 | 0.986 | 0.795 | 0.998 | 1,294 | 1,158 |
| MinHash | 14 × 9 | 46.2 | 46,738 | 0.982 | 0.902 | 0.962 | 0.995 | 1,715 | 1,232 |
| OPH, optimal densification | 9 × 13 | 8.3 | 34,650 | 0.788 | 0.976 | 0.786 | 0.999 | 1,278 | 1,153 |
| OPH, optimal densification | 14 × 9 | 8.3 | 44,572 | 0.963 | 0.927 | 0.950 | 0.997 | 1,667 | 1,231 |
| OPH, rotation densification | 9 × 13 | 2.7 | 36,753 | 0.791 | 0.923 | 0.788 | 0.997 | 1,680 | 1,180 |
| OPH, rotation densification | 14 × 9 | 2.7 | 49,065 | 0.964 | 0.843 | 0.950 | 0.996 | 2,288 | 1,235 |

Candidate rate by true Jaccard, MinHash, 9 × 13:

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.005 | 0.006 | 0.005 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.012 | 0.060 | 0.034 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.077 | 0.087 | 0.128 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.211 | 0.293 | 0.273 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.547 | 0.505 | 0.626 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.537 | 0.793 | 0.795 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 0.970 | 0.949 | 0.973 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

Candidate rate by true Jaccard, MinHash, 14 × 9:

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.063 | 0.067 | 0.071 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.203 | 0.259 | 0.249 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.463 | 0.537 | 0.540 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.746 | 0.837 | 0.756 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.889 | 0.911 | 0.959 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.992 | 0.994 | 0.989 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 1.000 | 1.000 | 1.000 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

Candidate rate by true Jaccard, OPH, optimal densification, 9 × 13:

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.000 | 0.003 | 0.005 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.052 | 0.036 | 0.034 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.112 | 0.139 | 0.128 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.220 | 0.249 | 0.273 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.358 | 0.439 | 0.626 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.643 | 0.715 | 0.795 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 0.930 | 0.986 | 0.973 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

Candidate rate by true Jaccard, OPH, optimal densification, 14 × 9:

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.021 | 0.067 | 0.071 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.186 | 0.223 | 0.249 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.398 | 0.489 | 0.540 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.678 | 0.732 | 0.756 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.794 | 0.899 | 0.959 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.971 | 0.998 | 0.989 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 1.000 | 1.000 | 1.000 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

Candidate rate by true Jaccard, OPH, rotation densification, 9 × 13:

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.009 | 0.066 | 0.005 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.091 | 0.224 | 0.034 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.196 | 0.299 | 0.128 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.317 | 0.395 | 0.273 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.370 | 0.603 | 0.626 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.646 | 0.832 | 0.795 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 0.930 | 0.993 | 0.973 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

Candidate rate by true Jaccard, OPH, rotation densification, 14 × 9:

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.057 | 0.241 | 0.071 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.254 | 0.423 | 0.249 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.580 | 0.659 | 0.540 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.778 | 0.849 | 0.756 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.799 | 0.936 | 0.959 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.971 | 0.999 | 0.989 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 1.000 | 1.000 | 1.000 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

Largest near-duplicate clusters (exact, J ≥ t):

- 267 files, {'plotly': 267}, e.g. `plotly/graph_objs/bar/_insidetextfont.py`
- 50 files, {'plotly': 50}, e.g. `plotly/graph_objs/bar/marker/colorbar/_tickformatstop.py`
- 49 files, {'plotly': 49}, e.g. `plotly/graph_objs/bar/_legendgrouptitle.py`
- 44 files, {'plotly': 44}, e.g. `plotly/graph_objs/bar/_hoverlabel.py`
- 37 files, {'plotly': 37}, e.g. `plotly/graph_objs/bar/marker/_colorbar.py`
- 37 files, {'plotly': 37}, e.g. `plotly/graph_objs/bar/marker/colorbar/_title.py`
- 32 files, {'google': 32}, e.g. `google/genai/_gaos/resources/interactions/annotation/__init__.py`
- 30 files, {'plotly': 30}, e.g. `plotly/graph_objs/bar/selected/_marker.py`
