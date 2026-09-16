# Example Plots

This folder holds one rendered example of every chart `ppbcc p2analysis` and
`ppbcc p3analysis` can produce. All but the complexity comparison show the same
benchmark problem — **matrix multiplication** across six GPU platforms — so the
charts can be compared directly; the complexity comparison shows **vector
addition**.

| File | Chart | Shows |
| --- | --- | --- |
| [`matrixmultiplication_cascade.pdf`](matrixmultiplication_cascade.pdf) | `cascade` | Efficiency decay across ranked platforms, plus performance portability Φ |
| [`matrixmultiplication_navchart.pdf`](matrixmultiplication_navchart.pdf) | `navchart` | Φ plotted against code complexity |
| [`matrixmultiplication_combined.pdf`](matrixmultiplication_combined.pdf) | `combined` | Cascade + Navchart + problem-size scaling in one figure |
| [`vecadd_complexity_comparison.pdf`](vecadd_complexity_comparison.pdf) | `complexity-comparison` | Halstead difficulty against SLOC, both relative to plain C++ |
| [`matrixmultiplication_heatmap.pdf`](matrixmultiplication_heatmap.pdf) | `heatmap` | Application efficiency per paradigm and platform |
| [`matrixmultiplication_boxplot.pdf`](matrixmultiplication_boxplot.pdf) | `boxplot` | Application-efficiency distribution per paradigm |
| [`matrixmultiplication_time_barplot.pdf`](matrixmultiplication_time_barplot.pdf) | `time-barplot` | Kernel time per paradigm, grouped by platform |

They are also embedded, with a description of what each one is good for, in the
[Available Plots](https://schuhmaj.github.io/performance-portability-code-complexity/usage/plots.html)
section of the documentation.

## Where the data comes from

Both input files live in the `results/` folder of the companion repository
[performance-portability-benchmark](https://github.com/schuhmaj/performance-portability-benchmark):

- **Runtimes** — the per-platform `Results_*.csv` files
  (`Results_AMD_MI210.csv`, `Results_Intel_Max_1550.csv`,
  `Results_NVIDIA_GH200.csv`,
  `Results_NVIDIA_RTX3080.csv`, `Results_NVIDIA_RTX4060.csv`,
  `Results_NVIDIA_RTX5080.csv`), each consolidated from Google Benchmark JSON
  reports by `ppbcc benchmark`.
- **Complexity** — `results/code-complexity/code-complexity.csv`, produced by
  that repository's `scripts/generate_code_complexity.py` on top of
  `ppbcc code-complexity`.

## Reproducing the plots

All commands are run from the `results/` folder of the benchmark repository:

```bash
cd <performance-portability-benchmark>/results
```

The shared options are:

- `-x "Cublas"` excludes the vendor-optimised reference, which would otherwise
  define the efficiency baseline on its own.
- `--non-zero-pp` computes Φ over the platforms an implementation actually
  supports instead of letting a missing platform force Φ to zero.
- `--remove-description` drops bracketed variant labels such as `[Naive]`.
- `-s avg` averages both metrics over the benchmark sizes.

### Cascade

```bash
ppbcc p2analysis cascade ./Results_* -n MatrixMultiplication \
  --non-zero-pp --remove-description -s avg -x "Cublas"
```

### Navchart

```bash
ppbcc p3analysis navchart ./code-complexity/code-complexity.csv ./Results_* \
  -n MatrixMultiplication -c halstead-difficulty --complexity-metric-absolute \
  --non-zero-pp --remove-description -s avg -x "Cublas"
```

### Combined

```bash
ppbcc p3analysis combined ./code-complexity/code-complexity.csv ./Results_* \
  -n MatrixMultiplication -c halstead-difficulty --log-complexity \
  --non-zero-pp --remove-description -s avg -x "Cublas"
```

### Complexity comparison

```bash
ppbcc p3analysis complexity-comparison ./code-complexity/code-complexity.csv ./Results_* \
  -n VecAdd -c halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "Cublas" --remove-description
```

### Heatmap

```bash
ppbcc p2analysis heatmap ./Results_* -n MatrixMultiplication \
  --non-zero-pp --remove-description -s 16384 -x "Cublas"
```

### Boxplot

```bash
ppbcc p2analysis boxplot ./Results_* -n MatrixMultiplication \
  --non-zero-pp --remove-description -s all -x "Cublas"
```

### Time bar plot

```bash
ppbcc p2analysis time-barplot ./Results_* -n MatrixMultiplication -t kernel \
  -x "Cublas" --remove-description -o matrixmultiplication_time_barplot.pdf
```

The time bar plot uses the largest benchmark size by default. Add
`--normalize-time-to-peak` to multiply every runtime by the platform's
published peak FLOP/s.

> [!NOTE]
> `ppbcc p2analysis` and `ppbcc p3analysis` reject option/chart combinations
> that would be silently misleading rather than ignoring them:
> - `complexity-comparison` rejects `--complexity-metric-absolute`, because its
>   identity line needs both metrics relative to plain C++.
> - `-H/--hardware` is valid for `boxplot` and `time-barplot` only.
> - `-t/--time` and `--normalize-time-to-peak` are valid for `time-barplot`
>   only, and `-t` rejects a runtime column the results do not contain.
> - `boxplot` needs `-s all` or an exact numeric size; `avg`, `best`, and
>   `worst` collapse the distribution the chart exists to show.

Add `-l/--legend` to save the legend as a separate PDF (`--legend--vertical`
for a single column), and `-e/--export-to-csv` to also write the underlying
application-efficiency and performance-portability numbers.
