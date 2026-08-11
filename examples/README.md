# Example Plots

This folder holds one rendered example of every chart `ppbcc p3analysis` can
produce. All five show the same benchmark problem — **matrix multiplication**
across six GPU platforms — so the charts can be compared directly.

| File | Chart | Shows |
| --- | --- | --- |
| [`matrixmultiplication_cascade.pdf`](matrixmultiplication_cascade.pdf) | `cascade` | Efficiency decay across ranked platforms, plus performance portability Φ |
| [`matrixmultiplication_navchart.pdf`](matrixmultiplication_navchart.pdf) | `navchart` | Φ plotted against code complexity |
| [`matrixmultiplication_combined.pdf`](matrixmultiplication_combined.pdf) | `combined` | Cascade + Navchart + problem-size scaling in one figure |
| [`matrixmultiplication_heatmap.pdf`](matrixmultiplication_heatmap.pdf) | `heatmap` | Application efficiency per paradigm and platform |
| [`matrixmultiplication_boxplot.pdf`](matrixmultiplication_boxplot.pdf) | `boxplot` | Application-efficiency distribution per paradigm |

They are also embedded, with a description of what each one is good for, in the
[Available Plots](https://schuhmaj.github.io/performance-portability-code-complexity/usage/plots.html)
section of the documentation.

## Where the data comes from

Both input files live in the `results/` folder of the companion repository
[performance-portability-benchmark](https://github.com/schuhmaj/performance-portability-benchmark):

- **Runtimes** — the per-platform `Results_*.csv` files
  (`Results_AMD_Instinct_MI210.csv`,
  `Results_INTEL_Data_Center_GPU_Max_1550.csv`, `Results_NVIDIA_GH200.csv`,
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
ppbcc p3analysis MatrixMultiplication ./Results_* -c cascade \
  --non-zero-pp --remove-description -s avg -x "Cublas"
```

### Navchart

```bash
ppbcc p3analysis MatrixMultiplication ./Results_* \
  --complexity ./code-complexity/code-complexity.csv -c navchart \
  --complexity-metric halstead-difficulty \
  --non-zero-pp --remove-description -s avg -x "Cublas"
```

### Combined

```bash
ppbcc p3analysis MatrixMultiplication ./Results_* \
  --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -x "Cublas"
```

### Heatmap

```bash
ppbcc p3analysis MatrixMultiplication ./Results_* -c heatmap \
  --non-zero-pp --remove-description -s 16384 -x "Cublas"
```

### Boxplot

```bash
ppbcc p3analysis MatrixMultiplication ./Results_* -c boxplot \
  --non-zero-pp --remove-description -s all -x "Cublas"
```

> [!NOTE]
> `ppbcc p3analysis` rejects option/chart combinations that would be silently
> misleading rather than ignoring them:
> - `--complexity` only affects `navchart` and `combined`; elsewhere it is
>   warned about and ignored.
> - `--normalize`/`--additive` require `navchart` or `combined` **with**
>   `--complexity`.
> - `--log-size` is valid for `combined` only.
> - `boxplot` needs `-s all` or an exact numeric size; `avg`, `best`, and
>   `worst` collapse the distribution the chart exists to show.

Add `-l/--legend` to save the legend as a separate PDF (`--legend--vertical`
for a single column), and `-e/--export-to-csv` to also write the underlying
application-efficiency and performance-portability numbers.
