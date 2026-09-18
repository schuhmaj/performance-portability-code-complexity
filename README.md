# ppbcc

**P**erformance-**p**ortability **b**enchmarking, analysis, plotting, and
**c**ode **c**omplexity for **C++** and **GPU-enriched C++**.

This repository is the tooling half of a performance-portability study of GPU
programming paradigms. The code under study — the same four algorithms (vector
addition, matrix multiplication, an n-body simulation, and a polyhedral gravity
model) implemented across CUDA, HIP, SYCL, Kokkos, RAJA, Alpaka, OpenMP,
OpenACC, OpenCL, Vulkan, Boost.Compute, WebGPU, Slang, Metal, and Stdpar —
lives in the companion repository
[**performance-portability-benchmark**](https://github.com/schuhmaj/performance-portability-benchmark),
together with the measurement results in its `results/` folder.

`ppbcc` provides the utilities on top of that: running the benchmarks,
computing application efficiency and performance portability, plotting the
results, and measuring code complexity. **The code-complexity analysis is
stand-alone** — it needs nothing but the source files you point it at, so it is
useful for any C++/GPU codebase.

📖 **[Documentation](https://schuhmaj.github.io/performance-portability-code-complexity/)**
 · 🖼️ **[Example plots](examples/)**

The package is split into five focused areas:

- `ppbcc.code_complexity` — Halstead and LOC/SLOC analysis with dialect-aware
  token counting (**stand-alone**).
- `ppbcc.benchmark` — discovers and runs Google Benchmark executables and
  consolidates their JSON reports into tidy CSV data.
- `ppbcc.profiling` — batch-runs a GPU profiler (Nsight Compute, Nsight
  Systems, Nsight Graphics, or LIKWID) over the same executables and turns the
  counters into a roofline model.
- `ppbcc.performance_portability` — loads benchmark CSVs and calculates
  application efficiency and performance portability.
- `ppbcc.plot` — Cascade, Navchart, combined, application-efficiency heatmap,
  boxplot, runtime bar plot, complexity comparison, and roofline visualizations.

The tool computes the classic Halstead measures (volume, difficulty, effort,
language level, ...) and line-based size metrics for a set of C++ source
files. Its distinguishing feature is that it understands the constructs of
GPU/parallel programming paradigms and counts them as **dialect operators**,
so you can quantify how much syntactic surface a paradigm adds on top of the
plain C++ baseline — the core question of performance-portability studies.

Supported dialects: **OpenMP, OpenACC, Kokkos, RAJA, Alpaka, CUDA, HIP,
Thrust, pCUDA ([AdaptiveCpp's portable CUDA dialect](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/pcuda.md)),
SYCL (AdaptiveCpp/DPC++), OpenCL (host API and `.cl` kernels), Vulkan,
Boost.Compute, WebGPU (host API and WGSL), GLSL compute shaders, Slang/HLSL,
Metal, stdpar** (`std::execution`).

## Credits

The performance-portability metrics and the Cascade/Navchart layouts are
inspired by the
[**P3 Analysis Library**](https://github.com/P3HPC/p3-analysis-library) by Pennycook et al.
If you use this tool, please also have a look at their performance-portability analysis in published work,
as this work builds upon theirs.

## Installation

```bash
pip install .            # or: pip install -e . for development
```

With conda:

```bash
conda env create -f environment.yaml
conda activate ppbcc
pip install -e .
```

Requires Python >= 3.12.

## Command line

Everything is reachable through the single `ppbcc` executable:

```bash
ppbcc code-complexity path/to/src -o complexity.csv
ppbcc benchmark -b path/to/build -p . -r '.*nbody.*' -H RTX5080 -o results
ppbcc profile -b path/to/build -p . -r '.*nbody.*' -H RTX5080 --roofline
ppbcc p2analysis cascade results.csv -n NBody -o nbody.pdf
ppbcc p3analysis navchart complexity.csv results.csv -n NBody
```

The package can also be run as a module: `python -m ppbcc code-complexity src/`.

> [!TIP]
> The tables below list the options you reach for most often, not every option.
> Run `ppbcc <command> --help` (or `-h`) for the exhaustive, always-current list.

### `ppbcc code-complexity`

Halstead complexity and LOC/SLOC metrics for C++ and GPU-enriched C++.

| Option | Meaning |
| --- | --- |
| `sources` | Files and/or directories (searched recursively for C++/kernel/shader extensions) |
| `-d, --dialect` | `auto` (default, per-file detection), `cpp` (baseline only), or dialect names such as `kokkos`, `cuda`, `kokkos,openmp` |
| `-m, --metrics` | Metric selection, e.g. `halstead`, `loc`, `sloc`, `dialect` (default: all) |
| `--diff` | Adds `baseline_*`/`delta_*` columns: metrics of the code with all dialect tokens removed, and the difference against that baseline |
| `--aggregate` | Appends a `TOTAL` row (operator/operand sets are merged before recomputing, so distinct counts are project-wide) |
| `--exclude-macro` | Regexes of macro names to disregard: invocations are removed and conditionals on them are resolved as if undefined, e.g. `'PPB_MARKER_\w+'` |
| `--exclude-header` | Globs of headers to disregard: they are not analysed and their `#include` lines are removed, e.g. `common/Marker.h` |
| `-o, --output` | CSV output path (`--csv-separator` changes the separator) |
| `--list-dialects` | List all known dialects and their aliases, then exit |

```bash
# Auto-detect the dialect of every file, print the table and save it as CSV
ppbcc code-complexity path/to/src -o report.csv

# Force a dialect, restrict the metrics, and report the plain-C++ delta
ppbcc code-complexity src/kokkos -d kokkos -m halstead loc --diff -o kokkos.csv
```

### `ppbcc benchmark`

Runs Google Benchmark targets and consolidates their JSON reports into one CSV.

| Option | Meaning |
| --- | --- |
| `-b, --build-dir` | Build folder used as the working directory for the whole pipeline |
| `-p, --path` | Directories searched recursively, relative to `--build-dir` |
| `-r, --regex` | **Required.** Pattern(s) matched against file paths to select executables (or reports) |
| `-x, --exclude` | Pattern(s) excluding matched paths, e.g. `'.*_cpp'` |
| `-s, --skip-benchmark` | Run nothing; locate existing JSON reports and only consolidate them |
| `-n, --dry-run` | List the matched executables (or reports) and exit |
| `-H, --hardware` | Value stored in the `Hardware` column, e.g. `"NVIDIA RTX5080"` |
| `-o, --output` | Base name of the consolidated CSV (defaults to a timestamped name) |

```bash
ppbcc benchmark -p src -H "NVIDIA RTX5080" -r "vec_.*" "nbody_.*" -x ".*_cpp" --dry-run
```

### `ppbcc profile`

Batch-profiles GPU kernels, consolidates the counters into one CSV, and draws a
roofline model. Expects executables built with `-DPPB_PROFILING=ON`.
`--profiler` picks the backend: `ncu` (default, one row per kernel launch),
`nsys`, `ngfx`, or `likwid` (one row per marked region, needs
`-DPPB_ENABLE_LIKWID=ON` too).

| Option | Meaning |
| --- | --- |
| `-b, --build-dir` | Build folder used as the working directory for the whole pipeline |
| `--profiler` | `ncu` (default), `nsys`, `ngfx` or `likwid` |
| `--profiler-path` | Path to that backend's CLI; found automatically otherwise |
| `-O, --option` | `NAME=VALUE` setting of the selected backend; `--help` lists the names |
| `--from-csv` | Skip profiling and plot from consolidated CSVs, merging several backends |
| `--timeout` | Wall-clock limit per executable; a run that hits it is skipped |
| `-p, --path` / `-r, --regex` / `-x, --exclude` | Executable discovery, exactly as for `ppbcc benchmark` (`--regex` is required) |
| `-d, --report-dir` | Where the profiler artefacts are written (default: `profiling`) |
| `-s, --skip-profile` | Run nothing; only re-parse the artefacts already in `--report-dir` |
| `-m, --memory-level` | Level the arithmetic intensity refers to: `dram` (default), `l2`, `l1` |
| `--precision` | `auto` (default, follows the build), `fp32`, `fp64`, `fp16` |
| `--region` | Plot only the regions matching a pattern, e.g. `--region evaluate` |
| `--all-kernels` | Keep the launches outside every named region (dropped by default) |
| `-a, --aggregate` | `sum` (default), `dominant`, or `none` — how launches become plot points |
| `--roofline [PATH]` | Render the roofline chart, optionally to a given path |
| `-l, --no-legend` | Leave the paradigm legend out of the roofline chart |
| `--no-csv` | Skip the consolidated CSV (for a pure collection run) |
| `--peak-performance` / `--peak-bandwidth` | Roofline ceilings for the backends without `peak_sustained` counters; `--peak-bandwidth` also converts the sampled percentages to bytes |
| `--analytic-flop` | `REGEX=FLOP` work model for `nsys`/`ngfx`, whose metric sets carry pipe utilisations rather than instruction counts |
| `-H, --hardware` / `-o, --output` | Hardware label and base name of the consolidated CSV |

```bash
ppbcc profile -b build-cuda-llvm-profiling -p src -r "polyhedral_.*" \
  -d profiling -H "NVIDIA RTX5080" -o Profiling_NVIDIA_RTX5080 --roofline

# The same run through the in-source markers instead
ppbcc profile --profiler likwid -b build-likwid -p src -r "polyhedral_.*" \
  -O lib="$LIKWID_PREFIX/lib" --peak-performance 5.74e13 --peak-bandwidth 9.59e11 \
  -d profiling-likwid -H "NVIDIA RTX5080" -o Profiling_LIKWID --roofline
```

```bash
# The paradigms without a CUDA context, and the merged roofline
ppbcc profile --profiler nsys -b build-cuda-llvm-profiling -p src -r "matMul_ocl$" "matMul_vulkan$" \
  -d profiling-nsys -O iterations=20 --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
  --analytic-flop "matMul_.*=137438953472" -H "NVIDIA RTX5080" -o Profiling_NSYS
ppbcc profile --from-csv Profiling_NVIDIA_RTX5080.csv Profiling_NSYS.csv -r "matMul_" \
  --roofline --no-csv -H "NVIDIA RTX5080"
```

Every row carries the NVTX region the launch happened in — `matmul`, or `init`
and `evaluate` — so a table holds the benchmark's own kernels and nothing else.
Build with `-DPPB_PROFILING=ON` (which turns on `PPB_ENABLE_NVTX`) for that;
without it the launches are unnamed and all of them are kept, with a warning.

> [!NOTE]
> Nsight Compute only sees CUDA kernels. OpenCL, Vulkan and host-only
> executables produce no report and are skipped with a warning; `--profiler
> nsys` is what measures them.

### `ppbcc p2analysis` and `ppbcc p3analysis`

Application efficiency, performance portability, and plots from benchmark CSVs.
`p2analysis PLOT CSV...` renders the charts that need benchmark results only
(`cascade`, `heatmap`, `double-heatmap`, `boxplot`, `time-barplot`); `p3analysis PLOT COMPLEXITY_CSV CSV...`
renders the charts that also need code complexity (`navchart`, `combined`,
`complexity-comparison`) plus `rank-correlation`, which writes a CSV table
instead of a figure.

| Option | Meaning |
| --- | --- |
| `PLOT` | Chart to render; the valid choices depend on the command |
| `COMPLEXITY_CSV` | `p3analysis` only: code-complexity CSV with one row per implementation |
| `CSV` | One or more CSVs produced by `ppbcc benchmark` |
| `-n, --name` | Benchmark problem; exact case-insensitive match preferred, unique substring accepted. Optional when the CSVs hold one problem; `rank-correlation` takes a comma-separated list and defaults to all of them |
| `-c, --complexity-metric` | `p3analysis` only: SLOC, Halstead vocabulary/length/volume/difficulty/effort (default: `halstead-difficulty`) |
| `--complexity-metric-absolute` | `p3analysis` only: plot absolute complexity instead of a percentage of plain C++ |
| `--compare-metric` | `complexity-comparison` only: x-axis metric (default: `sloc`) |
| `--correlation` | `rank-correlation` only: variable whose paradigm ordering is correlated — `pp` (default), `halstead-difficulty`, `sloc`, ... |
| `-s, --size` | `all` (default), an exact size, or `avg`/`best`/`worst` |
| `--average-over` | With `-s avg`: average Φ over sizes (`pp`, default) or compute Φ from size-averaged efficiencies (`efficiency`) |
| `--non-zero-pp` | Calculate Φ over supported platforms only |
| `-x, --exclude` / `-i, --include` | Regex filters on the `Description` column |
| `-t, --time` | `time-barplot` only: `wall-clock` (default), `kernel`, `force-update`, `neighbor-search` |
| `--normalize-time-to-peak` | `time-barplot` only: multiply runtimes by the platform's published peak FLOP/s |
| `--remove-description` | Drop bracketed labels such as `[Naive]`; efficiency charts combine variants by paradigm |
| `-l, --legend` | Save the legend as a separate PDF (`--legend--vertical` for one column) |
| `-e, --export-to-csv` | Also export efficiency and portability data as CSV |
| `-o, --output` | Output path; defaults to `<problem>_<chart>.pdf`, or `.csv` for `rank-correlation` |

```bash
ppbcc p3analysis combined ./code-complexity/code-complexity.csv ./Results_* \
  -n MatrixMultiplication -c halstead-difficulty --log-complexity \
  --non-zero-pp --remove-description -s avg -x "Cublas"

# Does one problem's paradigm ordering predict another's? Spearman's rho
# between every pair of problems, as a CSV matrix.
ppbcc p3analysis rank-correlation ./code-complexity/code-complexity.csv ./Results_* \
  --correlation pp --non-zero-pp -s avg -x "Cublas" -e
```

Not every option applies to every chart, and invalid combinations are rejected
rather than silently ignored: `-H/--hardware` is valid for `boxplot` and
`time-barplot` only, `time-barplot` needs one exact size (the largest by default), `boxplot`
needs `-s all` or an exact numeric size, `complexity-comparison` rejects
`--complexity-metric-absolute`, and `rank-correlation` — having no figure —
rejects `-l/--legend`, `--remove-description` and `--log-complexity`.
See [`examples/`](examples/) for one rendered example of every chart together
with the exact command that produced it.

## Python API

```python
from pathlib import Path
from ppbcc.code_complexity import evaluate

frame = evaluate(
    sources=[Path("src/")],            # files and/or directories
    language_dialect="auto",           # or "cpp", "kokkos", "cuda,openmp", ...
    metrics=["halstead", "loc"],       # None = all metrics
    diff=True,                         # add baseline_*/delta_* columns
    aggregate=False,                   # add a TOTAL row
    output=Path("report.csv"),         # optionally write the CSV directly
)
print(frame[["file", "dialect", "effort", "delta_effort"]])
```

The result is a `pandas.DataFrame` with one row per file. Columns include:

* `file`, `dialect` — the analysed file and the active dialect(s)
* `loc`, `sloc`, `comment_lines`, `blank_lines`
* Halstead: `distinct_operators` (n1), `distinct_operands` (n2),
  `total_operators` (N1), `total_operands` (N2), `vocabulary`, `length`,
  `calculated_length`, `volume`, `difficulty`, `effort`, `time_seconds`,
  `delivered_bugs`, `program_level`, `language_level`
* `dialect_distinct_operators`, `dialect_total_operators`,
  `dialect_distinct_operands`, `dialect_total_operands` — the tokens
  contributed by the GPU paradigm
* with `diff=True`: `baseline_<metric>` and `delta_<metric>` for the key
  Halstead columns

A worked example that analyses a whole benchmark suite implementation by
implementation is documented in the
[Code Complexity](https://schuhmaj.github.io/performance-portability-code-complexity/usage/code_complexity.html)
section.

## How tokens are counted

* The lexer strips comments and treats string/char literals, numbers and
  identifiers as operands; keywords, punctuation and preprocessor directives
  are operators. Closing brackets `)`, `]`, `}` pair with their (counted)
  opening bracket and are not counted again.
* Everything recognised as belonging to a dialect becomes a *dialect
  operator*: qualified names under a dialect namespace count as **one**
  operator (`Kokkos::parallel_for`), dialect keywords (`__global__`,
  `threadIdx`), API identifiers matched by patterns (`cudaMalloc`, `cl_mem`,
  `vkCreateBuffer`), CUDA/HIP kernel-launch brackets `<<< >>>`, and dialect
  pragmas (`#pragma omp ...`), whose clauses are operators and whose
  referenced variables are dialect operands.
* Namespace aliases are resolved (`namespace bc = boost::compute;` makes
  `bc::vector` a Boost.Compute operator).
* In `auto` mode dialects are detected per file via file extensions
  (`.cu`, `.hip`, `.cl`, `.comp`, `.wgsl`, `.slang`, ...), included headers,
  pragma prefixes, namespace usage and characteristic identifier patterns.

## Configuration

The keyword and dialect definitions live in TOML files packaged under
`src/ppbcc/code_complexity/share/` (`cpp_keywords.toml`, `dialects.toml`). They are
kept inside the package so that pip-installed wheels ship them; both can be
replaced at runtime with `--keywords-config` / `--dialects-config` (CLI) or
`keywords_path` / `dialects_path` (API). Adding a new dialect is a matter of
adding a new `[dialects.<name>]` table — no code changes required.

## Development

```bash
pip install -e ".[test]"
pytest
```

Building the documentation locally:

```bash
pip install -r docs/requirements.txt
cd docs && make html
```
