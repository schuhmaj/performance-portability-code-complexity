# ppbcc

Performance-portability benchmarking, P3 analysis/plotting, and Halstead
complexity analysis for **C++** and **GPU-enriched C++**.

The package is split into four focused areas:

- `ppbcc.benchmark` discovers and runs Google Benchmark executables and
  consolidates their JSON reports into tidy CSV data.
- `ppbcc.performance_portability` loads benchmark CSVs and calculates
  application efficiency and performance portability.
- `ppbcc.plot` contains Cascade, Navchart, combined, application-efficiency
  heatmap, and boxplot visualizations.
- `ppbcc.code_complexity` provides the original Halstead and LOC/SLOC analysis.

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

## Unified command line

The `ppbcc` executable exposes all three workflows as subcommands:

```bash
ppbcc code-complexity path/to/src -o complexity.csv
ppbcc benchmark -b path/to/build -p . -r '.*nbody.*' -H RTX5080 -o results
ppbcc p3analysis NBody results.csv --chart cascade -o nbody.pdf
ppbcc p3analysis NBody results.csv --chart heatmap --size all -o efficiency.pdf
ppbcc p3analysis NBody results.csv --chart boxplot --size 1048576 \
  --hardware "NVIDIA H100" -o spread.pdf
```

Each workflow also has standalone compatibility and prefixed executables:

| Workflow | Executables |
| --- | --- |
| Code complexity | `code-complexity`, `ppbcc-code-complexity` |
| Benchmark | `benchmark`, `ppbcc-benchmark` |
| P3 analysis | `p3analysis`, `ppbcc-p3analysis` |

Run any command with `--help` for its complete options. The benchmark command
supports `--skip-benchmark` to consolidate existing JSON reports. P3 analysis
supports Cascade, Navchart, combined, heatmap, and boxplot charts, plus CSV
export of application-efficiency and performance-portability data. The
heatmap places paradigms on the x-axis and platforms on the y-axis. The
boxplot shows each paradigm's application-efficiency distribution across
platforms and sizes, with paradigms sorted alphabetically. Pass
`-H/--hardware` to restrict a boxplot to one hardware platform; this option is
an error for every other chart type.

P3 analysis uses `--size all` by default. An exact numeric size restricts all
charts to that size. Cascade, Navchart, combined, and heatmap charts also
accept `average`/`mean`, `best`, and `worst`; boxplots require `all` or an
exact numeric size because summary modes remove the underlying distribution.
Use `-l/--legend` to save the chart legend as a separate PDF, and add
`--legend--vertical` to arrange its device and paradigm entries in one column.

## Command line usage

```bash
# Analyse a directory recursively, auto-detect the dialect of every file,
# print the table and save it as CSV:
code-complexity path/to/src -o report.csv

# Equivalent grouped module invocation:
python -m ppbcc code-complexity path/to/src -o report.csv

# Force a dialect (several may be combined), restrict the metrics, and
# additionally report the plain-C++ baseline and the paradigm's delta:
code-complexity src/kokkos -d kokkos -m halstead loc --diff -o kokkos.csv

# Aggregate all files into an additional TOTAL row:
code-complexity src/cuda -d cuda --aggregate

# More logging: -v (DEBUG), -vv (TRACE); default level is INFO.
code-complexity src -vv

# List all known dialects and their aliases:
code-complexity --list-dialects
```

Important options:

| Option | Meaning |
| --- | --- |
| `sources` | Files and/or directories (searched recursively for C++/kernel/shader extensions) |
| `-d, --dialect` | `auto` (default, per-file detection), `cpp` (baseline only), or dialect names such as `kokkos`, `cuda`, `opencl`, `kokkos,openmp` |
| `-m, --metrics` | Metric selection, e.g. `halstead`, `loc`, `sloc`, `halstead_effort`, `halstead_volume`, `dialect` (default: all) |
| `--diff` | Adds `baseline_*` / `delta_*` columns: metrics of the code with all dialect tokens removed, and the difference of the full metrics against that baseline |
| `--aggregate` | Appends a `TOTAL` row (operator/operand sets are merged before recomputing, so distinct counts are project-wide) |
| `-o, --output` | CSV output path (`--csv-separator` changes the separator) |
| `--keywords-config`, `--dialects-config` | Override the packaged TOML configuration |

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
pytest            # runs the suite in ./test
```
