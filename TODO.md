# TODO

## `ppbcc profile --skip-profile`: `--regex` matches different strings per backend

*Found 2026-09-13 while re-consolidating the polyhedral Nsight Systems reports of
performance-portability-benchmark after the analytic FLOP constant changed.*

### Problem

Without `--skip-profile`, `-r/--regex` (and `-x/--exclude`) select **executables**: `find_files()`
(`src/ppbcc/benchmark/runner.py`) matches every pattern with `re.search` against the full path of each
executable below `-p`, e.g. `…/src/polyhedralGravity/opencl/polyhedral_ocl`. Because the path ends in the
executable name, `$`-anchored patterns such as `"polyhedral_ocl$"` work, and the documented profiling
commands use them throughout.

With `--skip-profile`, the same options select **report artefacts** instead
(`src/ppbcc/profiling/cli.py`, discovery block), and each backend exposes a different string to the patterns:

| Backend | Artefacts returned by discovery | String `-r` is matched against | `"<executable>$"` |
|---|---|---|---|
| `likwid` | `likwid.find_profiled()`: name stubs, marker suffix stripped | `polyhedral_acc` | matches (from the code, not run) |
| `ngfx` | `ngfx.find_profiled()`: trace directories named after the executable | `polyhedral_vulkan` | matches |
| `nsys` | `nsys.find_profiled()`: SQLite exports, `.ref.sqlite` filtered out | `polyhedral_ocl.sqlite` (`path.name`) | **no match** |
| `ncu` | `reuse_reports` path: `find_files([report_dir], ...)` over `*.ncu-rep` | `…/profiling-rtx5080/polyhedral_omp.ncu-rep` (full path) | **no match** |

Reproduced with `--dry-run` in performance-portability-benchmark:

```bash
ppbcc profile --profiler ncu  -b . -d profiling-rtx5080      -r 'polyhedral_omp$'    --skip-profile --dry-run -H x  # 0 reports
ppbcc profile --profiler ncu  -b . -d profiling-rtx5080      -r 'polyhedral_omp'     --skip-profile --dry-run -H x  # 1 report
ppbcc profile --profiler nsys -b . -d profiling-nsys-rtx5080 -r 'polyhedral_ocl$'    --skip-profile --dry-run -H x  # 0
ppbcc profile --profiler nsys -b . -d profiling-nsys-rtx5080 -r 'polyhedral_ocl'     --skip-profile --dry-run -H x  # 1
ppbcc profile --profiler ngfx -b . -d profiling-ngfx-rtx5080 -r 'polyhedral_vulkan$' --skip-profile --dry-run -H x  # 1
```

### Consequences

- The natural workflow fails: copy the `-r` list of a profiling command, add `--skip-profile`, get
  `ERROR | Nothing matched the given path/regex.` (exit 1) for `nsys` and `ncu`, but not for `ngfx`/`likwid`.
- The error gives no hint that the patterns are now applied to report names.
- Dropping the anchor is not a clean workaround: `"polyhedral_ocl"` also matches a hypothetical
  `polyhedral_ocl_shared`. The benchmark repo currently documents `"^polyhedral_ocl(\.|$)"` for `nsys`
  (`results/ROOFLINE.md`, step 2) and consolidates `ncu` with `-r ".*"`.
- For `ncu`, patterns also see the report *directory*, so a pattern can match on the directory name
  (`-r "rtx5080"` selects every report) — different again from the other three backends.
- The `--regex` help says "matched against file paths to select executables (or reports with
  `--skip-profile`)", which is accurate for none of the backends in detail.

### Options

1. **Match against the executable name in every mode (recommended).**
   - Give each backend an `executable_name(path) -> str` that maps an artefact to the executable it came
     from: strip `.sqlite` (`nsys`), `.ncu-rep` (`ncu`); `ngfx` directory names and `likwid` stubs are
     already bare names.
   - In `cli.py`, apply `-r`/`-x` to that name for all four backends under `--skip-profile`, and drop the
     special `reuse_reports` double `find_files` call for `ncu` in favour of an `ncu.find_profiled()` like
     the other backends.
   - Keep the profiling mode as it is (full executable path), or also switch it to the executable name for
     full symmetry; the latter would stop patterns like `"opencl/"` from selecting by directory.
   - Update the `--regex`/`--exclude` help and the docs (`docs/usage/profiling.rst`, README table).
   - Tests: for each backend, the `$`-anchored patterns of a profiling command select the same set of
     executables when re-reading with `--skip-profile`.
   - Breaking change: anyone relying on `-r` matching the report directory part of an `ncu` path, or the
     `.sqlite` suffix of an `nsys` export, gets different matches. Unlikely to matter.

2. **Match against the artefact path consistently.** Make all four backends match against the full artefact
   path (as `ncu` does today) and document that `--skip-profile` patterns select report files. Keeps one
   rule, but the profiling command still cannot be reused verbatim, and `ngfx`/`likwid` behaviour changes.

3. **Try both.** Keep today's strings, but also accept an artefact when a pattern matches its executable
   name. Least disruptive and makes the anchored patterns work, at the cost of a matching rule that is hard
   to explain (a pattern can now select through either string).

4. **Documentation and diagnostics only.** Leave the matching alone; state the per-backend strings in the
   `--regex` help and docs, and when `--skip-profile` finds nothing but the artefacts' executable names
   would have matched, say so in the error ("patterns are matched against `polyhedral_ocl.sqlite`; drop the
   `$` anchor or use `(\.|$)`"). Cheapest, but keeps the trap.

Whichever option is chosen, remove the `(\.|$)` workaround from the benchmark repo's `results/ROOFLINE.md`
afterwards if it is no longer needed.

### Related, smaller

- `--dry-run` with `--skip-profile --profiler nsys` prints "Found N executable(s)" although it lists SQLite
  reports; `ncu` correctly says "report(s)".
