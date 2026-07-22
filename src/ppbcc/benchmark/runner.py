"""Discover and execute Google Benchmark targets."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from loguru import logger


def find_files(
    search_dirs: list[Path],
    patterns: list[str],
    require_executable: bool = False,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Recursively search ``search_dirs`` for files whose path matches any of the
    given regular expressions.

    Args:
        search_dirs: Directories to search recursively.
        patterns: Regular expression patterns. A file is kept if any pattern
            matches (``re.search``) against its path.
        require_executable: If True, only keep files with the executable bit set
            (used to locate benchmark binaries).
        exclude: Regular expression patterns. A file is dropped if any of these
            matches (``re.search``) against its path, even if it matched a
            include pattern (e.g. ``".*cpp.*"``).

    Returns:
        A sorted, de-duplicated list of resolved matching file paths.
    """
    compiled = [re.compile(p) for p in patterns]
    excluded = [re.compile(p) for p in (exclude or [])]
    kind = "executables" if require_executable else "files"
    logger.info(f"Searching {search_dirs} for {kind} matching: {patterns}")
    if excluded:
        logger.info(f"Excluding {kind} matching: {exclude}")

    matches: set[Path] = set()
    for directory in search_dirs:
        if not directory.exists():
            logger.warning(f"Search path {directory} does not exist, skipping")
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if require_executable and not os.access(path, os.X_OK):
                continue
            path_str = str(path)
            if not any(rx.search(path_str) for rx in compiled):
                continue
            if any(rx.search(path_str) for rx in excluded):
                logger.debug(f"Excluded {kind[:-1]}: {path}")
                continue
            resolved = path.resolve()
            if resolved not in matches:
                logger.debug(f"Matched {kind[:-1]}: {path}")
                matches.add(resolved)

    result = sorted(matches)
    logger.info(f"Found {len(result)} matching {kind}")
    return result


def print_found_files(files: list[Path], base: Path, are_reports: bool) -> None:
    """Pretty-print the discovered files as a boxed, numbered table (dry run)."""
    kind = "report(s)" if are_reports else "executable(s)"
    rows = []
    for i, path in enumerate(files, start=1):
        try:
            shown = path.relative_to(base)
        except ValueError:
            shown = path
        rows.append((str(i), str(shown)))

    title = f" Found {len(files)} {kind} "
    idx_w = max((len(r[0]) for r in rows), default=1)
    path_w = max((len(r[1]) for r in rows), default=0)
    inner = max(idx_w + path_w + 3, len(title))

    top = f"┌{'─' * inner}┐"
    sep = f"├{'─' * inner}┤"
    bottom = f"└{'─' * inner}┘"

    print(top)
    print(f"│{title.center(inner)}│")
    print(sep)
    if rows:
        for num, shown in rows:
            line = f" {num.rjust(idx_w)}  {shown.ljust(path_w)} "
            print(f"│{line.ljust(inner)}│")
    else:
        print(f"│{' (nothing matched) '.center(inner)}│")
    print(bottom)


def _run_target(target: Path, output_file: Path, stream_output: bool) -> None:
    """Run a single benchmark executable.

    Args:
        target: The benchmark binary to execute.
        output_file: The Google-Benchmark JSON report to write.
        stream_output: If True, forward the executable's stdout/stderr to the
            logger (at TRACE level), line by line, as it runs. Otherwise the
            output is discarded.

    Raises:
        subprocess.CalledProcessError: If the executable exits with a non-zero
            status.
    """
    cmd = [
        str(target),
        f"--benchmark_out={output_file.name}",
        "--benchmark_out_format=json",
    ]
    if not stream_output:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return

    # Stream the combined stdout/stderr through the logger so the executable's
    # progress is visible at the highest verbosity (-vv / TRACE).
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.trace(f"[{target.name}] {line.rstrip()}")
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_benchmarks(
    executables: list[Path],
    force: bool = False,
    json_prefix: str = "",
    stream_output: bool = False,
) -> list[Path]:
    """Run each benchmark executable, producing a ``<prefix><name>.json``
    Google-Benchmark report in the current working directory.

    Args:
        executables: Benchmark binaries to run.
        force: Re-run even if a report already exists (otherwise it is reused).
        json_prefix: String prepended to each report file name (e.g. ``"intel_"``).
        stream_output: If True, forward each executable's stdout/stderr to the
            logger at TRACE level (enabled at ``-vv``); otherwise it is discarded.

    Returns:
        The list of report files that exist after the run.
    """
    report_files: list[Path] = []
    for target in executables:
        output_file = Path(f"{json_prefix}{target.name}.json")
        if output_file.exists() and not force:
            logger.warning(
                f"Report {output_file} already exists, reusing it (use --force to re-run)"
            )
            report_files.append(output_file)
            continue
        try:
            logger.info(f"Benchmarking {target.name} ...")
            _run_target(target, output_file, stream_output)
            logger.success(f"Finished {target.name} -> {output_file}")
            report_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute {target}: {e}")
    return report_files


# --------------------------------------------------------------------------- #
# Loading / normalization
# --------------------------------------------------------------------------- #
