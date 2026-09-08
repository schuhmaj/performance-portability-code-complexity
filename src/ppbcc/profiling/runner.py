"""Batch execution of Nvidia Nsight Compute (``ncu``) over benchmark binaries."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from pathlib import Path

from loguru import logger

from ppbcc.profiling.metrics import ROOFLINE_METRICS

#: File name suffix Nsight Compute appends to ``--export``.
REPORT_SUFFIX = ".ncu-rep"


def aslr_prefix() -> list[str]:
    """Return the launcher that runs a child with ASLR disabled.

    Google Benchmark's ``main`` calls ``MaybeReenterWithoutASLR``, which
    ``execv``s the process to get reproducible timings. Nsight Compute attaches
    to the pre-exec process; with ``--target-processes all`` it then also
    attaches to the re-executed one, and on some binaries (``matMul_kokkos``,
    ``matMul_omp``) the profiler afterwards spins at 100 % CPU with the
    application suspended and never finishes. Starting the process with ASLR
    already off makes Google Benchmark skip the re-exec, and the same runs
    complete in seconds.

    Returns:
        The ``setarch ... -R`` argument vector, or an empty list if ``setarch``
        is unavailable (the run then simply keeps the re-exec).
    """
    setarch = shutil.which("setarch")
    if setarch is None:
        logger.warning(
            "setarch not found, so the benchmarks keep Google Benchmark's ASLR "
            "re-exec; Nsight Compute may hang on some of them."
        )
        return []
    return [setarch, platform.machine(), "-R"]


def find_ncu(explicit: str | None = None) -> str:
    """Locate the ``ncu`` executable.

    Args:
        explicit: User-supplied path, or ``None`` to search ``PATH``.

    Returns:
        Path to the profiler executable.

    Raises:
        FileNotFoundError: If no ``ncu`` executable can be found.
    """
    candidate = explicit or "ncu"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find the Nsight Compute CLI ({candidate!r}). Install the "
            "CUDA toolkit or pass --ncu with an explicit path."
        )
    logger.debug(f"Using profiler: {resolved}")
    return resolved


def profile_command(
    ncu: str,
    target: Path,
    report: Path,
    metrics: list[str] | None = None,
    benchmark_report: Path | None = None,
    extra_ncu_args: list[str] | None = None,
    extra_target_args: list[str] | None = None,
    disable_aslr: bool = True,
) -> list[str]:
    """Build the ``ncu`` command line for one benchmark executable.

    Args:
        ncu: Profiler executable.
        target: Benchmark binary to profile.
        report: Destination report; ``.ncu-rep`` is appended by the profiler.
        metrics: Metrics to collect, defaulting to :data:`ROOFLINE_METRICS`.
        benchmark_report: Where the target should write its Google-Benchmark
            JSON report. It carries the paradigm and precision of the run, which
            the profiler output itself does not know about.
        extra_ncu_args: Additional profiler arguments.
        extra_target_args: Additional arguments for the benchmark binary.
        disable_aslr: Whether to start the profiler through
            :func:`aslr_prefix`, which stops Google Benchmark from re-executing
            the process under the profiler.

    Returns:
        The argument vector to execute.
    """
    command = (aslr_prefix() if disable_aslr else []) + [
        ncu,
        # AdaptiveCpp, Kokkos and Google Benchmark itself may re-exec the
        # process, so the profiler has to follow child processes.
        "--target-processes",
        "all",
        # Every launch is reported with the NVTX range it happened under, which
        # is what tells the benchmark's own kernels from the runtime's setup
        # work; see src/common/Marker.h in the benchmark repository.
        "--nvtx",
        "--metrics",
        ",".join(metrics or ROOFLINE_METRICS),
        "--force-overwrite",
        "--export",
        str(report.with_suffix("")),
    ]
    command += extra_ncu_args or []
    command.append(str(target))
    if benchmark_report is not None:
        command += [
            f"--benchmark_out={benchmark_report}",
            "--benchmark_out_format=json",
        ]
    command += extra_target_args or []
    return command


def _run(
    command: list[str],
    name: str,
    stream_output: bool,
    timeout: float | None = None,
) -> None:
    """Execute one profiler run.

    Args:
        command: Argument vector to execute.
        name: Label used in log messages.
        stream_output: Whether to forward the profiler output to the logger at
            TRACE level instead of discarding it.
        timeout: Wall-clock limit in seconds, or ``None`` for no limit. A
            profiler that replays kernel launches can stop making progress
            without ever exiting, which without a limit blocks the whole batch.

    Raises:
        subprocess.CalledProcessError: If the profiler exits non-zero.
        subprocess.TimeoutExpired: If the run exceeds ``timeout``.
    """
    if not stream_output:
        subprocess.run(
            command,
            check=True,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    deadline = None if timeout is None else time.monotonic() + timeout
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            logger.trace(f"[{name}] {line.rstrip()}")
            if deadline is not None and time.monotonic() > deadline:
                process.kill()
                raise subprocess.TimeoutExpired(command, timeout)
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def run_profiles(
    executables: list[Path],
    output_dir: Path,
    ncu: str,
    force: bool = False,
    metrics: list[str] | None = None,
    extra_ncu_args: list[str] | None = None,
    extra_target_args: list[str] | None = None,
    stream_output: bool = False,
    timeout: float | None = None,
    disable_aslr: bool = True,
) -> list[Path]:
    """Profile every executable, one after another.

    Each run produces ``<output_dir>/<executable name>.ncu-rep`` next to the
    Google-Benchmark report ``<output_dir>/<executable name>.json``, so the
    profiling artefacts carry the same names as the binaries they came from.

    Args:
        executables: Benchmark binaries to profile.
        output_dir: Directory the reports are written to; created if missing.
        ncu: Profiler executable.
        force: Re-profile even when a report already exists.
        metrics: Metrics to collect, defaulting to :data:`ROOFLINE_METRICS`.
        extra_ncu_args: Additional profiler arguments.
        extra_target_args: Additional arguments for the benchmark binaries.
        stream_output: Whether to forward the profiler output to the logger.
        timeout: Wall-clock limit per executable in seconds, or ``None`` for no
            limit. A run that hits the limit is skipped, and the batch goes on.
        disable_aslr: Whether to launch through :func:`aslr_prefix`.

    Returns:
        The reports that exist after the batch, in execution order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for index, target in enumerate(executables, start=1):
        report = output_dir / f"{target.name}{REPORT_SUFFIX}"
        if report.exists() and not force:
            logger.warning(
                f"Report {report} already exists, reusing it (use --force to re-profile)"
            )
            reports.append(report)
            continue
        command = profile_command(
            ncu,
            target,
            report,
            metrics=metrics,
            benchmark_report=output_dir / f"{target.name}.json",
            extra_ncu_args=extra_ncu_args,
            extra_target_args=extra_target_args,
            disable_aslr=disable_aslr,
        )
        logger.info(f"[{index}/{len(executables)}] Profiling {target.name} ...")
        logger.debug(f"Command: {' '.join(command)}")
        try:
            _run(command, target.name, stream_output, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(
                f"Timed out after {timeout} s profiling {target.name}; skipping it."
            )
            continue
        except subprocess.CalledProcessError as error:
            logger.error(f"Failed to profile {target}: {error}")
            continue
        if not report.exists():
            # Nsight Compute only sees CUDA kernels, so OpenCL, Vulkan and
            # host-only paradigms legitimately end up without a report.
            logger.warning(
                f"{target.name} produced no report — it most likely launches no "
                "CUDA kernels (Nsight Compute sees no OpenCL, Vulkan or "
                "host-side work)"
            )
            continue
        logger.success(f"Finished {target.name} -> {report}")
        reports.append(report)
    return reports
