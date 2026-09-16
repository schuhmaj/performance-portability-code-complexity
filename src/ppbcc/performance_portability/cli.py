"""Command-line orchestration for P2 and P3 analysis."""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import seaborn as sns
from loguru import logger

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    BENCHMARK_PROBLEM,
    HARDWARE,
    PERFORMANCE_PORTABILITY,
    PRECISION,
    PROBLEM,
    PROBLEM_SIZE,
)
from ppbcc.performance_portability.complexity import (
    add_complexity_to_export,
    append_cpp_complexity_row,
    export_metrics_to_csv,
    load_complexity_baseline,
    load_complexity_data,
    load_complexity_export_data,
    merge_portability_complexity,
)
from ppbcc.performance_portability.metrics import (
    calculate_average_size_metrics,
    calculate_export_metrics,
    calculate_extreme_size_metrics,
    calculate_metrics,
    calculate_metrics_by_size,
    calculate_scaling_metrics,
)
from ppbcc.performance_portability.options import (
    P3_CHARTS,
    build_p2_parser,
    build_p3_parser,
)
from ppbcc.performance_portability.selection import (
    ALL_SIZE,
    AVERAGE_OVER_PP,
    AVERAGE_SIZE,
    BEST_SIZE,
    WORST_SIZE,
    _filter_problem_size,
    configure_logging,
    load_benchmark_csvs,
    select_problem_rows,
)
from ppbcc.plot.cascade import plot_cascade
from ppbcc.plot.complexity_comparison import plot_complexity_comparison
from ppbcc.plot.heatmap import (
    plot_efficiency_boxplot,
    plot_efficiency_heatmap,
)
from ppbcc.plot.navchart import plot_navchart
from ppbcc.plot.styles import (
    create_separate_legend,
    resolve_output_path,
    save_figure,
)


def p2analysis_main(argv: list[str] | None = None) -> int:
    """Run the P2 analysis command-line program (benchmark results only).

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status: zero on success, one on failure.
    """
    args = build_p2_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.hardware is not None and args.chart != "boxplot":
        logger.error("--hardware is only valid for boxplot charts.")
        return 1
    return _analyze(args)


def p3analysis_main(argv: list[str] | None = None) -> int:
    """Run the P3 analysis command-line program (benchmarks and complexity).

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status: zero on success, one on failure.
    """
    args = build_p3_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.chart == "complexity-comparison" and args.complexity_absolute:
        logger.error(
            "complexity-comparison compares two metrics on one shared scale "
            "relative to CPP, which --complexity-metric-absolute removes."
        )
        return 1
    return _analyze(args)


def _resolve_problem_query(benchmark_data: pd.DataFrame, name: str | None) -> str:
    """Return the problem query, defaulting to the only problem in the data.

    Args:
        benchmark_data: Combined benchmark results.
        name: User-supplied ``-n/--name`` query, or ``None``.

    Returns:
        The query to resolve against benchmark and complexity data.

    Raises:
        ValueError: If no name is given and the data holds several problems.
    """
    if name is not None:
        return name
    problems = sorted(
        {str(value) for value in benchmark_data[BENCHMARK_PROBLEM].dropna().unique()}
    )
    if len(problems) != 1:
        raise ValueError(
            "The benchmark CSVs contain several problems; select one with "
            f"-n/--name. Available problems: {problems}"
        )
    return problems[0]


def _analyze(args: argparse.Namespace) -> int:
    """Compute the metrics and render the chart selected on the command line.

    Args:
        args: Parsed arguments of ``p2analysis`` or ``p3analysis``. The
            complexity options are only read for the P3 charts, the hardware
            filter only for the boxplot.

    Returns:
        Process exit status: zero on success, one on failure.
    """
    sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
    uses_complexity = args.chart in P3_CHARTS

    try:
        if args.legend_vertical and not args.legend:
            raise ValueError("--legend--vertical requires -l/--legend.")
        if args.chart == "boxplot" and args.size in {
            AVERAGE_SIZE,
            BEST_SIZE,
            WORST_SIZE,
        }:
            raise ValueError(
                "boxplot requires --size all or an exact numeric size; "
                f"{args.size!r} collapses the efficiency distribution."
            )
        if args.average_over != AVERAGE_OVER_PP and args.size != AVERAGE_SIZE:
            raise ValueError(
                f"--average-over {args.average_over} requires --size avg; the "
                "other size modes do not average PP over sizes."
            )

        benchmark_data = load_benchmark_csvs(args.csv_files)
        efficiency_frames: list[pd.DataFrame] = []
        portability_frames: list[pd.DataFrame] = []
        export_efficiency_frames: list[pd.DataFrame] = []
        export_portability_frames: list[pd.DataFrame] = []
        scaling_frames: list[pd.DataFrame] = []
        boxplot_efficiency_frames: list[pd.DataFrame] = []
        problem_pairs: list[tuple[str, str]] = []

        for problem_query in (_resolve_problem_query(benchmark_data, args.name),):
            numeric_size = args.size if isinstance(args.size, float) else None
            selection_size = None if args.chart == "combined" else numeric_size
            all_size_rows, resolved_problem, description_is_workload = (
                select_problem_rows(
                    benchmark_data,
                    problem_query,
                    args.description_include,
                    args.description_exclude,
                    selection_size,
                    args.precision,
                )
            )
            problem_pairs.append((problem_query, resolved_problem))

            selected = all_size_rows
            if args.chart == "combined" and numeric_size is not None:
                selected = _filter_problem_size(
                    all_size_rows, numeric_size, resolved_problem
                )

            if args.size == AVERAGE_SIZE:
                problem_efficiency, problem_portability = (
                    calculate_average_size_metrics(
                        selected,
                        description_is_workload,
                        non_zero_pp=args.non_zero_pp,
                        average_over=args.average_over,
                    )
                )
            elif args.size in {BEST_SIZE, WORST_SIZE}:
                problem_efficiency, problem_portability = (
                    calculate_extreme_size_metrics(
                        selected,
                        description_is_workload,
                        args.size,
                        non_zero_pp=args.non_zero_pp,
                    )
                )
            else:
                problem_efficiency, problem_portability = calculate_metrics(
                    selected,
                    description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                )
            problem_efficiency.insert(0, PROBLEM, resolved_problem)
            problem_portability.insert(0, PROBLEM, resolved_problem)
            efficiency_frames.append(problem_efficiency)
            portability_frames.append(problem_portability)

            if args.chart == "boxplot":
                boxplot_rows = selected
                if args.hardware is not None:
                    hardware_labels = selected[HARDWARE].astype(str)
                    hardware_mask = hardware_labels.eq(args.hardware)
                    if not hardware_mask.any():
                        available_hardware = sorted(
                            hardware_labels.unique(), key=str.casefold
                        )
                        raise ValueError(
                            f"Hardware {args.hardware!r} has no selected rows for "
                            f"{resolved_problem}. Available hardware: "
                            f"{available_hardware}"
                        )
                    boxplot_rows = selected.loc[hardware_mask].copy()
                boxplot_efficiency, _ = calculate_metrics_by_size(
                    boxplot_rows,
                    description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                )
                boxplot_efficiency.insert(0, PROBLEM, resolved_problem)
                boxplot_efficiency_frames.append(boxplot_efficiency)

            if args.export_to_csv:
                export_rows, export_problem, export_description_is_workload = (
                    select_problem_rows(
                        benchmark_data,
                        problem_query,
                        args.description_include,
                        args.description_exclude,
                        None,
                        None,
                    )
                )
                export_efficiency, export_portability = calculate_export_metrics(
                    export_rows,
                    export_description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                    average_over=args.average_over,
                )
                export_efficiency.insert(0, PROBLEM, export_problem)
                export_portability.insert(0, PROBLEM, export_problem)
                export_efficiency_frames.append(export_efficiency)
                export_portability_frames.append(export_portability)
            if args.chart == "combined":
                problem_scaling = calculate_scaling_metrics(
                    all_size_rows,
                    description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                )
                problem_scaling.insert(0, PROBLEM, resolved_problem)
                scaling_frames.append(problem_scaling)

        efficiency = pd.concat(efficiency_frames, ignore_index=True)
        portability = pd.concat(portability_frames, ignore_index=True)
        problems = [resolved for _, resolved in problem_pairs]
        problem_title = " + ".join(problems)
        mode = args.chart
        output = resolve_output_path(args.output, problems, mode)

        navchart_data: pd.DataFrame | None = None
        metric: str | None = None
        export_efficiency: pd.DataFrame | None = None
        export_portability: pd.DataFrame | None = None
        if args.export_to_csv:
            export_efficiency = pd.concat(export_efficiency_frames, ignore_index=True)
            export_portability = pd.concat(export_portability_frames, ignore_index=True)

        if args.chart in {"navchart", "combined"}:
            navchart_frames: list[pd.DataFrame] = []
            for problem_query, resolved_problem in problem_pairs:
                complexity, current_metric = load_complexity_data(
                    args.complexity,
                    problem_query,
                    args.complexity_metric,
                    normalize=not args.complexity_absolute,
                )
                if metric is not None and current_metric != metric:
                    raise ValueError(
                        "Complexity metric labels differ between problems: "
                        f"{metric!r} and {current_metric!r}."
                    )
                metric = current_metric
                problem_pp = portability.loc[
                    portability[PROBLEM] == resolved_problem
                ].drop(columns=PROBLEM)
                problem_navchart = merge_portability_complexity(
                    problem_pp, complexity, current_metric
                )
                problem_navchart.insert(0, PROBLEM, resolved_problem)
                navchart_frames.append(problem_navchart)

            navchart_data = pd.concat(navchart_frames, ignore_index=True)
            logger.debug(f"Navchart data:\n{navchart_data.to_string(index=False)}")
            assert metric is not None
            if args.log_complexity and navchart_data[metric].le(0.0).any():
                raise ValueError(
                    "--log-complexity requires all plotted complexity values "
                    "to be positive."
                )

        comparison_data: pd.DataFrame | None = None
        comparison_labels: tuple[str, str] | None = None
        comparison_baselines: tuple[float, float] | None = None
        if args.chart == "complexity-comparison":
            comparison_frames: list[pd.DataFrame] = []
            baseline_sets: set[tuple[float, float]] = set()
            for problem_query, resolved_problem in problem_pairs:
                problem_pp = portability.loc[
                    portability[PROBLEM] == resolved_problem
                ].drop(columns=PROBLEM)
                merged: pd.DataFrame | None = None
                labels: list[str] = []
                for request in (args.compare_metric, args.complexity_metric):
                    # Both metrics are taken relative to the CPP baseline so
                    # they share one dimensionless scale and the identity line
                    # of the comparison chart is meaningful.
                    complexity, label = load_complexity_data(
                        args.complexity,
                        problem_query,
                        request,
                        normalize=True,
                    )
                    labels.append(label)
                    matched = merge_portability_complexity(
                        problem_pp, complexity, label
                    ).drop(columns=PERFORMANCE_PORTABILITY)
                    merged = (
                        matched
                        if merged is None
                        else merged.merge(matched, on=APPLICATION, how="inner")
                    )
                if labels[0] == labels[1]:
                    raise ValueError(
                        "complexity-comparison needs two different "
                        f"metrics; --compare-metric and --complexity-metric "
                        f"both resolve to {labels[0]!r}."
                    )
                assert merged is not None
                merged.insert(0, PROBLEM, resolved_problem)
                comparison_frames.append(merged)
                if comparison_labels is not None and comparison_labels != tuple(labels):
                    raise ValueError(
                        "Complexity metric labels differ between problems: "
                        f"{comparison_labels} and {tuple(labels)}."
                    )
                comparison_labels = (labels[0], labels[1])
                baseline_sets.add(
                    (
                        load_complexity_baseline(
                            args.complexity, problem_query, args.compare_metric
                        ),
                        load_complexity_baseline(
                            args.complexity,
                            problem_query,
                            args.complexity_metric,
                        ),
                    )
                )
            # Percentages of two different baselines cannot be keyed by one
            # box, so a multi-problem chart states no absolute values at all.
            if len(baseline_sets) == 1:
                comparison_baselines = baseline_sets.pop()
            elif baseline_sets:
                logger.info(
                    "Problems have different CPP baselines; the comparison "
                    "chart omits the absolute reference values."
                )
            comparison_data = pd.concat(comparison_frames, ignore_index=True)
            logger.debug(
                f"Comparison data:\n{comparison_data.to_string(index=False)}"
            )

        if args.export_to_csv and uses_complexity:
            assert export_efficiency is not None
            assert export_portability is not None
            enriched_efficiency_frames: list[pd.DataFrame] = []
            enriched_portability_frames: list[pd.DataFrame] = []
            for problem_query, resolved_problem in problem_pairs:
                export_complexity = load_complexity_export_data(
                    args.complexity, problem_query
                )
                portability_mask = export_portability[PROBLEM] == resolved_problem
                problem_export_portability = add_complexity_to_export(
                    export_portability.loc[portability_mask].copy(),
                    export_complexity,
                )
                efficiency_mask = export_efficiency[PROBLEM] == resolved_problem
                problem_export_efficiency = add_complexity_to_export(
                    export_efficiency.loc[efficiency_mask].copy(),
                    export_complexity,
                )
                enriched_portability_frames.append(
                    append_cpp_complexity_row(
                        problem_export_portability,
                        export_complexity,
                        resolved_problem,
                    )
                )
                enriched_efficiency_frames.append(
                    append_cpp_complexity_row(
                        problem_export_efficiency,
                        export_complexity,
                        resolved_problem,
                    )
                )
            export_efficiency = pd.concat(
                enriched_efficiency_frames, ignore_index=True, sort=False
            )
            export_portability = pd.concat(
                enriched_portability_frames, ignore_index=True, sort=False
            )

        if args.chart == "combined":
            assert navchart_data is not None
            assert metric is not None
            scaling_data = pd.concat(scaling_frames, ignore_index=True)
            figure = plot_cascade(
                efficiency,
                portability,
                problem_title,
                remove_description=args.remove_description,
                navchart_data=navchart_data,
                complexity_metric=metric,
                scaling_data=scaling_data,
                log_complexity=args.log_complexity,
                selected_size=None if args.size == ALL_SIZE else args.size,
                average_over=args.average_over,
                show_legends=not args.legend,
            )
        elif args.chart == "navchart":
            assert navchart_data is not None
            assert metric is not None
            figure = plot_navchart(
                navchart_data,
                metric,
                problem_title,
                remove_description=args.remove_description,
                log_complexity=args.log_complexity,
                show_legends=not args.legend,
            )
        elif args.chart == "complexity-comparison":
            assert comparison_data is not None
            assert comparison_labels is not None
            figure = plot_complexity_comparison(
                comparison_data,
                comparison_labels[0],
                comparison_labels[1],
                problem_title,
                remove_description=args.remove_description,
                log_axes=args.log_complexity,
                show_legends=not args.legend,
                baselines=comparison_baselines,
            )
        elif args.chart == "heatmap":
            figure = plot_efficiency_heatmap(
                efficiency,
                problem_title,
                remove_description=args.remove_description,
                selected_size=args.size,
            )
        elif args.chart == "boxplot":
            boxplot_efficiency = pd.concat(boxplot_efficiency_frames, ignore_index=True)
            figure = plot_efficiency_boxplot(
                boxplot_efficiency,
                problem_title,
                remove_description=args.remove_description,
                selected_size=args.size,
            )
        else:
            figure = plot_cascade(
                efficiency,
                portability,
                problem_title,
                remove_description=args.remove_description,
                selected_size=None if args.size == ALL_SIZE else args.size,
                average_over=args.average_over,
                show_legends=not args.legend,
            )

        save_figure(figure, output)
        if args.export_to_csv:
            assert export_efficiency is not None
            assert export_portability is not None
            export_identity_columns = {
                PROBLEM,
                APPLICATION,
                PROBLEM_SIZE,
                PRECISION,
                PERFORMANCE_PORTABILITY,
            }
            complexity_columns = [
                column
                for column in export_portability.columns
                if column not in export_identity_columns
            ]
            export_efficiency = export_efficiency[
                [
                    PROBLEM,
                    APPLICATION,
                    PROBLEM_SIZE,
                    PRECISION,
                    *complexity_columns,
                    HARDWARE,
                    APPLICATION_EFFICIENCY,
                ]
            ]
            export_portability = export_portability[
                [
                    PROBLEM,
                    APPLICATION,
                    PROBLEM_SIZE,
                    PRECISION,
                    *complexity_columns,
                    PERFORMANCE_PORTABILITY,
                ]
            ]
            export_metrics_to_csv(export_efficiency, export_portability, output)
        if args.legend and args.chart in {
            "cascade",
            "navchart",
            "combined",
            "complexity-comparison",
        }:
            legend_source = navchart_data if args.chart == "navchart" else portability
            assert legend_source is not None
            legend_applications = list(
                dict.fromkeys(legend_source[APPLICATION].astype(str))
            )
            legend_figure = create_separate_legend(
                legend_applications,
                problems,
                sorted(efficiency[HARDWARE].astype(str).unique()),
                remove_description=args.remove_description,
                vertical=args.legend_vertical,
            )
            legend_output = output.with_name(f"{output.stem}_legend.pdf")
            save_figure(legend_figure, legend_output)
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(p3analysis_main())
