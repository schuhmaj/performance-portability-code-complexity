"""Command-line orchestration for P3 analysis."""

from __future__ import annotations

import sys

import pandas as pd
import seaborn as sns
from loguru import logger

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
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
from ppbcc.performance_portability.options import build_parser
from ppbcc.performance_portability.selection import (
    ALL_SIZE,
    AVERAGE_SIZE,
    BEST_SIZE,
    WORST_SIZE,
    _filter_problem_size,
    configure_logging,
    load_benchmark_csvs,
    select_problem_rows,
)
from ppbcc.plot.cascade import plot_cascade
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


def main(argv: list[str] | None = None) -> int:
    """Run the P3 analysis command-line program.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status: zero on success, one on failure.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

    try:
        if args.chart in {"navchart", "combined"} and args.complexity is None:
            raise ValueError(f"--chart {args.chart} requires --complexity.")
        if args.chart == "boxplot" and args.size in {
            AVERAGE_SIZE,
            BEST_SIZE,
            WORST_SIZE,
        }:
            raise ValueError(
                "--chart boxplot requires --size all or an exact numeric size; "
                f"{args.size!r} collapses the efficiency distribution."
            )
        if (args.normalize or args.additive) and (
            args.complexity is None or args.chart not in {"navchart", "combined"}
        ):
            raise ValueError(
                "--normalize/--additive are only valid for navchart/combined "
                "with --complexity."
            )
        if args.log_complexity and args.chart not in {"navchart", "combined"}:
            raise ValueError(
                "--log-complexity is only valid for navchart/combined charts."
            )
        if args.log_size and args.chart != "combined":
            raise ValueError("--log-size is only valid for combined charts.")
        if (
            args.chart not in {"navchart", "combined"}
            and args.complexity is not None
            and not args.export_to_csv
        ):
            logger.warning(f"Ignoring --complexity for --chart {args.chart}.")

        benchmark_data = load_benchmark_csvs(args.csv_files)
        efficiency_frames: list[pd.DataFrame] = []
        portability_frames: list[pd.DataFrame] = []
        export_efficiency_frames: list[pd.DataFrame] = []
        export_portability_frames: list[pd.DataFrame] = []
        scaling_frames: list[pd.DataFrame] = []
        boxplot_efficiency_frames: list[pd.DataFrame] = []
        problem_pairs: list[tuple[str, str]] = []

        for problem_query in (args.name,):
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
                boxplot_efficiency, _ = calculate_metrics_by_size(
                    selected,
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
            assert args.complexity is not None
            navchart_frames: list[pd.DataFrame] = []
            for problem_query, resolved_problem in problem_pairs:
                complexity, current_metric = load_complexity_data(
                    args.complexity,
                    problem_query,
                    args.complexity_metric,
                    args.normalize,
                    args.additive,
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

        if args.export_to_csv and args.complexity is not None:
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
            if args.log_size and scaling_data[PROBLEM_SIZE].le(0.0).any():
                raise ValueError(
                    "--log-size requires all plotted problem sizes to be positive."
                )
            figure = plot_cascade(
                efficiency,
                portability,
                problem_title,
                remove_description=args.remove_description,
                navchart_data=navchart_data,
                complexity_metric=metric,
                scaling_data=scaling_data,
                log_complexity=args.log_complexity,
                log_size=args.log_size,
                selected_size=None if args.size == ALL_SIZE else args.size,
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
        elif args.chart == "heatmap":
            figure = plot_efficiency_heatmap(
                efficiency,
                problem_title,
                remove_description=args.remove_description,
                selected_size=args.size,
            )
        elif args.chart == "boxplot":
            boxplot_efficiency = pd.concat(boxplot_efficiency_frames, ignore_index=True)
            application_order = portability.sort_values(
                PERFORMANCE_PORTABILITY, ascending=False
            )[APPLICATION].astype(str)
            figure = plot_efficiency_boxplot(
                boxplot_efficiency,
                problem_title,
                remove_description=args.remove_description,
                selected_size=args.size,
                application_order=application_order,
            )
        else:
            figure = plot_cascade(
                efficiency,
                portability,
                problem_title,
                remove_description=args.remove_description,
                selected_size=None if args.size == ALL_SIZE else args.size,
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
        if args.legend and args.chart in {"cascade", "navchart", "combined"}:
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
            )
            legend_output = output.with_name(f"{output.stem}_legend.pdf")
            save_figure(legend_figure, legend_output)
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
