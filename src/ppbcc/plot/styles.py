"""Shared styles, legends, and output helpers for P3 plots."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    NullFormatter,
    NullLocator,
)

from ppbcc.performance_portability.complexity import (
    _complexity_candidates,
    _normalize_token,
)

PP_SYMBOL = r"$\mathrm{ꟼ\!\!P}$"
PLOT_MARKER_SIZE = 10
LEGEND_MARKER_SIZE = 11
SCATTER_MARKER_AREA = 140

TAB20 = matplotlib.colormaps["tab20"].colors
FRAMEWORK_COLOR_MAP = {
    "CPP": TAB20[0],
    "Stdpar": TAB20[1],
    "Vulkan": TAB20[2],
    "Slang-Vulkan": TAB20[3],
    "AdaptiveCpp": TAB20[4],
    "Kokkos": TAB20[6],
    "RAJA": TAB20[10],
    "Alpaka": TAB20[12],
    "Cuda": TAB20[8],
    "Slang-Cuda": TAB20[9],
    "Cublas": TAB20[13],
    "Hip": "black",
    "OpenACC": TAB20[14],
    "OpenMP": TAB20[18],
    "OpenCL": TAB20[16],
    "Boost": TAB20[17],
}
HARDWARE_VENDOR_PATTERNS = {
    "NVIDIA": re.compile(
        r"(?i)\b(nvidia|geforce|quadro|tesla|rtx\d*|gtx\d*|gh\d+|h\d{2,3}|a\d{2,3})\b"
    ),
    "AMD": re.compile(r"(?i)\b(amd|radeon|instinct|mi\d+)\b"),
    "Intel": re.compile(r"(?i)\b(intel|xeon|arc|pvc)\b"),
}


def _slugify(value: str) -> str:
    """Convert a value into a compact filename component.

    Args:
        value: Arbitrary text.

    Returns:
        A filesystem-friendly lowercase slug.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "plot"


def resolve_output_path(
    output: Path | None,
    problems: list[str],
    mode: str,
) -> Path:
    """Resolve the plot output path and default PDF extension.

    Args:
        output: User-supplied path, or ``None``.
        problems: Resolved problem names.
        mode: Selected plot mode.

    Returns:
        Output path with a suffix.
    """
    problem_slug = _slugify("_".join(problems))
    path = output or Path(f"{problem_slug}_{_slugify(mode)}.pdf")
    if not path.suffix:
        path = path.with_suffix(".pdf")
    return path


def _hardware_vendor(hardware: str) -> str:
    """Classify a hardware label using vendor regular expressions.

    Args:
        hardware: Hardware label from the benchmark CSV.

    Returns:
        ``NVIDIA``, ``AMD``, ``Intel``, or ``Other``.
    """
    for vendor, pattern in HARDWARE_VENDOR_PATTERNS.items():
        if pattern.search(hardware):
            return vendor
    return "Other"


def _hardware_colors(platforms: list[str]) -> dict[str, object]:
    """Assign vendor-family color shades to hardware platforms.

    NVIDIA hardware uses green shades, AMD red shades, Intel blue shades, and
    unmatched labels neutral gray shades.

    Args:
        platforms: Hardware labels.

    Returns:
        Color mapping keyed by hardware label.
    """
    palette_names = {
        "NVIDIA": "Greens",
        "AMD": "Reds",
        "Intel": "Blues",
        "Other": "Greys",
    }
    colors: dict[str, object] = {}
    for vendor, palette_name in palette_names.items():
        matches = sorted(
            platform for platform in platforms if _hardware_vendor(platform) == vendor
        )
        shades = sns.color_palette(palette_name, n_colors=len(matches) + 2)[2:]
        colors.update(dict(zip(matches, shades)))
    return colors


def _framework_colors(
    applications: list[str], remove_description: bool = False
) -> dict[str, object]:
    """Assign stable framework colors based on the requested tab20 scheme.

    Args:
        applications: Application/framework labels.
        remove_description: Whether bracketed variants use their base framework
            color so the shortened legend remains unambiguous.

    Returns:
        Color mapping keyed by application label.
    """
    known = sorted(
        (
            (_normalize_token(framework), color)
            for framework, color in FRAMEWORK_COLOR_MAP.items()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    result: dict[str, object] = {}
    unmatched: list[str] = []
    for application in dict.fromkeys(applications):
        style_name = _display_application(application, remove_description)
        candidate_tokens = [
            _normalize_token(candidate)
            for candidate in _complexity_candidates(style_name)
        ]
        for framework_token, color in known:
            if any(framework_token in token for token in candidate_tokens):
                result[application] = color
                break
        else:
            unmatched.append(application)

    fallback = sns.color_palette("husl", n_colors=max(len(unmatched), 1))
    result.update(dict(zip(sorted(unmatched), fallback)))
    return result


def _problem_markers(problems: list[str]) -> dict[str, str]:
    """Assign the common dot marker to every benchmark problem.

    Args:
        problems: Resolved benchmark problem names.

    Returns:
        Marker mapping keyed by problem.
    """
    return {problem: "o" for problem in dict.fromkeys(problems)}


def _display_application(application: str, remove_description: bool) -> str:
    """Format an application label for a legend.

    Args:
        application: Full application label.
        remove_description: Whether to remove bracketed descriptions.

    Returns:
        Legend label.
    """
    if not remove_description:
        return application
    return re.sub(r"\[[^]]*]", "", application).strip()


def font_points(key: str, scale: float = 1.0) -> float:
    """Resolve a font-size rcParam to points and scale it.

    A size rcParam may be a number or one of Matplotlib's relative keywords
    such as ``"medium"``, which seaborn replaces with a number only once a
    theme is applied. Going through ``FontProperties`` handles both.

    Args:
        key: Font-size rcParam name, e.g. ``"axes.labelsize"``.
        scale: Factor applied to the resolved size.

    Returns:
        The scaled size in points.
    """
    size = FontProperties(size=matplotlib.rcParams[key]).get_size_in_points()
    return size * scale


#: Axis titles for the complexity metrics. The loader's own label ("Source
#: Lines of Code [normalized]") is too wide for a small panel and does not say
#: what "normalized" is relative to.
SHORT_METRIC_NAMES = {
    "sloc": "SLOC",
    "source lines of code": "SLOC",
    "halstead vocabulary": r"Halstead $\eta$",
    "halstead program length": "Halstead $N$",
    "halstead volume": "Halstead $V$",
    "halstead difficulty": "Halstead $D$",
    "halstead effort": "Halstead $E$",
}
#: How each scaling mode of the complexity loader is spelled out on an axis.
SCALING_SUFFIXES = {
    "normalized": "[% of sequential C++]",
    "additive": "[added over sequential C++]",
    "absolute": "[absolute]",
}


def short_metric_label(metric: str) -> str:
    """Turn a complexity column name into a compact axis title.

    Args:
        metric: Column name from the complexity loader, for example
            ``"Halstead Difficulty [normalized]"``.

    Returns:
        A short title such as ``"Halstead $D$ [% of sequential C++]"``. A metric
        or scaling mode that is not recognised is returned unchanged, so the
        caller never loses information it cannot re-derive.
    """
    name, _, bracket = metric.partition("[")
    short = SHORT_METRIC_NAMES.get(name.strip().casefold())
    if short is None:
        return metric
    if not bracket:
        return short
    suffix = SCALING_SUFFIXES.get(bracket.rstrip("]").strip().casefold())
    return metric if suffix is None else f"{short} {suffix}"


#: Mantissa sets tried when labelling a logarithmic complexity axis, coarse to
#: fine. Normalized complexity spans well under one decade, so the decade
#: boundaries Matplotlib would use on their own can miss the data entirely.
_LOG_TICK_STEPS = (
    (1.0, 2.0, 5.0),
    (1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
    (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0),
    (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0),
)
_LOG_TICK_TARGET = 4


def format_relative_log_axis(axis) -> None:
    """Label a logarithmic complexity axis with plain decimal ticks.

    Matplotlib's default log locator labels a range such as 100-700 as
    ``10^2``, ``4x10^2``, ``6x10^2``, which is unreadable at figure scale, and
    over a range such as 110-190 it places no major tick at all. Round values
    inside the limits are used instead, choosing the mantissa set whose tick
    count comes closest to _LOG_TICK_TARGET so that both a narrow and a wide
    range stay readable. If no set yields at least two ticks the axis is left
    alone rather than stripped of its labels.

    Args:
        axis: The x or y axis to format.
    """
    low, high = sorted(axis.get_view_interval())
    if not (low > 0.0 and high > low):
        return

    def ticks_for(steps: tuple[float, ...]) -> list[float]:
        values: list[float] = []
        decade = math.floor(math.log10(low))
        while decade <= math.ceil(math.log10(high)):
            values.extend(step * 10.0**decade for step in steps)
            decade += 1
        return [value for value in values if low <= value <= high]

    candidates = [ticks_for(steps) for steps in _LOG_TICK_STEPS]
    usable = [found for found in candidates if len(found) >= 2]
    if not usable:
        return
    best = min(usable, key=lambda found: abs(len(found) - _LOG_TICK_TARGET))
    axis.set_major_locator(FixedLocator(best))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def use_inward_ticks(axes: plt.Axes) -> None:
    """Keep a right-hand y axis from reading as negative numbers.

    With Matplotlib's default outward ticks a right-hand axis draws the tick
    mark between the spine and its label, so ``1.0`` reads as ``-1.0``. Turning
    the ticks inward is not enough on its own under seaborn's whitegrid style,
    whose spine is too faint to separate them: the labels also move further out
    and the spine is darkened, so the axis line is unmistakably between the
    tick and the number.

    Args:
        axes: Axes whose right-hand y axis is corrected.
    """
    axes.yaxis.set_tick_params(direction="in", length=3.5, pad=8)
    spine = axes.spines.get("right")
    if spine is not None:
        spine.set_visible(True)
        spine.set_color("0.35")
        spine.set_linewidth(1.0)


def _application_legend_handles(
    applications: list[str],
    colors: dict[str, object],
    remove_description: bool,
) -> list[Line2D]:
    """Create framework-color legend handles.

    Args:
        applications: Ordered application labels.
        colors: Application color mapping.
        remove_description: Whether to hide bracketed descriptions.

    Returns:
        Matplotlib legend handles.
    """
    handles: list[Line2D] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    ordered_applications = sorted(
        dict.fromkeys(applications),
        key=lambda application: _display_application(
            application, remove_description
        ).casefold(),
    )
    for application in ordered_applications:
        label = _display_application(application, remove_description)
        color = tuple(matplotlib.colors.to_rgba(colors[application]))
        key = (label, color)
        if key in seen:
            continue
        seen.add(key)
        handles.append(Line2D([0], [0], color=color, linewidth=3.0, label=label))
    return handles


def _problem_legend_handles(
    problems: list[str], markers: dict[str, str]
) -> list[Line2D]:
    """Create problem-marker legend handles.

    Args:
        problems: Ordered problem names.
        markers: Problem marker mapping.

    Returns:
        Matplotlib legend handles.
    """
    return [
        Line2D(
            [0],
            [0],
            color="black",
            marker=markers[problem],
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            markeredgewidth=1.5,
            label=problem,
        )
        for problem in dict.fromkeys(problems)
    ]


def create_separate_legend(
    applications: list[str],
    problems: list[str],
    platforms: list[str],
    remove_description: bool = False,
    vertical: bool = False,
) -> plt.Figure:
    """Create a standalone paradigm/device legend.

    Paradigms are placed at the top and devices at the bottom. A problem-marker
    section is included between them when more than one problem is plotted. By
    default each section is a separate four-column legend; vertical mode
    combines the sections into one single-column legend with subheadings.

    Args:
        applications: Application labels in display order.
        problems: Problem names in display order.
        platforms: Hardware labels in display order.
        remove_description: Whether bracketed descriptions are hidden.
        vertical: Whether legend entries are arranged in one column.

    Returns:
        A standalone legend figure.
    """
    colors = _framework_colors(applications, remove_description)
    markers = _problem_markers(problems)
    platform_colors = _hardware_colors(platforms)
    platform_labels = {
        platform: chr(ord("A") + index) if index < 26 else str(index + 1)
        for index, platform in enumerate(platforms)
    }
    paradigm_handles = _application_legend_handles(
        applications, colors, remove_description
    )
    device_handles = [
        Patch(
            facecolor=platform_colors[name],
            edgecolor="black",
            label=f"{platform_labels[name]}: {name}",
        )
        for name in platforms
    ]

    if vertical:
        sections = [("Paradigm", paradigm_handles)]
        if len(problems) > 1:
            sections.append(
                (
                    "Problem",
                    _problem_legend_handles(problems, markers),
                )
            )
        sections.append(("Device", device_handles))

        grouped_handles = []
        heading_indexes = []
        for heading, handles in sections:
            heading_indexes.append(len(grouped_handles))
            grouped_handles.append(
                Line2D([], [], linestyle="None", marker=None, label=heading)
            )
            grouped_handles.extend(handles)

        height = max(2.0, 0.42 * len(grouped_handles) + 0.5)
        figure = plt.figure(figsize=(6.5, height))
        legend = figure.legend(
            handles=grouped_handles,
            loc="center",
            ncol=1,
            frameon=True,
            labelspacing=0.45,
        )
        legend_texts = legend.get_texts()
        for index in heading_indexes:
            legend_texts[index].set_fontweight("bold")
        return figure

    paradigm_rows = max(1, (len(paradigm_handles) + 3) // 4)
    device_rows = max(1, (len(device_handles) + 3) // 4)
    problem_rows = 1 if len(problems) > 1 else 0
    height = 1.0 + 0.55 * (paradigm_rows + device_rows + problem_rows)
    figure = plt.figure(figsize=(13.0, height))
    figure.legend(
        handles=paradigm_handles,
        title="Paradigm",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
    )
    if len(problems) > 1:
        figure.legend(
            handles=_problem_legend_handles(problems, markers),
            title="Problem",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=4,
            frameon=True,
        )
    figure.legend(
        handles=device_handles,
        title="Device",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
    )
    return figure


def save_figure(figure: plt.Figure, output: Path) -> None:
    """Save a figure, creating its parent directory if necessary.

    Args:
        figure: Figure to save.
        output: Destination path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    logger.success(f"Wrote plot: {output.resolve()}")
