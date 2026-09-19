"""The two charts of a report, as PNG files.

`pareto.png` shows quality against the p95 of the time to the first answer
token. `latency.png` shows the p50 and the p95 of each mode on one axis.

Colour does one job in each chart. In the Pareto chart, blue is a mode on the
front and grey is a dominated mode; grey is a deliberate de-emphasis, and the
front line, the legend and the direct labels carry the same fact, so colour is
never the only channel. In the latency chart, blue and orange are the first
two slots of a palette that was validated for colour-vision deficiency. Text
is always ink, never a series colour. The time axis is logarithmic, because a
timeout of 60 s and an answer in 0.5 s must be readable on the same chart.

The figures are made with the object interface and the Agg canvas. No global
pyplot state exists, so the module is safe in a process with no display.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from modebench.config import RoleSlo
from modebench.stats.aggregate import ModeSummary
from modebench.stats.pareto import ParetoPoint, pareto_front
from modebench.stats.percentiles import Estimate

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
DE_EMPHASIS = "#8d8c85"
_TICKS_S = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0)
_MARKER = {"s": 70, "edgecolors": SURFACE, "linewidths": 2.0}


def _seconds(value: float, _position: float) -> str:
    return f"{value:g} s"


def _new_axes(width: float, height: float) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), facecolor=SURFACE)
    FigureCanvasAgg(figure)
    axes: Any = figure.add_subplot(1, 1, 1)
    return figure, axes


def _style_axes(ax: Any, low_s: float, high_s: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(low_s, high_s)
    ticks = [tick for tick in _TICKS_S if low_s <= tick <= high_s]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_seconds))
    ax.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _limits(values_s: Sequence[float]) -> tuple[float, float]:
    low = max(0.05, min(values_s) * 0.7)
    high = max(max(values_s) * 1.5, low * 2.0)
    return low, high


def _draw_objectives(ax: Any, roles: Mapping[str, RoleSlo], low_s: float, high_s: float) -> None:
    for role, slo in roles.items():
        limit_s = slo.ttfat_p95_ms_max / 1000.0
        if low_s <= limit_s <= high_s:
            ax.axvline(limit_s, color=INK_SECONDARY, linewidth=1.0, zorder=1)
            ax.annotate(
                f"{role}: p95 <= {limit_s:g} s",
                (limit_s, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color=INK_SECONDARY,
            )


def _save(figure: Figure, path: Path) -> Path:
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    return path


def plot_pareto(
    summaries: Sequence[ModeSummary], roles: Mapping[str, RoleSlo], path: Path
) -> Path | None:
    """Write the Pareto chart. Return None if no mode has a quality estimate."""
    graded: list[tuple[ModeSummary, Estimate]] = []
    for candidate in summaries:
        if candidate.quality is not None:
            graded.append((candidate, candidate.quality))
    if not graded:
        return None
    points = [
        ParetoPoint(summary.mode_id, quality.value, summary.ttfat_p95_ms.value)
        for summary, quality in graded
    ]
    front = pareto_front(points)
    front_keys = {point.key for point in front}
    bounds = [s.ttfat_p95_ms.ci_low / 1000.0 for s, _ in graded]
    bounds += [s.ttfat_p95_ms.ci_high / 1000.0 for s, _ in graded]
    low_s, high_s = _limits(bounds)
    figure, ax = _new_axes(9.5, 6.0)
    _style_axes(ax, low_s, high_s)
    _draw_objectives(ax, roles, low_s, high_s)
    for summary, quality in graded:
        p95 = summary.ttfat_p95_ms
        x = p95.value / 1000.0
        y = quality.value
        on_front = summary.mode_id in front_keys
        ax.errorbar(
            x,
            y,
            xerr=[[max(0.0, x - p95.ci_low / 1000.0)], [max(0.0, p95.ci_high / 1000.0 - x)]],
            yerr=[[max(0.0, y - quality.ci_low)], [max(0.0, quality.ci_high - y)]],
            fmt="none",
            ecolor=DE_EMPHASIS if on_front else GRID,
            elinewidth=1.0,
            zorder=2,
        )
        ax.scatter([x], [y], color=SERIES_BLUE if on_front else DE_EMPHASIS, zorder=4, **_MARKER)
        ax.annotate(
            summary.mode_id,
            (x, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK if on_front else INK_SECONDARY,
            zorder=5,
        )
    if len(front) > 1:
        ax.plot(
            [point.latency_ms / 1000.0 for point in front],
            [point.quality for point in front],
            color=SERIES_BLUE,
            linewidth=2.0,
            zorder=3,
        )
    ax.scatter([], [], color=SERIES_BLUE, label="On the Pareto front", **_MARKER)
    ax.scatter([], [], color=DE_EMPHASIS, label="Dominated", **_MARKER)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(
        "TTFAT p95, cold cache (log scale). Bars: 95% interval", color=INK_SECONDARY, fontsize=9
    )
    ax.set_ylabel("Quality (0 to 1)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        "Quality against time to the first answer token", color=INK, fontsize=12, loc="left"
    )
    return _save(figure, path)


def plot_latency(
    summaries: Sequence[ModeSummary], roles: Mapping[str, RoleSlo], path: Path
) -> Path | None:
    """Write the latency chart: one row for each mode, a dot for the p50 and one for the p95."""
    if not summaries:
        return None
    ordered = sorted(summaries, key=lambda summary: summary.ttfat_p95_ms.value, reverse=True)
    p50 = [summary.ttfat_p50_ms.value / 1000.0 for summary in ordered]
    p95 = [summary.ttfat_p95_ms.value / 1000.0 for summary in ordered]
    low_s, high_s = _limits([*p50, *p95])
    rows = list(range(len(ordered)))
    figure, ax = _new_axes(9.5, 1.2 + 0.45 * len(ordered))
    _style_axes(ax, low_s, high_s)
    _draw_objectives(ax, roles, low_s, high_s)
    ax.hlines(rows, p50, p95, color=DE_EMPHASIS, linewidth=2.0, zorder=2)
    ax.scatter(p50, rows, color=SERIES_BLUE, zorder=3, label="p50", **_MARKER)
    ax.scatter(p95, rows, color=SERIES_ORANGE, zorder=3, label="p95", **_MARKER)
    ax.set_yticks(rows)
    ax.set_yticklabels([summary.mode_id for summary in ordered], fontsize=9, color=INK)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    ax.set_xlabel(
        "Time to the first answer token, cold cache (log scale)", color=INK_SECONDARY, fontsize=9
    )
    ax.set_title("TTFAT of each mode: p50 and p95", color=INK, fontsize=12, loc="left")
    return _save(figure, path)
