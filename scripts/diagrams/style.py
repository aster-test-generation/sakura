"""Shared figure styling for the paper's diagrams.

Every figure here is generated at a large figsize and then scaled down by LaTeX,
often by a factor of five or more. Point sizes chosen against the generated PDF
are therefore misleading: a 0.8pt frame placed at a fifth of its natural width
renders as a 0.16pt hairline, which is what made the first drafts look faint.

So the conventions below are declared in *paper* points -- the size the reader
actually sees -- and converted to matplotlib points per figure via PaperStyle,
which knows how wide that figure is placed. Two figures generated at different
figsizes and placed at different widths still come out visually identical.

To retune the whole paper, edit the PAPER_* constants. To fix a single figure
that sits at an unexpected width, edit that script's PAPER_WIDTH_IN.
"""

from dataclasses import dataclass
from typing import Any

# Target appearance, in points as rendered in the paper. These reproduce the
# settled look of the RQ5 complexity panels at their 1.75in placement.
PAPER_TICK_LABEL_PT = 8.2
PAPER_TITLE_PT = 8.6
PAPER_LEGEND_PT = 7.0
PAPER_AXIS_LINE_PT = 0.58
PAPER_TICK_WIDTH_PT = 0.58
PAPER_TICK_LENGTH_PT = 2.33
PAPER_BAR_EDGE_PT = 0.35
PAPER_ERROR_LINE_PT = 0.43
PAPER_ERROR_CAP_PT = 1.17
PAPER_SERIES_LINE_PT = 0.78
PAPER_MARKER_PT = 2.72

# Libertine's regular weight thins out badly at these reductions.
TICK_LABEL_WEIGHT = "semibold"

# Pinned so larger labels cannot push the auto-locator to 0/25/50/75/100 on some
# panels and not others.
Y_TICKS = [0, 20, 40, 60, 80, 100]

# Bar geometry. The bars are widened by reclaiming the space around them rather
# than by shrinking the labels, which barely moves them. X_MARGIN is set so the
# padding at the box edges comes out roughly equal to GROUP_GAP, keeping the
# spacing even across the panel instead of pinched at the ends.
BAR_WIDTH = 0.12
GROUP_GAP = 0.10
X_MARGIN = 0.03


@dataclass(frozen=True)
class PaperStyle:
    """Converts paper-space sizes to matplotlib sizes for one figure."""

    fig_width_in: float
    paper_width_in: float

    @property
    def scale(self) -> float:
        """Factor LaTeX applies to this figure when placing it."""
        return self.paper_width_in / self.fig_width_in

    def pt(self, paper_pt: float) -> float:
        """Size in matplotlib points that renders as paper_pt in the paper."""
        return paper_pt / self.scale

    @property
    def tick_label_size(self) -> float:
        return self.pt(PAPER_TICK_LABEL_PT)

    @property
    def title_size(self) -> float:
        return self.pt(PAPER_TITLE_PT)

    @property
    def legend_size(self) -> float:
        return self.pt(PAPER_LEGEND_PT)

    def errorbar_kwargs(self) -> dict[str, Any]:
        """Line, marker and error-bar keywords for ax.errorbar."""
        error_line = self.pt(PAPER_ERROR_LINE_PT)
        return {
            "linewidth": self.pt(PAPER_SERIES_LINE_PT),
            "markersize": self.pt(PAPER_MARKER_PT),
            "capsize": self.pt(PAPER_ERROR_CAP_PT),
            "capthick": error_line,
            "elinewidth": error_line,
        }

    def bar_kwargs(self) -> dict[str, Any]:
        """Edge and error-bar keywords for ax.bar."""
        error_line = self.pt(PAPER_ERROR_LINE_PT)
        return {
            "edgecolor": "black",
            "linewidth": self.pt(PAPER_BAR_EDGE_PT),
            "capsize": self.pt(PAPER_ERROR_CAP_PT),
            "error_kw": {"elinewidth": error_line, "capthick": error_line},
        }

    def style_axes(self, ax: Any, pin_yticks: bool = True) -> None:
        """Thicken the frame and ticks so they stay legible at paper scale."""
        for spine in ax.spines.values():
            spine.set_linewidth(self.pt(PAPER_AXIS_LINE_PT))
        if pin_yticks:
            ax.set_yticks(Y_TICKS)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.tick_label_size,
            width=self.pt(PAPER_TICK_WIDTH_PT),
            length=self.pt(PAPER_TICK_LENGTH_PT),
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(TICK_LABEL_WEIGHT)
