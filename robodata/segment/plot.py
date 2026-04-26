"""Timeline visualization for segmentation results."""

from __future__ import annotations

from pathlib import Path

from robodata.segment.models import SegmentationReport

# Maximum episodes to show before truncating
_MAX_ROWS = 50

# Phase → color mapping for labeled segments
PHASE_COLORS: dict[str, str] = {
    "idle": "#9e9e9e",         # gray
    "reaching": "#42a5f5",     # blue
    "grasping": "#ef5350",     # red
    "transporting": "#ff9800", # orange
    "placing": "#66bb6a",      # green
    "retracting": "#ab47bc",   # purple
    "fine_manipulation": "#fdd835",  # yellow
    "moving": "#29b6f6",       # light blue
    "unknown": "#bdbdbd",      # light gray
}


def plot_segmentation(report: SegmentationReport, output_path: str | Path) -> None:
    """Generate a horizontal timeline PNG showing segments per episode.

    Each episode is a row with colored rectangles for segments and
    vertical lines at changepoints. If segments have labels, colors
    are assigned by phase and a legend is shown.

    Args:
        report: Segmentation report with per-episode results.
        output_path: Path to save the PNG file.

    Raises:
        MissingDependencyError: If matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend — no Qt/display needed
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError:
        from robodata.core.exceptions import MissingDependencyError

        raise MissingDependencyError(
            dependency="matplotlib",
            feature="segment --plot",
            install_hint="pip install robodata[visualize]",
        )

    episodes = report.per_episode
    if not episodes:
        return

    truncated = len(episodes) > _MAX_ROWS
    if truncated:
        episodes = episodes[:_MAX_ROWS]

    # Detect if any segments have labels
    has_labels = any(
        seg.label for ep in episodes for seg in ep.segments
    )

    n_episodes = len(episodes)
    fig_height = max(3, 0.4 * n_episodes + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = plt.get_cmap("tab10")
    seen_labels: set[str] = set()

    for row, ep in enumerate(episodes):
        if not ep.segments:
            continue
        for seg in ep.segments:
            if has_labels and seg.label:
                color = PHASE_COLORS.get(seg.label, "#bdbdbd")
                seen_labels.add(seg.label)
            else:
                color = cmap(seg.start % 10)

            rect = Rectangle(
                (seg.start, row - 0.35),
                seg.duration_frames,
                0.7,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Changepoint lines
        for cp in ep.changepoints:
            ax.plot([cp, cp], [row - 0.4, row + 0.4], color="black", linewidth=1, alpha=0.7)

    # Labels and formatting
    ax.set_yticks(range(n_episodes))
    ax.set_yticklabels(
        [ep.episode_id for ep in episodes],
        fontsize=max(6, 10 - n_episodes // 10),
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Episode")

    max_frames = max((ep.num_frames for ep in episodes), default=100)
    ax.set_xlim(0, max_frames)
    ax.set_ylim(-0.5, n_episodes - 0.5)
    ax.invert_yaxis()

    title = f"Segmentation: {report.dataset_path}"
    if report.summary:
        title += f"  (mean={report.summary.get('mean_segments', '?')} segments/ep)"
    if truncated:
        title += f"  [showing {_MAX_ROWS}/{report.num_episodes}]"
    ax.set_title(title, fontsize=10)

    # Add legend if we have labeled segments
    if has_labels and seen_labels:
        # Order legend entries consistently
        ordered = [l for l in PHASE_COLORS if l in seen_labels]
        handles = [
            Patch(facecolor=PHASE_COLORS[label], edgecolor="white", label=label)
            for label in ordered
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            title="Phase",
            title_fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
