#Test branch
from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def natural_sort_key(text: str) -> List:
    """
    Generate a key for natural sorting that handles mixed alphanumeric strings.
    This ensures that FAI69_1, FAI69_2, FAI69_10 are sorted in numerical order
    rather than lexicographical order.
    
    Args:
        text: String to generate sort key for
        
    Returns:
        List of alternating strings and integers for proper sorting
    """
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', text)]


def natural_sort(items: List[str]) -> List[str]:
    """
    Sort a list of strings using natural sorting (numerical order for numbers).
    
    Args:
        items: List of strings to sort
        
    Returns:
        List sorted using natural ordering
    """
    return sorted(items, key=natural_sort_key)


class BoxplotBinStack:
    """Bin & Stack with optional nested grouping, conventional boxplots, and
       global auto-shrink of dots that also tightens Y-axis binning and re-stacks.
       Adds a minimum dot size floor; when reached, allow controlled, even overlap.
       NEW: Per-*cell* (combined category/box) spacing is constant across all rows in that cell.
       RULE: If more than 6 boxes are rendered, hide per-box stats and rotate x-tick labels vertically.
    """

    GROUP_WIDTH = 0.8          # horizontal span allocated per PRIMARY category (center ± 0.4)
    BOX_WIDTH_FACTOR = 0.55    # fraction of slot width used for each box

    def __init__(
        self,
        dot_size: float = 0.8,
        gap_factor: float = 0.1,
        gap_scale: float = 1.0,
        min_px_gap: float = 0.0,
        show_boxplots: bool = True,
        color_by: str = "secondary",            # "secondary" | "primary" | "group"
        nested: bool = True,                    # True = primary on x, secondary side-by-side
        show_primary_guides: bool = True,       # dashed vertical guides between primaries
        primary_guide_color: str = "#B0B0B0",
        primary_guide_linestyle: str = "--",
        primary_guide_linewidth: float = 0.6,
        primary_guide_alpha: float = 0.9,
        stats_box_edgecolor: str = "lightgrey",
        ref_linestyle: str = "--",
        ref_linewidth: float = 0.8,
        # minimum dot radius (points). Once reached, stop global shrinking.
        min_dot_radius_pt: float = 1.6,
        fai_column_display_name: Optional[str] = None,  # Display name for grouped variables
    ):
        self.dot_size = float(dot_size)
        self.gap_factor = float(gap_factor)
        self.gap_scale = float(gap_scale)
        self.min_px_gap = float(min_px_gap)
        self.show_boxplots = bool(show_boxplots)
        self.color_by = color_by.lower()
        self.nested = bool(nested)
        self.show_primary_guides = show_primary_guides
        self.primary_guide_color = primary_guide_color
        self.primary_guide_linestyle = primary_guide_linestyle
        self.primary_guide_linewidth = float(primary_guide_linewidth)
        self.primary_guide_alpha = float(primary_guide_alpha)
        self.stats_box_edgecolor = stats_box_edgecolor
        self.ref_linestyle = ref_linestyle
        self.ref_linewidth = float(ref_linewidth)

        # minimum dot radius floor (in points)
        self.min_dot_radius_pt = float(min_dot_radius_pt)
        
        # display name for grouped variables
        self.fai_column_display_name = fai_column_display_name

        # runtime bin/marker info
        self.current_bin_info: Optional[Dict[str, Any]] = None

        # runtime rendering flags (set per call)
        self._suppress_stats: bool = False
        self._vertical_xticks: bool = False

        # palette
        self.colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

    # ------------------------------- Public API -------------------------------

    def generate_visualization(
        self,
        df: pd.DataFrame,
        fai_column: str,
        categorical_column: Optional[str],         # primary
        secondary_categorical: Optional[str] = None,
        show_reference_lines: bool = True,
        title: Optional[str] = None,
        target: Optional[float] = None,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if fai_column not in df.columns:
                return {"error": f'FAI column "{fai_column}" not found in data'}

            # --- Prepare data & grouping
            if not categorical_column or categorical_column == "None":
                plot_data = df[[fai_column]].dropna().copy()
                use_grouping = False
                group_column = "All Data"
            else:
                # debug
                logger.info("=== BOXPLOT BIN STACK DEBUG INFO ===")
                logger.info(f"Boxplot Debug - DataFrame columns: {list(df.columns)}")
                logger.info(f"Boxplot Debug - fai_column: {fai_column}")
                logger.info(f"Boxplot Debug - categorical_column: {categorical_column}")
                logger.info(f"Boxplot Debug - secondary_categorical: {secondary_categorical}")
                logger.info(f"Boxplot Debug - DataFrame shape: {df.shape}")

                if categorical_column not in df.columns:
                    logger.error(f"Boxplot Debug - ❌ categorical_column '{categorical_column}' NOT found in DataFrame")
                    logger.error(f"Boxplot Debug - Available columns: {list(df.columns)}")
                    logger.error(f"Boxplot Debug - DataFrame head: {df.head().to_dict()}")
                    return {"error": f'Categorical column "{categorical_column}" not found in data'}
                else:
                    logger.info(f"Boxplot Debug - ✅ categorical_column '{categorical_column}' found in DataFrame")
                    logger.info(f"Boxplot Debug - categorical_column unique values: {df[categorical_column].unique()}")

                logger.info("=== END BOXPLOT BIN STACK DEBUG INFO ===")
                cols = [fai_column, categorical_column]
                if secondary_categorical and secondary_categorical in df.columns and secondary_categorical != "None":
                    cols.append(secondary_categorical)
                plot_data = df[cols].dropna().copy()
                use_grouping = True
                if secondary_categorical and secondary_categorical in plot_data.columns and secondary_categorical != "None":
                    plot_data["group"] = (
                        plot_data[categorical_column].astype(str) + " | " + plot_data[secondary_categorical].astype(str)
                    )
                    group_column = "group"
                else:
                    group_column = categorical_column

            # -------- NEW: determine how many boxes will be rendered
            if not use_grouping:
                total_boxes = 1
            else:
                total_boxes = int(plot_data[group_column].astype(str).nunique())

            # Set runtime flags based on rule (> 6 boxes)
            self._suppress_stats = total_boxes > 8
            self._vertical_xticks = total_boxes > 8

            description = "Specifications from spec sheet"

            # --- Figure
            fig, ax = plt.subplots(figsize=(14, 10))
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("black")

            # --- Determine y-limits prelim (include refs)
            y_data = plot_data[fai_column]
            reference_lines = {"target": target, "usl": usl, "lsl": lsl} if show_reference_lines else {}
            y_min, y_max = self._span_with_refs(y_data, reference_lines)

            padding = (y_max - y_min) * 0.05 if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.set_xlim(-0.5, 0.5)  # temp so transforms are valid
            self._ensure_layout(ax)  # freeze layout before point→data conversions

            # --- Initial pixel-aware marker sizing (pre-shrink)
            base_area_pt2 = 50.0
            dot_area_pt2 = base_area_pt2 * (self.dot_size ** 2)
            dot_radius_pt = (dot_area_pt2 ** 0.5) / 2.0
            dot_diam_x = 2.0 * self._points_to_data_units_x(ax, dot_radius_pt)
            dot_diam_y = 2.0 * self._points_to_data_units_y(ax, dot_radius_pt)

            # --- Initial Y binning
            bin_width = max((y_max - y_min) / 100.0, dot_diam_y * 1.05)
            num_bins = max(5, int(np.ceil((y_max - y_min + 2 * padding) / bin_width)))

            y0 = (y_min - padding)
            y1 = (y_max + padding)
            total_span = y1 - y0
            bin_width = total_span / num_bins

            # Store bin info
            self.current_bin_info = {
                "y_min": y0, "y_max": y1,
                "bin_width": bin_width, "num_bins": num_bins,
                "dot_area_pt2": dot_area_pt2, "dot_radius_pt": dot_radius_pt,
                "dot_diam_x": dot_diam_x, "dot_diam_y": dot_diam_y,
            }

            # --- Draw
            if use_grouping:
                if self.nested and (secondary_categorical and secondary_categorical in plot_data.columns):
                    self._create_grouped_bin_stack_nested(
                        ax, plot_data, fai_column, categorical_column, secondary_categorical,
                        show_reference_lines, target, usl, lsl, title, y_min, y_max, padding, display_name
                    )
                else:
                    self._create_grouped_bin_stack_flat(
                        ax, plot_data, fai_column, group_column, categorical_column,
                        secondary_categorical, show_reference_lines, target, usl, lsl, title, y_min, y_max, padding, display_name
                    )
            else:
                self._create_single_bin_stack(
                    ax, plot_data, fai_column, show_reference_lines, target, usl, lsl, title, y_min, y_max, padding, display_name
                )

            # --- Export
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            plt.close(fig)

            # --- Stats payload (still returned; only on-plot stats are hidden)
            stats = self._calculate_statistics(plot_data, fai_column, group_column if use_grouping else "All Data", use_grouping)

            return {
                "plot_image": img_b64,
                "statistics": stats,
                "groups": [str(k) for k in stats.keys()],
                "reference_lines": {
                    "target": float(target) if target is not None else None,
                    "usl": float(usl) if usl is not None else None,
                    "lsl": float(lsl) if lsl is not None else None,
                    "description": str(description) if description is not None else None
                },
            }

        except Exception as e:
            logger.exception("Boxplot Bin & Stack generation failed")
            return {"error": f"Boxplot Bin & Stack generation failed: {e}"}

    # ------------------------------- Layout (nested) --------------------------

    def _create_grouped_bin_stack_nested(
        self,
        ax,
        plot_data: pd.DataFrame,
        fai_column: str,
        primary_col: str,
        secondary_col: str,
        show_reference_lines: bool,
        target: Optional[float],
        usl: Optional[float],
        lsl: Optional[float],
        title: Optional[str],
        y_min: float,
        y_max: float,
        padding: float,
        display_name: Optional[str] = None,
    ):
        primaries = natural_sort(list(pd.Series(plot_data[primary_col].astype(str)).unique()))
        secondaries = natural_sort(list(pd.Series(plot_data[secondary_col].astype(str)).unique()))

        P, S = len(primaries), len(secondaries)
        primary_x = np.arange(P, dtype=float)

        inner_width = self.GROUP_WIDTH * 0.9
        if S == 1:
            offsets = np.array([0.0])
            slot_width = inner_width
        else:
            slot_width = inner_width / S
            offsets = (np.arange(S) - (S - 1) / 2.0) * slot_width

        ax.set_xlim(-0.5, P - 0.5)
        self._ensure_layout(ax)
        self._refresh_x_diameter(ax)

        # Auto-shrink iterative cycle (shrink -> rebin -> re-evaluate)
        self._auto_shrink_cycle(
            ax=ax, slot_width=slot_width, y_min=y_min, y_max=y_max, padding=padding,
            mode="nested",
            plot_data=plot_data, fai_column=fai_column,
            primary_col=primary_col, secondary_col=secondary_col,
            primaries=primaries, secondaries=secondaries
        )

        self._ensure_layout(ax)
        self._refresh_x_diameter(ax)

        # Per-CELL spacing
        row_spacing_by_cell = self._compute_row_spacing_per_cell_global(
            ax=ax, slot_width=slot_width, mode="nested",
            plot_data=plot_data, fai_column=fai_column,
            primary_col=primary_col, secondary_col=secondary_col,
            primaries=primaries, secondaries=secondaries
        )

        # Optional guides
        if self.show_primary_guides:
            self._draw_primary_dividers_between_groups(ax, primary_x)

        # Legend (by secondary or primary)
        legend_handles: List[Any] = []
        if self.color_by == "secondary":
            for si, s in enumerate(secondaries):
                legend_handles.append(plt.Line2D([0], [0], color=self.colors[si % len(self.colors)], lw=6, label=str(s)))
        elif self.color_by == "primary":
            for pi, p in enumerate(primaries):
                legend_handles.append(plt.Line2D([0], [0], color=self.colors[pi % len(self.colors)], lw=6, label=str(p)))

        # Draw each cell
        for pi, p in enumerate(primaries):
            for si, s in enumerate(secondaries):
                mask = (plot_data[primary_col].astype(str) == p) & (plot_data[secondary_col].astype(str) == s)
                series = plot_data.loc[mask, fai_column]
                if len(series) == 0:
                    continue

                x_center = primary_x[pi] + offsets[si]

                # Color choice
                if self.color_by == "secondary":
                    color = self.colors[si % len(self.colors)]
                elif self.color_by == "primary":
                    color = self.colors[pi % len(self.colors)]
                else:  # group-specific
                    color = self.colors[(pi * S + si) % len(self.colors)]

                # Box behind dots
                if self.show_boxplots:
                    self._draw_boxplot(ax, series, x_center, color, width=slot_width * self.BOX_WIDTH_FACTOR)

                # Dots (use per-cell spacing)
                self._plot_group_bin_stack(
                    ax, series, float(x_center), f"{p} | {s}", color, slot_width,
                    row_spacing=row_spacing_by_cell[(p, s)]
                )

        # Labels
        ax.set_xticks(primary_x)
        ax.set_xticklabels(primaries)
        self._maybe_rotate_xticklabels(ax)   # NEW
        ax.set_xlabel(primary_col)
        # Use display name if provided, otherwise use fai_column
        y_label = display_name if display_name else fai_column
        ax.set_ylabel(y_label)
        
        # Debug logging for title
        logger.info(f"Boxplot Debug - Nested: title='{title}', y_label='{y_label}', primary_col='{primary_col}', secondary_col='{secondary_col}'")
        final_title = title or f"{y_label} by {primary_col} and {secondary_col}"
        logger.info(f"Boxplot Debug - Nested: final_title='{final_title}'")
        ax.set_title(final_title)

        # Reference lines & legend
        if show_reference_lines:
            self._add_reference_lines(ax, target, usl, lsl)
        if legend_handles:
            title_txt = secondary_col if self.color_by == "secondary" else (primary_col if self.color_by == "primary" else None)
            leg = ax.legend(handles=legend_handles, title=title_txt, loc="center left", bbox_to_anchor=(1.01, 0.5))
            leg._legend_box.align = "left"

        ax.grid(False)
        ax.set_axisbelow(True)

    # ------------------------------- Layout (flat) ----------------------------

    def _create_grouped_bin_stack_flat(
        self,
        ax,
        plot_data: pd.DataFrame,
        fai_column: str,
        group_column: str,
        primary_col: str,
        secondary_col: Optional[str],
        show_reference_lines: bool,
        target: Optional[float],
        usl: Optional[float],
        lsl: Optional[float],
        title: Optional[str],
        y_min: float,
        y_max: float,
        padding: float,
        display_name: Optional[str] = None,
    ):
        groups = natural_sort(list(plot_data[group_column].astype(str).unique()))
        n = len(groups)
        x_positions = np.arange(n, dtype=float)

        # color lookup lists
        primaries = natural_sort(list(plot_data[primary_col].astype(str).unique()))
        secondaries = natural_sort(list(plot_data[secondary_col].astype(str).unique())) if secondary_col and secondary_col in plot_data.columns else []

        ax.set_xlim(-0.5, n - 0.5)
        self._ensure_layout(ax)
        self._refresh_x_diameter(ax)

        slot_width = self.GROUP_WIDTH  # one slot per flat group

        # Auto-shrink iterative cycle
        self._auto_shrink_cycle(
            ax=ax, slot_width=slot_width, y_min=y_min, y_max=y_max, padding=padding,
            mode="flat",
            plot_data=plot_data, fai_column=fai_column,
            group_column=group_column, groups=groups
        )

        self._ensure_layout(ax)
        self._refresh_x_diameter(ax)

        # Per-CELL spacing
        row_spacing_by_cell = self._compute_row_spacing_per_cell_global(
            ax=ax, slot_width=slot_width, mode="flat",
            plot_data=plot_data, fai_column=fai_column,
            group_column=group_column, groups=groups
        )

        for gi, g in enumerate(groups):
            series = plot_data.loc[plot_data[group_column].astype(str) == g, fai_column]
            if len(series) == 0:
                continue

            # color scheme
            if self.color_by == "primary" and "|" in g:
                primary_val = g.split(" | ")[0]
                color = self.colors[primaries.index(primary_val) % len(self.colors)]
            elif self.color_by == "secondary" and "|" in g and secondaries:
                secondary_val = g.split(" | ")[1]
                color = self.colors[secondaries.index(secondary_val) % len(self.colors)]
            else:
                color = self.colors[gi % len(self.colors)]

            if self.show_boxplots:
                self._draw_boxplot(ax, series, x_positions[gi], color, width=slot_width * self.BOX_WIDTH_FACTOR)

            self._plot_group_bin_stack(
                ax, series, float(x_positions[gi]), g, color, slot_width,
                row_spacing=row_spacing_by_cell[g]
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups)
        self._maybe_rotate_xticklabels(ax)   # NEW
        ax.set_xlabel(" | ".join([c for c in [primary_col, secondary_col] if c]))
        # Use display name if provided, otherwise use fai_column
        y_label = display_name if display_name else fai_column
        ax.set_ylabel(y_label)
        
        # Debug logging for title
        logger.info(f"Boxplot Debug - Flat: title='{title}', y_label='{y_label}', primary_col='{primary_col}', secondary_col='{secondary_col}'")
        final_title = title or f"{y_label} vs. {primary_col}" + (f" | {secondary_col}" if secondary_col else "")
        logger.info(f"Boxplot Debug - Flat: final_title='{final_title}'")
        ax.set_title(final_title)

        if show_reference_lines:
            self._add_reference_lines(ax, target, usl, lsl)

        ax.grid(False)
        ax.set_axisbelow(True)

    # ------------------------------- Layout (single) --------------------------

    def _create_single_bin_stack(
        self,
        ax,
        plot_data: pd.DataFrame,
        fai_column: str,
        show_reference_lines: bool,
        target: Optional[float],
        usl: Optional[float],
        lsl: Optional[float],
        title: Optional[str],
        y_min: float,
        y_max: float,
        padding: float,
        display_name: Optional[str] = None,
    ):
        series = plot_data[fai_column]
        color = self.colors[0]

        ax.set_xlim(-0.5, 0.5)
        self._ensure_layout(ax)
        self._refresh_x_diameter(ax)

        slot_width = self.GROUP_WIDTH

        # Auto-shrink iterative cycle
        self._auto_shrink_cycle(
            ax=ax, slot_width=slot_width, y_min=y_min, y_max=y_max, padding=padding,
            mode="single",
            series=series
        )

        self._ensure_layout(ax)
        self._refresh_x_diameter(ax)

        # Per-CELL spacing (single cell)
        row_spacing_by_cell = self._compute_row_spacing_per_cell_global(
            ax=ax, slot_width=slot_width, mode="single", series=series
        )

        if self.show_boxplots and len(series) > 0:
            self._draw_boxplot(ax, series, 0.0, color, width=slot_width * self.BOX_WIDTH_FACTOR)

        self._plot_group_bin_stack(
            ax, series, 0.0, "All Data", color, slot_width,
            row_spacing=row_spacing_by_cell["ALL"]
        )

        ax.set_xticks([0])
        ax.set_xticklabels(["All Data"])
        self._maybe_rotate_xticklabels(ax)   # NEW (harmless for single)
        ax.set_xlabel("Data")
        # Use display name if provided, otherwise use fai_column
        y_label = display_name if display_name else fai_column
        ax.set_ylabel(y_label)
        
        # Debug logging for title
        logger.info(f"Boxplot Debug - Single: title='{title}', y_label='{y_label}'")
        final_title = title or f"{y_label} Distribution (All Data)"
        logger.info(f"Boxplot Debug - Single: final_title='{final_title}'")
        ax.set_title(final_title)

        if show_reference_lines:
            self._add_reference_lines(ax, target, usl, lsl)

        ax.grid(False)
        ax.set_axisbelow(True)

    # ------------------------------- Core plotting ----------------------------

    def _plot_group_bin_stack(
        self,
        ax,
        group_data: pd.Series,
        x_pos: float,
        group_name: str,
        color: str,
        slot_width: float,
        row_spacing: Dict[int, float],   # per-*cell* fixed spacing (data units) for each bin
    ):
        if len(group_data) == 0 or self.current_bin_info is None:
            return

        y0 = self.current_bin_info["y_min"]
        bin_w = self.current_bin_info["bin_width"]
        num_bins = self.current_bin_info["num_bins"]
        dot_area_pt2 = self.current_bin_info["dot_area_pt2"]

        # Assign to bins using the *current* binning
        vals = group_data.to_numpy()
        idx = self._assign_bins(vals)

        # Per-slot bounds
        x_left = x_pos - slot_width / 2.0
        x_right = x_pos + slot_width / 2.0
        max_row_width = x_right - x_left

        for b in range(num_bins):
            mask = (idx == b)
            if not np.any(mask):
                continue

            n = int(mask.sum())
            row_y = y0 + (b + 0.5) * bin_w  # BIN CENTER

            if n == 1:
                x_positions = np.array([x_pos], dtype=float)
            else:
                spacing = row_spacing.get(b, 0.0)
                if spacing <= 0.0:
                    x_positions = np.array([x_pos], dtype=float)
                else:
                    total_row_width = (n - 1) * spacing
                    if total_row_width > max_row_width + 1e-12 and total_row_width > 0:
                        spacing *= max_row_width / total_row_width
                        total_row_width = (n - 1) * spacing
                    start = x_pos - total_row_width / 2.0
                    x_positions = start + np.arange(n, dtype=float) * spacing

            ax.scatter(
                x_positions,
                np.full(n, row_y, dtype=float),
                c=color,
                alpha=0.9,
                s=dot_area_pt2,
                edgecolors="none",
                linewidth=0.0,
                zorder=3,
            )

        # -------- per-box stats text (now conditional)
        if not self._suppress_stats:
            y_top = ax.get_ylim()[1]
            ax.text(
                x_pos,
                y_top,
                self._format_group_stats(group_data, group_name),
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="none"),
                zorder=5,
            )

        # Group name positioning based on rule
        if self._vertical_xticks:  # When rule is triggered, show at top vertically
            y_top = ax.get_ylim()[1]
            y_bottom = ax.get_ylim()[0]
            y_range = y_top - y_bottom
            
            # More conservative positioning - use larger margin to ensure text stays within bounds
            margin = 0.08 * y_range  # 8% of y-range as margin
            text_position = y_top - margin  # Position well below the top edge
            
            # Add mean to group name for triggered version
            mean_value = group_data.mean()
            if pd.notna(mean_value):
                display_text = f"{group_name} (Mean: {mean_value:.3f})"
            else:
                display_text = group_name
            
            ax.text(
                x_pos,
                text_position,
                display_text,
                ha="center",
                va="top",  # Anchor to top of text
                rotation=90,
                rotation_mode="anchor",
                fontsize=5,  # Even smaller font size
                fontweight="normal",
                color="black",
                zorder=5,
            )
        else:  # When rule is NOT triggered, show at bottom normally
            y_bottom = ax.get_ylim()[0]
            offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

            ax.text(
                x_pos,
                y_bottom + offset,
                group_name,
                ha="center",
                va="bottom",
                rotation=0,
                rotation_mode="anchor",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=5,
            )

    # ------------------------------- Auto-shrink logic ------------------------

    def _ensure_layout(self, ax):
        try:
            ax.figure.canvas.draw()
        except Exception:
            pass

    def _assign_bins(self, vals: np.ndarray) -> np.ndarray:
        if self.current_bin_info is None or len(vals) == 0:
            return np.array([], dtype=int)
        y0 = self.current_bin_info["y_min"]
        bin_w = self.current_bin_info["bin_width"]
        num_bins = self.current_bin_info["num_bins"]
        idx = np.floor((vals - y0) / max(bin_w, 1e-12)).astype(int)
        return np.clip(idx, 0, num_bins - 1)

    def _counts_per_bin_series(self, series: pd.Series) -> np.ndarray:
        if self.current_bin_info is None or len(series) == 0:
            return np.zeros(self.current_bin_info["num_bins"] if self.current_bin_info else 0, dtype=int)
        idx = self._assign_bins(series.to_numpy())
        counts = np.bincount(idx, minlength=self.current_bin_info["num_bins"])
        return counts.astype(int)

    def _max_row_in_series(self, series: pd.Series) -> int:
        if self.current_bin_info is None or len(series) == 0:
            return 1
        counts = self._counts_per_bin_series(series)
        return int(counts.max()) if counts.size else 1

    def _max_row_in_groups_flat(self, plot_data: pd.DataFrame, fai_column: str, group_column: str, groups: List[str]) -> int:
        if self.current_bin_info is None:
            return 1
        mr = 1
        for g in groups:
            series = plot_data.loc[plot_data[group_column].astype(str) == g, fai_column]
            mr = max(mr, self._max_row_in_series(series))
        return mr

    def _max_row_in_groups_nested(self, plot_data: pd.DataFrame, fai_column: str, primary_col: str, secondary_col: str, primaries: List[str], secondaries: List[str]) -> int:
        if self.current_bin_info is None:
            return 1
        mr = 1
        for p in primaries:
            for s in secondaries:
                mask = (plot_data[primary_col].astype(str) == p) & (plot_data[secondary_col].astype(str) == s)
                series = plot_data.loc[mask, fai_column]
                mr = max(mr, self._max_row_in_series(series))
        return mr

    def _auto_shrink_cycle(self, *, ax, slot_width: float, y_min: float, y_max: float, padding: float,
                           mode: str, plot_data: Optional[pd.DataFrame] = None,
                           fai_column: Optional[str] = None,
                           group_column: Optional[str] = None, groups: Optional[List[str]] = None,
                           primary_col: Optional[str] = None, secondary_col: Optional[str] = None,
                           primaries: Optional[List[str]] = None, secondaries: Optional[List[str]] = None,
                           series: Optional[pd.Series] = None,
                           max_iters: int = 3, tol: float = 0.995) -> None:
        if self.current_bin_info is None:
            return

        for _ in range(max_iters):
            if mode == "flat":
                assert plot_data is not None and fai_column and group_column and groups is not None
                max_row = self._max_row_in_groups_flat(plot_data, fai_column, group_column, groups)
            elif mode == "nested":
                assert plot_data is not None and fai_column and primary_col and secondary_col and primaries is not None and secondaries is not None
                max_row = self._max_row_in_groups_nested(plot_data, fai_column, primary_col, secondary_col, primaries, secondaries)
            else:  # single
                assert series is not None
                max_row = self._max_row_in_series(series)

            scale = self._compute_needed_scale_for_counts(ax, max_row, slot_width)
            if scale >= tol:
                break

            hit_floor = self._apply_global_dot_scale(ax, scale)
            self._post_scale_rebin(ax, y_min, y_max, padding)
            if hit_floor:
                break

    def _compute_needed_scale_for_counts(self, ax, max_count_in_row: int, slot_width: float) -> float:
        if self.current_bin_info is None or max_count_in_row <= 1:
            return 1.0

        dot_diam_x = self.current_bin_info["dot_diam_x"]
        spacing = float(dot_diam_x) * (1.0 + self.gap_scale * self.gap_factor * self.dot_size)
        if self.min_px_gap > 0.0:
            spacing += self._px_to_data_x(ax, self.min_px_gap)

        total_row_width = (max_count_in_row - 1) * spacing
        if total_row_width <= 0:
            return 1.0

        scale_needed = min(1.0, slot_width / total_row_width)

        r_pt_now = self.current_bin_info["dot_radius_pt"]
        min_scale_allowed = max(0.0, self.min_dot_radius_pt / max(r_pt_now, 1e-12))

        return max(scale_needed, min_scale_allowed)

    def _apply_global_dot_scale(self, ax, scale: float) -> bool:
        if self.current_bin_info is None or scale >= 1.0:
            return False

        r_pt_now = self.current_bin_info["dot_radius_pt"]
        r_pt_new = r_pt_now * scale

        hit_floor = False
        if r_pt_new < self.min_dot_radius_pt:
            r_pt_new = self.min_dot_radius_pt
            hit_floor = True

        self.current_bin_info["dot_radius_pt"] = r_pt_new
        self.current_bin_info["dot_area_pt2"] = (2.0 * r_pt_new) ** 2

        self.current_bin_info["dot_diam_x"] = 2.0 * self._points_to_data_units_x(ax, r_pt_new)
        self.current_bin_info["dot_diam_y"] = 2.0 * self._points_to_data_units_y(ax, r_pt_new)
        return hit_floor

    def _post_scale_rebin(self, ax, y_min: float, y_max: float, padding: float):
        if self.current_bin_info is None:
            return
        y0 = y_min - padding
        y1 = y_max + padding
        total_span = y1 - y0

        dot_diam_y = self.current_bin_info["dot_diam_y"]
        new_bin_width = max((y_max - y_min) / 100.0, dot_diam_y * 1.05)
        new_num_bins = max(5, int(np.ceil((y_max - y_min + 2 * padding) / new_bin_width)))

        new_bin_width = total_span / new_num_bins

        self.current_bin_info.update({
            "y_min": y0,
            "y_max": y1,
            "bin_width": new_bin_width,
            "num_bins": new_num_bins,
        })

    # ---------- Per-*cell* spacing (shared across bins within that cell) ------

    def _compute_row_spacing_per_cell_global(
        self,
        *,
        ax,
        slot_width: float,
        mode: str,  # "nested" | "flat" | "single"
        plot_data: Optional[pd.DataFrame] = None,
        fai_column: Optional[str] = None,
        group_column: Optional[str] = None,
        groups: Optional[List[str]] = None,
        primary_col: Optional[str] = None,
        secondary_col: Optional[str] = None,
        primaries: Optional[List[str]] = None,
        secondaries: Optional[List[str]] = None,
        series: Optional[pd.Series] = None,
    ) -> Dict[Any, Dict[int, float]]:
        if self.current_bin_info is None:
            return {}
        num_bins = self.current_bin_info["num_bins"]

        def counts_for_series(s: pd.Series) -> np.ndarray:
            if s is None or len(s) == 0:
                return np.zeros(num_bins, dtype=int)
            idx = self._assign_bins(s.to_numpy())
            return np.bincount(idx, minlength=num_bins).astype(int)

        # base spacing at current dot diameter
        dot_diam_x = self.current_bin_info["dot_diam_x"]
        base_spacing = float(dot_diam_x) * (1.0 + self.gap_scale * self.gap_factor * self.dot_size)
        if self.min_px_gap > 0.0:
            base_spacing += self._px_to_data_x(ax, self.min_px_gap)

        per_cell: Dict[Any, Dict[int, float]] = {}

        if mode == "single":
            assert series is not None
            arr = counts_for_series(series)
            global_max = int(arr.max()) if arr.size else 1
            spacing = 0.0 if global_max <= 1 else (base_spacing if (global_max - 1) * base_spacing <= slot_width else (slot_width / (global_max - 1)))
            per_cell["ALL"] = {b: spacing for b in range(num_bins)}
            return per_cell

        if mode == "flat":
            assert plot_data is not None and fai_column and group_column and groups is not None
            for g in groups:
                s = plot_data.loc[plot_data[group_column].astype(str) == g, fai_column]
                arr = counts_for_series(s)
                global_max = int(arr.max()) if arr.size else 1
                spacing = 0.0 if global_max <= 1 else (base_spacing if (global_max - 1) * base_spacing <= slot_width else (slot_width / (global_max - 1)))
                per_cell[g] = {b: spacing for b in range(num_bins)}
            return per_cell

        # nested
        assert plot_data is not None and fai_column and primary_col and secondary_col and primaries is not None and secondaries is not None
        for p in primaries:
            for s in secondaries:
                mask = (plot_data[primary_col].astype(str) == p) & (plot_data[secondary_col].astype(str) == s)
                ser = plot_data.loc[mask, fai_column]
                arr = counts_for_series(ser)
                global_max = int(arr.max()) if arr.size else 1
                spacing = 0.0 if global_max <= 1 else (base_spacing if (global_max - 1) * base_spacing <= slot_width else (slot_width / (global_max - 1)))
                per_cell[(p, s)] = {b: spacing for b in range(num_bins)}
        return per_cell

    # ------------------------------- Boxplot (five-number) --------------------

    def _draw_boxplot(self, ax, series: pd.Series, x_pos: float, color: str, width: float):
        bp = ax.boxplot(
            [series.to_numpy()],
            positions=[x_pos],
            widths=[width],
            vert=True,
            patch_artist=False,
            showfliers=False,
            whis=1.5,
            manage_ticks=False,
            zorder=2,
        )
        for element in ["boxes", "whiskers", "caps", "medians"]:
            for artist in bp[element]:
                artist.set_color(color)
                artist.set_alpha(0.95)
                artist.set_linewidth(1.4)

    # ------------------------------- Primary guides ---------------------------

    def _draw_primary_dividers_between_groups(self, ax, primary_x: np.ndarray):
        if self.current_bin_info is None or len(primary_x) < 2:
            return
        y0 = self.current_bin_info["y_min"]
        y1 = self.current_bin_info["y_max"]
        for i in range(len(primary_x) - 1):
            mid = (primary_x[i] + primary_x[i + 1]) / 2.0
            ax.vlines(
                mid, y0, y1,
                colors=self.primary_guide_color,
                linestyles=self.primary_guide_linestyle,
                linewidths=self.primary_guide_linewidth,
                alpha=self.primary_guide_alpha,
                zorder=1
            )

    # ------------------------------- Reference lines --------------------------

    def _add_reference_lines(self, ax, target: Optional[float], usl: Optional[float], lsl: Optional[float]):
        if usl is not None:
            ax.axhline(y=usl, color="red", linestyle=self.ref_linestyle, alpha=0.9, linewidth=self.ref_linewidth, label=f"USL {usl:.3f}")
            ax.text(ax.get_xlim()[1], usl, f"USL {usl:.3f}", ha="left", va="top", fontsize=8, color="red")
        if target is not None:
            ax.axhline(y=target, color="navy", linestyle=self.ref_linestyle, alpha=0.9, linewidth=self.ref_linewidth, label=f"Target {target:.3f}")
            ax.text(ax.get_xlim()[1], target, f"Target {target:.3f}", ha="right", va="bottom", fontsize=8, color="navy")
        if lsl is not None:
            ax.axhline(y=lsl, color="red", linestyle=self.ref_linestyle, alpha=0.9, linewidth=self.ref_linewidth, label=f"LSL {lsl:.3f}")
            ax.text(ax.get_xlim()[1], lsl, f"LSL {lsl:.3f}", ha="left", va="bottom", fontsize=8, color="red")

    # ------------------------------- Converters -------------------------------

    def _points_to_data_units_x(self, ax, points: float) -> float:
        fig = ax.figure
        ax_width_px = ax.get_window_extent().width
        px = (points / 72.0) * fig.dpi
        x0, x1 = ax.get_xlim()
        data_per_px = (x1 - x0) / ax_width_px if ax_width_px > 0 else 0.0
        return px * data_per_px

    def _points_to_data_units_y(self, ax, points: float) -> float:
        fig = ax.figure
        ax_height_px = ax.get_window_extent().height
        px = (points / 72.0) * fig.dpi
        y0, y1 = ax.get_ylim()
        data_per_px = (y1 - y0) / ax_height_px if ax_height_px > 0 else 0.0
        return px * data_per_px

    def _px_to_data_x(self, ax, px: float) -> float:
        x0, x1 = ax.get_xlim()
        ax_width_px = ax.get_window_extent().width
        return (x1 - x0) / ax_width_px * px if ax_width_px > 0 else 0.0

    def _refresh_x_diameter(self, ax):
        if self.current_bin_info is None:
            return
        r_pt = self.current_bin_info["dot_radius_pt"]
        self.current_bin_info["dot_diam_x"] = 2.0 * self._points_to_data_units_x(ax, r_pt)

    # ------------------------------- Stats & specs ----------------------------

    def _format_group_stats(self, group_data: pd.Series, group_name: str) -> str:
        stats = {
            "  Mean": group_data.mean(),
            "  Min": group_data.min(),
            "  Median": group_data.median(),
            "  Max": group_data.max(),
            "  Std Dev": group_data.std(),
            "  N": len(group_data),
        }
        lines = [f"{group_name}:"]
        for k, v in stats.items():
            if pd.notna(v):
                if isinstance(v, (int, np.integer)):
                    lines.append(f"{k}: {v}")
                else:
                    lines.append(f"{k}: {v:.4f}")
        return "\n".join(lines)

    def _calculate_statistics(
        self, plot_data: pd.DataFrame, fai_column: str, group_column: str, use_grouping: bool
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if not use_grouping:
            s = plot_data[fai_column]
            stats["All Data"] = self._stats_for_series(s)
        else:
            for g in natural_sort(list(plot_data[group_column].astype(str).unique())):
                s = plot_data.loc[plot_data[group_column].astype(str) == g, fai_column]
                stats[str(g)] = self._stats_for_series(s)
        return stats

    def _stats_for_series(self, s: pd.Series) -> Dict[str, Any]:
        return {
            "count": int(s.count()),
            "mean": self._safe_float(s.mean()),
            "median": self._safe_float(s.median()),
            "std": self._safe_float(s.std()),
            "min": self._safe_float(s.min()),
            "max": self._safe_float(s.max()),
        }

    def _safe_float(self, v) -> Optional[float]:
        try:
            if pd.isna(v):
                return None
            if hasattr(v, 'item'):
                v = v.item()
            return float(v)
        except Exception:
            return None

    def _span_with_refs(self, data: pd.Series, refs: Dict[str, Optional[float]]) -> Tuple[float, float]:
        dmin = float(np.nanmin(data.to_numpy()).item())
        dmax = float(np.nanmax(data.to_numpy()).item())
        for k in ("target", "usl", "lsl"):
            v = refs.get(k)
            if v is not None:
                dmin = min(dmin, float(v))
                dmax = max(dmax, float(v))
        if dmax - dmin == 0:
            delta = max(1.0, abs(dmin) * 0.01)
            return float(dmin - delta), float(dmax + delta)
        return float(dmin), float(dmax)

    # ------------------------------- Helpers ----------------------------------

    def _maybe_rotate_xticklabels(self, ax):
        """Rotate xtick labels vertically if the rule is active."""
        if self._vertical_xticks:
            for lab in ax.get_xticklabels():
                lab.set_rotation(90)
                lab.set_ha("right")


def generate_boxplot_bin_stack_visualization(
    df: pd.DataFrame,
    fai_column: str,
    categorical_column: Optional[str],         # primary
    secondary_categorical: Optional[str] = None,
    show_reference_lines: bool = True,
    dot_size: float = 0.8,
    title: Optional[str] = None,
    gap_factor: float = 0.1,
    gap_scale: float = 1.0,
    min_px_gap: float = 0.0,
    show_boxplots: bool = True,
    color_by: str = "secondary",
    nested: bool = True,
    show_primary_guides: bool = True,
    target: Optional[float] = None,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    min_dot_radius_pt: float = 1.6,
    fai_column_display_name: Optional[str] = None,  # Display name for grouped variables
) -> Dict[str, Any]:
    logger.info("=== BOXPLOT BIN STACK VISUALIZATION FUNCTION DEBUG ===")
    logger.info(f"Function Debug - Input DataFrame columns: {list(df.columns)}")
    logger.info(f"Function Debug - Input fai_column: {fai_column}")
    logger.info(f"Function Debug - Input categorical_column: {categorical_column}")
    logger.info(f"Function Debug - Input secondary_categorical: {secondary_categorical}")
    logger.info(f"Function Debug - Input fai_column_display_name: {fai_column_display_name}")
    logger.info(f"Function Debug - Input title: {title}")
    logger.info(f"Function Debug - Input DataFrame shape: {df.shape}")
    logger.info(f"Function Debug - Input DataFrame head: {df.head().to_dict()}")
    
    # Use display name for labels and titles if provided, otherwise use fai_column
    display_name = fai_column_display_name if fai_column_display_name else fai_column
    logger.info(f"Function Debug - Computed display_name: {display_name}")

    if fai_column not in df.columns:
        logger.error(f"Function Debug - ❌ fai_column '{fai_column}' NOT found in DataFrame")
        return {"error": f'FAI column "{fai_column}" not found in data'}

    if categorical_column and categorical_column != "None":
        if categorical_column not in df.columns:
            logger.error(f"Function Debug - ❌ categorical_column '{categorical_column}' NOT found in DataFrame")
            return {"error": f'Categorical column "{categorical_column}" not found in data'}
        else:
            logger.info(f"Function Debug - ✅ categorical_column '{categorical_column}' found in DataFrame")
            logger.info(f"Function Debug - categorical_column unique values: {df[categorical_column].unique()}")

    if secondary_categorical and secondary_categorical != "None":
        if secondary_categorical not in df.columns:
            logger.error(f"Function Debug - ❌ secondary_categorical '{secondary_categorical}' NOT found in DataFrame")
            return {"error": f'Secondary categorical column "{secondary_categorical}" not found in data'}
        else:
            logger.info(f"Function Debug - ✅ secondary_categorical '{secondary_categorical}' found in DataFrame")
            logger.info(f"Function Debug - secondary_categorical unique values: {df[secondary_categorical].unique()}")

    logger.info("=== END BOXPLOT BIN STACK VISUALIZATION FUNCTION DEBUG ===")
    gen = BoxplotBinStack(
        dot_size=dot_size,
        gap_factor=gap_factor,
        gap_scale=gap_scale,
        min_px_gap=min_px_gap,
        show_boxplots=show_boxplots,
        color_by=color_by,
        nested=nested,
        show_primary_guides=show_primary_guides,
        min_dot_radius_pt=min_dot_radius_pt,
        fai_column_display_name=fai_column_display_name,
    )
    return gen.generate_visualization(
        df,
        fai_column=fai_column,
        categorical_column=categorical_column,
        secondary_categorical=secondary_categorical,
        show_reference_lines=show_reference_lines,
        title=title,
        target=target,
        usl=usl,
        lsl=lsl,
        display_name=fai_column_display_name,
    )
