# -- import packages: ---------------------------------------------------------
import anndata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import tempfile
import os

# -- set typing: --------------------------------------------------------------
from typing import Callable, Dict, List, Optional, Union


# -- Helper functions: --------------------------------------------------------
def _create_expr_default_background(adata_sim, ax, use_key, background_s, background_inner_s):
    """Default background with black outline and white fill."""
    xu = adata_sim.obsm[use_key]
    if isinstance(xu, pd.DataFrame):
        xu = xu.values
    ax.scatter(xu[:, 0], xu[:, 1], c="k", ec="None", rasterized=True, s=background_s)
    ax.scatter(xu[:, 0], xu[:, 1], c="w", ec="None", rasterized=True, s=background_inner_s)


def _create_expr_grouped_background(
    adata_sim, ax, use_key, background_groupby, background_cmap, background_s, background_inner_s
):
    """Background colored by group membership."""
    groups = adata_sim.obs[background_groupby]
    xu = adata_sim.obsm[use_key]
    if isinstance(xu, pd.DataFrame):
        xu = xu.values

    unique_groups = groups.unique()
    if background_cmap is None:
        default_colors = plt.cm.tab10.colors
        group_colors = {g: default_colors[i % len(default_colors)] for i, g in enumerate(unique_groups)}
    else:
        group_colors = background_cmap

    for group in unique_groups:
        mask = groups == group
        c = group_colors.get(group, "k")
        ax.scatter(xu[mask, 0], xu[mask, 1], c=c, ec="None", rasterized=True, s=background_s)
        ax.scatter(xu[mask, 0], xu[mask, 1], c="w", ec="None", rasterized=True, s=background_inner_s)


def _draw_umap_labels(ax, umap_labels):
    """Draw text labels on the UMAP axes."""
    if umap_labels is None:
        return
    for label in umap_labels:
        text = label.get("text", "")
        x = label.get("x", 0)
        y = label.get("y", 0)
        kwargs = {k: v for k, v in label.items() if k not in ("text", "x", "y")}
        # Set defaults
        kwargs.setdefault("fontsize", 10)
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("va", "center")
        ax.text(x, y, text, **kwargs)


def _create_expression_progenitor_frame(
    adata_sim,
    ax_umap,
    ax_expr,
    background_fn,
    progenitor_x,
    progenitor_y,
    progenitor_color,
    progenitor_s,
    progenitor_label,
    show_time_label,
    time_label_loc,
    time_label_fmt,
    time_label_fontsize,
    t_min,
    t_max,
    umap_title,
    x_all,
    y_all,
    umap_cmap,
    color,
    expr_ylim,
    x_label,
    y_label,
    gene,
    plot_groups,
    expr_cmap,
    linewidth,
    umap_labels=None,
):
    """Create content for dual-panel progenitor intro frame."""
    # === UMAP Panel ===
    background_fn(adata_sim, ax_umap)

    ax_umap.scatter(
        progenitor_x,
        progenitor_y,
        c=progenitor_color,
        s=progenitor_s,
        edgecolors="white",
        linewidths=1.5,
        zorder=300,
    )

    ax_umap.annotate(
        progenitor_label,
        xy=(progenitor_x, progenitor_y),
        xytext=(progenitor_x + 1.5, progenitor_y + 1.5),
        fontsize=12,
        fontweight="bold",
        color=progenitor_color,
        arrowprops=dict(arrowstyle="->", color=progenitor_color, lw=2),
        zorder=301,
    )

    if show_time_label:
        ax_umap.text(
            time_label_loc[0],
            time_label_loc[1],
            time_label_fmt.format(t_min),
            transform=ax_umap.transAxes,
            fontsize=time_label_fontsize,
            verticalalignment="top",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    if umap_title:
        ax_umap.set_title(umap_title, fontsize=12)

    for spine in ax_umap.spines.values():
        spine.set_visible(False)
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.set_xlabel("")
    ax_umap.set_ylabel("")
    ax_umap.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
    ax_umap.set_ylim(y_all.min() - 0.5, y_all.max() + 0.5)

    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
    sm = cm.ScalarMappable(cmap=umap_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_umap, shrink=0.6, orientation="horizontal", location="bottom")
    cbar.set_label(color, fontsize=10)

    cbar_ax = cbar.ax
    cbar_xmin, cbar_xmax = cbar_ax.get_xlim()
    cbar_ax.axvline(x=cbar_xmin, color="dodgerblue", linewidth=2, zorder=10)

    # Draw UMAP labels
    _draw_umap_labels(ax_umap, umap_labels)

    # === Expression Panel ===
    ax_expr.set_xlim(t_min, t_max)
    ax_expr.set_ylim(expr_ylim)
    ax_expr.set_xlabel(x_label, fontsize=10)
    ax_expr.set_ylabel(y_label, fontsize=10)
    ax_expr.set_title(f"$\\it{{{gene}}}$", fontsize=11)
    ax_expr.spines["top"].set_visible(False)
    ax_expr.spines["right"].set_visible(False)
    ax_expr.grid(True, alpha=0.3, zorder=0)

    ax_expr.axvline(x=t_min, color="dodgerblue", linewidth=2, linestyle="--", alpha=0.7, zorder=5)

    for group in plot_groups:
        ax_expr.plot([], [], color=expr_cmap.get(group, "gray"), linewidth=linewidth, label=group)
    ax_expr.legend(loc="best", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=8)


def _create_expression_frame(
    adata_sim,
    ax_umap,
    ax_expr,
    t_current,
    frame_alpha,
    background_fn,
    x_all,
    y_all,
    time_values,
    color_values,
    umap_cmap,
    s,
    alpha,
    trail_alpha,
    leading_edge_scale,
    vmin,
    vmax,
    show_time_label,
    time_label_loc,
    time_label_fmt,
    time_label_fontsize,
    umap_title,
    t_min,
    t_max,
    color,
    stats_full,
    plot_groups,
    expr_cmap,
    linewidth,
    show_std,
    std_alpha,
    expr_ylim,
    x_label,
    y_label,
    gene,
    umap_labels=None,
    **kwargs,
):
    """Create content for a dual-panel animation frame."""
    # === UMAP Panel ===
    background_fn(adata_sim, ax_umap)

    mask = time_values <= t_current
    x = x_all[mask]
    y = y_all[mask]
    c = color_values[mask]
    t_pts = time_values[mask]

    trail_mask = t_pts < t_current
    if np.any(trail_mask):
        ax_umap.scatter(
            x[trail_mask],
            y[trail_mask],
            c=c[trail_mask],
            cmap=umap_cmap,
            s=s,
            alpha=alpha * trail_alpha * frame_alpha,
            vmin=vmin,
            vmax=vmax,
            zorder=200,
            edgecolors="none",
            **kwargs,
        )

    leading_mask = t_pts == t_current
    if np.any(leading_mask):
        ax_umap.scatter(
            x[leading_mask],
            y[leading_mask],
            c=c[leading_mask],
            cmap=umap_cmap,
            s=s * leading_edge_scale,
            alpha=frame_alpha,
            vmin=vmin,
            vmax=vmax,
            zorder=202,
            edgecolors="none",
            **kwargs,
        )

    if show_time_label:
        ax_umap.text(
            time_label_loc[0],
            time_label_loc[1],
            time_label_fmt.format(t_current),
            transform=ax_umap.transAxes,
            fontsize=time_label_fontsize,
            verticalalignment="top",
            fontweight="bold",
            alpha=frame_alpha,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8 * frame_alpha),
        )

    if umap_title:
        ax_umap.set_title(umap_title, fontsize=12)

    for spine in ax_umap.spines.values():
        spine.set_visible(False)
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.set_xlabel("")
    ax_umap.set_ylabel("")
    ax_umap.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
    ax_umap.set_ylim(y_all.min() - 0.5, y_all.max() + 0.5)

    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
    sm = cm.ScalarMappable(cmap=umap_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_umap, shrink=0.6, orientation="horizontal", location="bottom")
    cbar.set_label(color, fontsize=10)

    cbar_ax = cbar.ax
    cbar_xmin, cbar_xmax = cbar_ax.get_xlim()
    progress_x = (
        cbar_xmin + (t_current - t_min) / (t_max - t_min) * (cbar_xmax - cbar_xmin)
        if t_max > t_min
        else cbar_xmax
    )
    cbar_ax.axvline(x=progress_x, color="dodgerblue", linewidth=2, zorder=10)

    # Draw UMAP labels
    _draw_umap_labels(ax_umap, umap_labels)

    # === Expression Panel ===
    stats_current = stats_full[stats_full["time"] <= t_current]

    for group in plot_groups:
        group_data = stats_current[stats_current["group"] == group].sort_values("time")
        if len(group_data) == 0:
            continue

        t = group_data["time"].values
        mean = group_data["mean"].values
        std = group_data["std"].values
        color_line = expr_cmap.get(group, "gray")

        ax_expr.plot(t, mean, color=color_line, linewidth=linewidth, label=group, alpha=frame_alpha, zorder=2)

        if show_std:
            ax_expr.fill_between(
                t,
                mean - std,
                mean + std,
                color=color_line,
                alpha=std_alpha * frame_alpha,
                linewidth=0,
                zorder=1,
            )

    ax_expr.set_xlim(t_min, t_max)
    ax_expr.set_ylim(expr_ylim)
    ax_expr.set_xlabel(x_label, fontsize=10)
    ax_expr.set_ylabel(y_label, fontsize=10)
    ax_expr.set_title(f"$\\it{{{gene}}}$", fontsize=11)
    ax_expr.spines["top"].set_visible(False)
    ax_expr.spines["right"].set_visible(False)
    ax_expr.grid(True, alpha=0.3, zorder=0)

    ax_expr.axvline(
        x=t_current, color="dodgerblue", linewidth=2, linestyle="--", alpha=0.7 * frame_alpha, zorder=5
    )

    handles, labels = ax_expr.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_expr.legend(
        by_label.values(), by_label.keys(), loc="best", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=8
    )


# -- API-facing function: -----------------------------------------------------
def simulation_expression_gif(
    adata_sim: anndata.AnnData,
    gene: str,
    savename: str = "simulation_expression.gif",
    groupby: str = "final_state",
    groups: Optional[List[str]] = None,
    color: str = "t",
    use_key: str = "X_umap",
    time_key: str = "t",
    gene_key: str = "X_gene_inv",
    gene_ids_key: str = "gene_ids",
    figsize: tuple = (12, 6),
    expr_width_scale: float = 0.8,
    expr_height_scale: float = 0.8,
    # UMAP panel options
    umap_cmap: Union[str, mcolors.Colormap] = "plasma_r",
    s: float = 10.0,
    alpha: float = 0.8,
    background_fn: Optional[Callable] = None,
    background_groupby: Optional[str] = None,
    background_cmap: Optional[Dict[str, str]] = None,
    background_s: float = 100.0,
    background_inner_s: float = 65.0,
    umap_labels: Optional[List[Dict]] = None,
    # Expression panel options
    expr_cmap: Optional[Dict[str, str]] = None,
    linewidth: float = 2.0,
    show_std: bool = True,
    std_alpha: float = 0.2,
    x_label: str = "t(d)",
    y_label: str = "Log-norm. expression",
    # Shared options
    show_time_label: bool = True,
    time_label_fmt: str = "t = {:.1f}d",
    time_label_loc: tuple = (0.05, 0.95),
    time_label_fontsize: int = 12,
    umap_title: Optional[str] = None,
    fps: int = 10,
    duration: Optional[float] = None,
    hold_frames: int = 10,
    fade_frames: int = 8,
    leading_edge_scale: float = 2.0,
    trail_alpha: float = 0.5,
    show_progenitor: bool = True,
    progenitor_frames: int = 8,
    progenitor_label: str = "Progenitor",
    progenitor_s: float = 80.0,
    progenitor_color: str = "dodgerblue",
    dpi: int = 100,
    return_fig: bool = False,
    **kwargs,
) -> Union[str, tuple]:
    """
    Create a dual-panel GIF with synchronized UMAP trajectory and gene expression.

    Left panel shows the simulation trajectory growing over UMAP space.
    Right panel shows temporal gene expression (mean ± std) growing over time.

    Parameters
    ----------
    adata_sim : anndata.AnnData
        Simulated data from ``sdq.tl.simulate()``, with UMAP coordinates
        in obsm and gene expression after ``sdq.tl.invert_scaled_gex()``.
    gene : str
        Gene name to plot in the expression panel.
    savename : str, default="simulation_expression.gif"
        Output filename for the GIF.
    groupby : str, default="final_state"
        Column in ``adata_sim.obs`` for grouping trajectories in expression panel.
    groups : List[str], optional
        Specific groups to plot. If None, plots all groups.
    color : str, default="t"
        What to color UMAP points by. Can be column in obs or gene name.
    use_key : str, default="X_umap"
        Key in ``adata_sim.obsm`` containing UMAP coordinates.
    time_key : str, default="t"
        Column in ``adata_sim.obs`` containing time values.
    gene_key : str, default="X_gene_inv"
        Key in ``adata_sim.obsm`` containing gene expression matrix.
    gene_ids_key : str, default="gene_ids"
        Key in ``adata_sim.uns`` containing gene names.
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches for the dual-panel figure.
    expr_width_scale : float, default=0.8
        Width of expression panel relative to UMAP panel.
        0.8 means expression panel is 80% as wide as UMAP panel.
    expr_height_scale : float, default=0.8
        Height scaling for expression panel. Uses GridSpec height_ratios
        to make expression panel shorter. 0.8 means expression panel
        is 80% as tall, with remaining space as padding.
    umap_cmap : str or Colormap, default="plasma_r"
        Colormap for UMAP continuous values.
    s : float, default=10.0
        Point size for simulation points on UMAP.
    alpha : float, default=0.8
        Point transparency on UMAP.
    background_fn : Callable, optional
        Custom function to plot UMAP background. Should accept (adata_sim, ax).
    background_groupby : str, optional
        Column in obs to group background cells by (e.g., "final_state").
        When set, background cells are colored by group using background_cmap.
    background_cmap : Dict[str, str], optional
        Mapping from group names to colors for background. Example:
        {"Mon.": "orange", "Neu.": "#4a7298"}
    background_s : float, default=100.0
        Point size for background outer points.
    background_inner_s : float, default=65.0
        Point size for background inner points.
    umap_labels : List[Dict], optional
        List of label dictionaries to draw on the UMAP panel. Each dict should
        have keys "text", "x", "y", and optionally any matplotlib text kwargs
        like "color", "fontsize", "weight", "ha", "va". Example:
        [{"text": "Monocyte", "x": 10.5, "y": 10, "color": "#F08700", "weight": "bold"}]
    expr_cmap : Dict[str, str], optional
        Mapping from group names to colors for expression panel.
    linewidth : float, default=2.0
        Width of mean lines in expression panel.
    show_std : bool, default=True
        Whether to show standard deviation shading in expression panel.
    std_alpha : float, default=0.2
        Transparency of std shading.
    x_label : str, default="t(d)"
        X-axis label for expression panel.
    y_label : str, default="Log-norm. expression"
        Y-axis label for expression panel.
    show_time_label : bool, default=True
        Whether to show time label on UMAP panel.
    time_label_fmt : str, default="t = {:.1f}d"
        Format string for time label.
    time_label_loc : tuple, default=(0.05, 0.95)
        Location of time label in axes coordinates.
    time_label_fontsize : int, default=12
        Font size for time label.
    umap_title : str, optional
        Title for UMAP panel.
    fps : int, default=10
        Frames per second for the GIF.
    duration : float, optional
        Total duration in seconds. If provided, overrides fps.
    hold_frames : int, default=10
        Number of frames to hold at the end before fading.
    fade_frames : int, default=8
        Number of frames for the fade-out transition.
    leading_edge_scale : float, default=2.0
        Size multiplier for leading edge points on UMAP.
    trail_alpha : float, default=0.5
        Alpha multiplier for trail points on UMAP.
    show_progenitor : bool, default=True
        Whether to show progenitor intro frames at the start.
    progenitor_frames : int, default=8
        Number of frames to hold on the progenitor before starting animation.
    progenitor_label : str, default="Progenitor"
        Label text for the progenitor annotation.
    progenitor_s : float, default=80.0
        Point size for the progenitor marker.
    progenitor_color : str, default="dodgerblue"
        Color for the progenitor marker and annotation.
    dpi : int, default=100
        Resolution for each frame.
    return_fig : bool, default=False
        If True, also returns the final frame's (fig, ax_umap, ax_expr) tuple.
    **kwargs
        Additional keyword arguments passed to UMAP scatter.

    Returns
    -------
    str or tuple
        Path to the saved GIF file. If return_fig=True, returns
        (savename, fig, ax_umap, ax_expr) tuple with the final frame.

    Examples
    --------
    >>> import scdiffeq as sdq
    >>> sdq.pl.simulation_expression_gif(
    ...     adata_sim,
    ...     gene="Spi1",
    ...     groupby="final_state",
    ...     expr_cmap={"Mon.": "orange", "Neu.": "#4a7298"}
    ... )
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL (Pillow) is required for GIF creation. Install it with: pip install Pillow")

    # -- Get UMAP coordinates -------------------------------------------------
    umap_coords = adata_sim.obsm[use_key]
    if isinstance(umap_coords, pd.DataFrame):
        x_all = umap_coords.iloc[:, 0].values
        y_all = umap_coords.iloc[:, 1].values
    else:
        x_all = umap_coords[:, 0]
        y_all = umap_coords[:, 1]

    # -- Get time values ------------------------------------------------------
    time_values = adata_sim.obs[time_key].values
    unique_times = np.sort(np.unique(time_values))
    t_min, t_max = unique_times.min(), unique_times.max()

    # -- Compute progenitor mean position (t=0 cells) -------------------------
    t0_mask = time_values == t_min
    progenitor_x = np.mean(x_all[t0_mask])
    progenitor_y = np.mean(y_all[t0_mask])

    # -- Get color values for UMAP --------------------------------------------
    color_values = None

    if color in adata_sim.obs.columns:
        color_values = adata_sim.obs[color].values
    else:
        gene_ids_raw = adata_sim.uns.get(gene_ids_key, {})
        if isinstance(gene_ids_raw, dict):
            gene_names = list(gene_ids_raw.values())
            if color in gene_names:
                gene_idx = gene_names.index(color)
                expr_matrix = adata_sim.obsm[gene_key]
                if isinstance(expr_matrix, pd.DataFrame):
                    color_values = expr_matrix.iloc[:, gene_idx].values
                else:
                    color_values = expr_matrix[:, gene_idx]
        else:
            if isinstance(gene_ids_raw, (pd.Index, pd.Series)):
                gene_ids = gene_ids_raw.tolist()
            elif isinstance(gene_ids_raw, np.ndarray):
                gene_ids = gene_ids_raw.tolist()
            elif isinstance(gene_ids_raw, list):
                gene_ids = gene_ids_raw
            else:
                gene_ids = list(gene_ids_raw) if gene_ids_raw else []

            if color in gene_ids:
                gene_idx = gene_ids.index(color)
                expr_matrix = adata_sim.obsm[gene_key]
                if isinstance(expr_matrix, pd.DataFrame):
                    color_values = expr_matrix.iloc[:, gene_idx].values
                else:
                    color_values = expr_matrix[:, gene_idx]

    if color_values is None:
        color_values = time_values

    vmin, vmax = np.nanmin(color_values), np.nanmax(color_values)

    # -- Get gene expression data for expression panel ------------------------
    gene_ids_raw = adata_sim.uns[gene_ids_key]

    if isinstance(gene_ids_raw, dict):
        gene_names_list = list(gene_ids_raw.values())
        if gene not in gene_names_list:
            preview = gene_names_list[:5]
            raise ValueError(
                f"Gene '{gene}' not found in adata_sim.uns['{gene_ids_key}']. " f"Available genes: {preview}..."
            )
        gene_idx = gene_names_list.index(gene)
    else:
        if isinstance(gene_ids_raw, (pd.Index, pd.Series)):
            gene_ids_list = gene_ids_raw.tolist()
        elif isinstance(gene_ids_raw, np.ndarray):
            gene_ids_list = gene_ids_raw.tolist()
        elif isinstance(gene_ids_raw, list):
            gene_ids_list = gene_ids_raw
        else:
            gene_ids_list = list(gene_ids_raw)

        if gene not in gene_ids_list:
            preview = gene_ids_list[:5] if len(gene_ids_list) >= 5 else gene_ids_list
            raise ValueError(
                f"Gene '{gene}' not found in adata_sim.uns['{gene_ids_key}']. " f"Available genes: {preview}..."
            )
        gene_idx = gene_ids_list.index(gene)

    expr_matrix = adata_sim.obsm[gene_key]
    if isinstance(expr_matrix, pd.DataFrame):
        expression = expr_matrix.iloc[:, gene_idx].values
    else:
        expression = expr_matrix[:, gene_idx]

    group_labels = adata_sim.obs[groupby].values

    df = pd.DataFrame({"expression": expression, "time": time_values, "group": group_labels})
    stats_full = df.groupby(["time", "group"])["expression"].agg(["mean", "std"]).reset_index()

    unique_groups = df["group"].unique()
    if groups is not None:
        plot_groups = [g for g in groups if g in unique_groups]
    else:
        plot_groups = list(unique_groups)

    if expr_cmap is None:
        default_colors = plt.cm.tab10.colors
        expr_cmap = {g: default_colors[i % len(default_colors)] for i, g in enumerate(plot_groups)}

    expr_ymin = (stats_full["mean"] - stats_full["std"]).min()
    expr_ymax = (stats_full["mean"] + stats_full["std"]).max()
    expr_y_margin = (expr_ymax - expr_ymin) * 0.1
    expr_ylim = (expr_ymin - expr_y_margin, expr_ymax + expr_y_margin)

    # -- Setup figure layout --------------------------------------------------
    width_ratios = (1.0, expr_width_scale)
    # For height: use 2 rows, expression panel spans only top portion
    # height_ratios = (expr_height_scale, 1 - expr_height_scale) for the expr column
    # But UMAP spans both rows

    def _create_dual_figure():
        """Create figure with UMAP (full height) and expression (scaled height) panels."""
        fig = plt.figure(figsize=figsize)
        # 2 columns: UMAP gets width_ratios[0], expression gets width_ratios[1]
        # 2 rows for expression column: top row is expr_height_scale, bottom is padding
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=width_ratios,
            height_ratios=(expr_height_scale, 1 - expr_height_scale),
            hspace=0.05,
        )
        # UMAP spans both rows in column 0
        ax_umap = fig.add_subplot(gs[:, 0])
        # Expression panel only in top row of column 1
        ax_expr = fig.add_subplot(gs[0, 1])
        return fig, ax_umap, ax_expr

    # -- Setup background function --------------------------------------------
    if background_fn is None:
        if background_groupby is not None:
            background_fn = lambda adata, ax: _create_expr_grouped_background(
                adata, ax, use_key, background_groupby, background_cmap, background_s, background_inner_s
            )
        else:
            background_fn = lambda adata, ax: _create_expr_default_background(
                adata, ax, use_key, background_s, background_inner_s
            )

    # -- Create frames --------------------------------------------------------
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_idx = 0

        # Progenitor intro frames
        if show_progenitor and progenitor_frames > 0:
            fig, ax_umap, ax_expr = _create_dual_figure()
            _create_expression_progenitor_frame(
                adata_sim,
                ax_umap,
                ax_expr,
                background_fn,
                progenitor_x,
                progenitor_y,
                progenitor_color,
                progenitor_s,
                progenitor_label,
                show_time_label,
                time_label_loc,
                time_label_fmt,
                time_label_fontsize,
                t_min,
                t_max,
                umap_title,
                x_all,
                y_all,
                umap_cmap,
                color,
                expr_ylim,
                x_label,
                y_label,
                gene,
                plot_groups,
                expr_cmap,
                linewidth,
                umap_labels=umap_labels,
            )
            plt.tight_layout()
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            progenitor_img = Image.open(frame_path)
            for _ in range(progenitor_frames):
                frames.append(progenitor_img.copy())
            frame_idx += 1

        # Main animation frames
        for i, t in enumerate(unique_times):
            fig, ax_umap, ax_expr = _create_dual_figure()
            _create_expression_frame(
                adata_sim,
                ax_umap,
                ax_expr,
                t,
                1.0,
                background_fn,
                x_all,
                y_all,
                time_values,
                color_values,
                umap_cmap,
                s,
                alpha,
                trail_alpha,
                leading_edge_scale,
                vmin,
                vmax,
                show_time_label,
                time_label_loc,
                time_label_fmt,
                time_label_fontsize,
                umap_title,
                t_min,
                t_max,
                color,
                stats_full,
                plot_groups,
                expr_cmap,
                linewidth,
                show_std,
                std_alpha,
                expr_ylim,
                x_label,
                y_label,
                gene,
                umap_labels=umap_labels,
                **kwargs,
            )
            plt.tight_layout()
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            frames.append(Image.open(frame_path))
            frame_idx += 1

        # Hold frames at the end
        last_frame = frames[-1]
        for _ in range(hold_frames):
            frames.append(last_frame.copy())

        # Fade out frames
        for fade_i in range(fade_frames):
            fade_alpha = 1.0 - (fade_i + 1) / fade_frames
            fig, ax_umap, ax_expr = _create_dual_figure()
            _create_expression_frame(
                adata_sim,
                ax_umap,
                ax_expr,
                unique_times[-1],
                fade_alpha,
                background_fn,
                x_all,
                y_all,
                time_values,
                color_values,
                umap_cmap,
                s,
                alpha,
                trail_alpha,
                leading_edge_scale,
                vmin,
                vmax,
                show_time_label,
                time_label_loc,
                time_label_fmt,
                time_label_fontsize,
                umap_title,
                t_min,
                t_max,
                color,
                stats_full,
                plot_groups,
                expr_cmap,
                linewidth,
                show_std,
                std_alpha,
                expr_ylim,
                x_label,
                y_label,
                gene,
                umap_labels=umap_labels,
                **kwargs,
            )
            plt.tight_layout()
            frame_path = os.path.join(tmpdir, f"fade_{fade_i:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            frames.append(Image.open(frame_path))

        # -- Create GIF -------------------------------------------------------
        if duration is not None:
            frame_duration = int(duration * 1000 / len(unique_times))
        else:
            frame_duration = int(1000 / fps)

        frames[0].save(
            savename,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
        )

    if return_fig:
        final_fig, final_ax_umap, final_ax_expr = _create_dual_figure()
        _create_expression_frame(
            adata_sim,
            final_ax_umap,
            final_ax_expr,
            unique_times[-1],
            1.0,
            background_fn,
            x_all,
            y_all,
            time_values,
            color_values,
            umap_cmap,
            s,
            alpha,
            trail_alpha,
            leading_edge_scale,
            vmin,
            vmax,
            show_time_label,
            time_label_loc,
            time_label_fmt,
            time_label_fontsize,
            umap_title,
            t_min,
            t_max,
            color,
            stats_full,
            plot_groups,
            expr_cmap,
            linewidth,
            show_std,
            std_alpha,
            expr_ylim,
            x_label,
            y_label,
            gene,
            umap_labels=umap_labels,
            **kwargs,
        )
        plt.tight_layout()
        return savename, final_fig, final_ax_umap, final_ax_expr

    return savename
