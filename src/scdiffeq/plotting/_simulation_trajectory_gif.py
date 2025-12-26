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


# -- API-facing function: -----------------------------------------------------
def simulation_trajectory_gif(
    adata_sim: anndata.AnnData,
    savename: str = "simulation_trajectory.gif",
    color: str = "t",
    use_key: str = "X_umap",
    time_key: str = "t",
    gene_key: str = "X_gene_inv",
    gene_ids_key: str = "gene_ids",
    figsize: tuple = (6, 6),
    cmap: Union[str, mcolors.Colormap] = "plasma_r",
    s: float = 10.0,
    alpha: float = 0.8,
    background_fn: Optional[Callable] = None,
    background_s: float = 100.0,
    background_inner_s: float = 65.0,
    show_time_label: bool = True,
    time_label_fmt: str = "t = {:.1f}d",
    time_label_loc: tuple = (0.05, 0.95),
    time_label_fontsize: int = 12,
    title: Optional[str] = None,
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
    **kwargs,
) -> str:
    """
    Create a GIF of simulation trajectories growing over UMAP space.

    Parameters
    ----------
    adata_sim : anndata.AnnData
        Simulated data from ``sdq.tl.simulate()``, with UMAP coordinates
        in obsm.
    savename : str, default="simulation_trajectory.gif"
        Output filename for the GIF.
    color : str, default="t"
        What to color points by. Can be column in obs or gene name.
    use_key : str, default="X_umap"
        Key in ``adata_sim.obsm`` containing UMAP coordinates.
    time_key : str, default="t"
        Column in ``adata_sim.obs`` containing time values.
    gene_key : str, default="X_gene_inv"
        Key in ``adata_sim.obsm`` containing gene expression matrix.
    gene_ids_key : str, default="gene_ids"
        Key in ``adata_sim.uns`` containing gene names.
    figsize : tuple, default=(6, 6)
        Figure size (width, height) in inches.
    cmap : str or Colormap, default="plasma_r"
        Colormap for continuous values.
    s : float, default=10.0
        Point size for simulation points.
    alpha : float, default=0.8
        Point transparency.
    background_fn : Callable, optional
        Custom function to plot background. Should accept (adata_sim, ax).
        If None, uses default background (black outline with white fill).
    background_s : float, default=100.0
        Point size for background outer points.
    background_inner_s : float, default=65.0
        Point size for background inner points.
    show_time_label : bool, default=True
        Whether to show time label on each frame.
    time_label_fmt : str, default="t = {:.1f}d"
        Format string for time label.
    time_label_loc : tuple, default=(0.05, 0.95)
        Location of time label in axes coordinates.
    time_label_fontsize : int, default=12
        Font size for time label.
    title : str, optional
        Plot title.
    fps : int, default=10
        Frames per second for the GIF.
    duration : float, optional
        Total duration in seconds. If provided, overrides fps.
    hold_frames : int, default=10
        Number of frames to hold at the end before fading.
    fade_frames : int, default=8
        Number of frames for the fade-out transition.
    leading_edge_scale : float, default=2.0
        Size multiplier for leading edge points (current time step).
    trail_alpha : float, default=0.5
        Alpha multiplier for trail points (older time steps), relative to base alpha.
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
    **kwargs
        Additional keyword arguments passed to scatter.

    Returns
    -------
    str
        Path to the saved GIF file.

    Examples
    --------
    >>> import scdiffeq as sdq
    >>> # Basic usage
    >>> sdq.pl.simulation_trajectory_gif(adata_sim, savename="my_sim.gif")
    >>> # With custom background
    >>> def my_background(adata_sim, ax):
    ...     xu = adata_sim.obsm["X_umap"]
    ...     ax.scatter(xu[:, 0], xu[:, 1], c="lightgray", s=50)
    >>> sdq.pl.simulation_trajectory_gif(adata_sim, background_fn=my_background)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for GIF creation. "
            "Install it with: pip install Pillow"
        )

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

    # -- Get color values -----------------------------------------------------
    color_values = None

    if color in adata_sim.obs.columns:
        color_values = adata_sim.obs[color].values
    else:
        # Try gene name
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
        # Default to time
        color_values = time_values

    # -- Compute global color limits ------------------------------------------
    vmin, vmax = np.nanmin(color_values), np.nanmax(color_values)

    # -- Default background function ------------------------------------------
    def default_background(adata_sim, ax):
        xu = adata_sim.obsm[use_key]
        if isinstance(xu, pd.DataFrame):
            xu = xu.values
        ax.scatter(xu[:, 0], xu[:, 1], c="k", ec="None", rasterized=True, s=background_s)
        ax.scatter(xu[:, 0], xu[:, 1], c="w", ec="None", rasterized=True, s=background_inner_s)

    if background_fn is None:
        background_fn = default_background

    # -- Helper to create progenitor intro frame ------------------------------
    def create_progenitor_frame():
        fig, ax = plt.subplots(figsize=figsize)

        # Plot background
        background_fn(adata_sim, ax)

        # Plot progenitor point
        ax.scatter(
            progenitor_x, progenitor_y,
            c=progenitor_color, s=progenitor_s,
            edgecolors='white', linewidths=1.5,
            zorder=300
        )

        # Add arrow annotation pointing to progenitor
        ax.annotate(
            progenitor_label,
            xy=(progenitor_x, progenitor_y),
            xytext=(progenitor_x + 1.5, progenitor_y + 1.5),
            fontsize=12,
            fontweight='bold',
            color=progenitor_color,
            arrowprops=dict(
                arrowstyle='->',
                color=progenitor_color,
                lw=2,
            ),
            zorder=301
        )

        # Time label
        if show_time_label:
            ax.text(
                time_label_loc[0], time_label_loc[1],
                time_label_fmt.format(t_min),
                transform=ax.transAxes,
                fontsize=time_label_fontsize,
                verticalalignment='top',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        # Formatting
        if title:
            ax.set_title(title, fontsize=12)

        # Remove all spines, ticks, and labels for clean UMAP look
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Fix axis limits to full data range
        ax.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
        ax.set_ylim(y_all.min() - 0.5, y_all.max() + 0.5)

        # Add colorbar with progress indicator at start
        norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, orientation='horizontal', location='bottom')
        cbar.set_label(color, fontsize=10)

        # Progress indicator at start
        cbar_ax = cbar.ax
        cbar_xmin, cbar_xmax = cbar_ax.get_xlim()
        cbar_ax.axvline(x=cbar_xmin, color='dodgerblue', linewidth=2, zorder=10)

        return fig, ax

    # -- Helper to create a single frame --------------------------------------
    def create_frame(t_current, frame_alpha=1.0, is_final=False):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot background
        background_fn(adata_sim, ax)

        # Get points up to current time
        mask = time_values <= t_current
        x = x_all[mask]
        y = y_all[mask]
        c = color_values[mask]
        t_pts = time_values[mask]

        # Plot trail points (not at current time) with faded alpha
        trail_mask = t_pts < t_current
        if np.any(trail_mask):
            ax.scatter(
                x[trail_mask], y[trail_mask],
                c=c[trail_mask], cmap=cmap, s=s,
                alpha=alpha * trail_alpha * frame_alpha,
                vmin=vmin, vmax=vmax, zorder=200,
                edgecolors='none', **kwargs
            )

        # Plot leading edge points (at current time) - larger and full alpha
        leading_mask = t_pts == t_current
        if np.any(leading_mask):
            scatter = ax.scatter(
                x[leading_mask], y[leading_mask],
                c=c[leading_mask], cmap=cmap,
                s=s * leading_edge_scale,
                alpha=frame_alpha,
                vmin=vmin, vmax=vmax, zorder=202,
                edgecolors='none', **kwargs
            )
        else:
            # Need scatter for colorbar reference
            scatter = ax.scatter(
                x, y, c=c, cmap=cmap, s=s,
                alpha=alpha * trail_alpha * frame_alpha,
                vmin=vmin, vmax=vmax, zorder=200,
                edgecolors='none', **kwargs
            )

        # Time label
        if show_time_label:
            ax.text(
                time_label_loc[0], time_label_loc[1],
                time_label_fmt.format(t_current),
                transform=ax.transAxes,
                fontsize=time_label_fontsize,
                verticalalignment='top',
                fontweight='bold',
                alpha=frame_alpha,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8 * frame_alpha)
            )

        # Formatting
        if title:
            ax.set_title(title, fontsize=12)

        # Remove all spines, ticks, and labels for clean UMAP look
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Fix axis limits to full data range
        ax.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
        ax.set_ylim(y_all.min() - 0.5, y_all.max() + 0.5)

        # Add colorbar with progress indicator
        # Create a ScalarMappable for the colorbar
        norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, orientation='horizontal', location='bottom')
        cbar.set_label(color, fontsize=10)

        # Add progress indicator on colorbar
        cbar_ax = cbar.ax
        # Get the actual x-limits of the colorbar axis (horizontal orientation)
        cbar_xmin, cbar_xmax = cbar_ax.get_xlim()
        # Map current time to colorbar position
        progress_x = cbar_xmin + (t_current - t_min) / (t_max - t_min) * (cbar_xmax - cbar_xmin) if t_max > t_min else cbar_xmax
        # Draw a marker on the colorbar at current time position
        cbar_ax.axvline(x=progress_x, color='dodgerblue', linewidth=2, zorder=10)

        return fig, ax

    # -- Create frames --------------------------------------------------------
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_idx = 0

        # Progenitor intro frames
        if show_progenitor and progenitor_frames > 0:
            fig, ax = create_progenitor_frame()
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            progenitor_img = Image.open(frame_path)
            for _ in range(progenitor_frames):
                frames.append(progenitor_img.copy())
            frame_idx += 1

        # Main animation frames
        for i, t in enumerate(unique_times):
            fig, ax = create_frame(t)

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
            fig, ax = create_frame(unique_times[-1], frame_alpha=fade_alpha)

            frame_path = os.path.join(tmpdir, f"fade_{fade_i:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            frames.append(Image.open(frame_path))

        # -- Create GIF -------------------------------------------------------
        if duration is not None:
            frame_duration = int(duration * 1000 / len(unique_times))  # Base on main frames
        else:
            frame_duration = int(1000 / fps)

        frames[0].save(
            savename,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0
        )

    return savename
