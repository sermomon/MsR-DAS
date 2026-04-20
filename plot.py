
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from processing import spectral_decomposition, normalize_band

def plot_rgb(
    tr: np.ndarray,
    fs: float,
    bands: list[list[float]],
    dist: np.ndarray | None = None,
    time: np.ndarray | None = None,
    order: int = 5,
    perc: float = 90,
    band_labels: list[str] | None = None,
    band_colors: list[str] | None = None,
    show_legend: bool = True,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Distance (m)",
    label_fontsize: float = 12,
    tick_fontsize: float = 10,
    title_fontsize: float = 14,
    legend_fontsize: float = 10,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot a RGB composite image from DAS strain data using spectral decomposition.

    Applies bandpass filtering to decompose the input strain data into three
    frequency bands, normalizes each band independently, and displays them as
    the R, G, and B channels of a composite image.

    Parameters
    ----------
    tr : np.ndarray
        2D DAS strain array of shape (n_channels, n_samples).
    fs : float
        Sampling frequency [Hz].
    bands : list of [f_low, f_high]
        Exactly three frequency bands [Hz] mapped to [R, G, B] respectively.
        Each band must satisfy 0 < f_low < f_high < fs / 2.
    dist : np.ndarray, optional
        1D array of channel distances [m] of shape (n_channels,). Used to
        label the y-axis. If None, channel indices are used instead.
    time : np.ndarray, optional
        1D array of time values [s] of shape (n_samples,). Used to label the
        x-axis. If None, sample indices are used instead.
    order : int, optional
        Butterworth filter order (default: 5).
    perc : float, optional
        Percentile used for per-band normalization, in the range (0, 100]
        (default: 90).
    band_labels : list of str, optional
        Custom labels for the three bands shown in the legend. Defaults to
        ['B1: {f_low}–{f_high} Hz', ...].
    band_colors : list of str, optional
        Colors used for the legend patches, mapped to [R, G, B] channels.
        Defaults to ['red', 'green', 'blue'].
    show_legend : bool, optional
        Whether to display the band legend (default: True).
    figsize : tuple of float, optional
        Figure size in inches as (width, height) (default: (12, 8)).
    title : str, optional
        Plot title. If None, no title is shown.
    xlabel : str, optional
        X-axis label (default: 'Time (s)').
    ylabel : str, optional
        Y-axis label (default: 'Distance (m)').
    label_fontsize : float, optional
        Font size for axis labels (default: 12).
    tick_fontsize : float, optional
        Font size for axis tick labels (default: 10).
    title_fontsize : float, optional
        Font size for the plot title (default: 14).
    legend_fontsize : float, optional
        Font size for the legend (default: 10).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot into. If None, a new figure and axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        The Figure and Axes objects for further customization.
    """

    if len(bands) != 3:
        raise ValueError(f"bands must contain exactly 3 entries, got {len(bands)}.")
    if band_labels is not None and len(band_labels) != 3:
        raise ValueError(f"band_labels must have exactly 3 entries, got {len(band_labels)}.")
    if band_colors is not None and len(band_colors) != 3:
        raise ValueError(f"band_colors must have exactly 3 entries, got {len(band_colors)}.")

    if band_labels is None:
        band_labels = [f"B{i+1}: {b[0]}–{b[1]} Hz" for i, b in enumerate(bands)]
    if band_colors is None:
        band_colors = ["red", "green", "blue"]

    tr_stack = spectral_decomposition(tr, fs, bands, order=order, stack=True)
    norm_stack = normalize_band(tr_stack, perc=perc)          # (3, n_channels, n_samples)
    rgb = np.stack([norm_stack[0], norm_stack[1], norm_stack[2]], axis=-1)  # (n_channels, n_samples, 3)

    t0, t1 = (time[0],  time[-1])  if time is not None else (0, tr.shape[1] - 1)
    d0, d1 = (dist[0], dist[-1]) if dist is not None else (0, tr.shape[0] - 1)
    extent = [t0, t1, d1, d0]  # [left, right, bottom, top] — d1 bottom so depth increases downward

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.imshow(rgb[::-1], aspect="auto", extent=extent, interpolation="none")

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    if show_legend:
        patches = [
            mpatches.Patch(color=c, label=l)
            for c, l in zip(band_colors, band_labels)
        ]
        ax.legend(handles=patches, fontsize=legend_fontsize, loc="upper right")

    fig.tight_layout()

    return fig, ax