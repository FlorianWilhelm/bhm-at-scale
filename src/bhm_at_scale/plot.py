from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from .utils import make_intervals


def _data2axes_trans(ax, x, y=0):
    x, y = ax.transData.transform((x, y))
    x, y = ax.transAxes.inverted().transform((x, y))
    return x, y


def _add_axes_sep(ax, p_x):
    p_x, _ = _data2axes_trans(ax, p_x)
    line = plt.Line2D([p_x, p_x], [-.07, 0.], transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def plot_sales_preds(df,
                     hlines: Optional[List[Tuple[str, float, dict]]] = None,
                     lines: Optional[List[Tuple[np.array, str, dict]]] = None,
                     intervals: Optional[List[Tuple[np.array, np.array, dict]]] = None):
    """Nicely plot sales and predictions of a store

    Args:
        df: dataframe indexed by store id
        hlines: list of horizontal lines as tuple of (label, value, style dict)
        lines: lines like predictions as tuple of (points, label, style dict)
        intervals: error intervals as tuple of (lower points, upper points, style dict)
    """
    if hlines is None:
        hlines = []
    if lines is None:
        lines = []
    if intervals is None:
        intervals = []

    fig, ax1 = plt.subplots()
    x = np.arange(len(df))

    g1 = sns.scatterplot(x=x, y=df['Customers'], ax=ax1, color='red', label='Customers', marker='x')
    ax1.set_ylabel('Customers')

    g1.set_xticklabels(g1.get_xticklabels(), rotation=90)

    ax1.set_ylim([0., 1.05 * ax1.get_ylim()[1]])
    ax2 = ax1.twinx()
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Promo 1
    for i, (a, b) in enumerate(make_intervals(df['Promo'].to_numpy())):
        label = i * '_' + 'Promo 1'
        ax2.axvspan(a - 0.5, b - 0.5, facecolor='#e2e67e', alpha=0.4, zorder=1, hatch='.', edgecolor='black',
                    label=label)

    # Promo 2
    for i, (a, b) in enumerate(make_intervals(df['Promo2'].to_numpy())):
        label = i * '_' + 'Promo 2'
        ax2.axvspan(a - 0.5, b - 0.5, facecolor='#2ca02c', alpha=0.05, zorder=1, hatch='///', edgecolor='black',
                    label=label)

    # StateHoliday
    for i, (a, b) in enumerate(make_intervals(df['StateHoliday'].to_numpy())):
        label = i * '_' + 'StateHoliday'
        ax2.axvspan(a - 0.5, b - 0.5, facecolor='#5da7e3', alpha=0.2, zorder=1, hatch='-', edgecolor='black',
                    label=label)

    # SchoolHoliday
    for i, (a, b) in enumerate(make_intervals(df['SchoolHoliday'].to_numpy())):
        label = i * '_' + 'SchoolHoliday'
        ax2.axvspan(a - 0.5, b - 0.5, facecolor='#2ca091', alpha=0.2, zorder=1, hatch='|', edgecolor='black',
                    label=label)

    g2 = sns.barplot(x=x, y=df['Sales'], color='b', ax=ax2, alpha=0.9)

    for y, label, style in lines:
        ax2.plot(x, y, label=label, **style)

    for label, value, style in hlines:
        plt.axhline(value, label=label, **style)

    for lower, upper, style in intervals:
        plt.fill_between(x, lower, upper, **style)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    if label := ax2.get_legend():
        label.remove()

    ax2.set_ylabel('Sales')

    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.margins(0.02)
    for t in range(len(df) - 1):
        _add_axes_sep(ax1, t + 0.5)

    plt.xticks(x, df['Date'])
    plt.tight_layout()
    ax1.grid(False)

    g1.set_zorder(ax2.get_zorder() + 1)
    g1.patch.set_visible(False)

    plt.show()
    return g2


class PlotStore(object):
    """Functor to plot sales history and prediction of a single store"""
    def __init__(self, *, df_orig: pd.DataFrame, df_preds: pd.DataFrame):
        self.df_preds = df_preds.set_index(['StoreId', 'metric'])
        self.df_orig = df_orig.set_index('StoreId')

    def __call__(self, store_id: int, selector: slice):
        orig = self.df_orig.loc[store_id].iloc[selector]
        store_df = self.df_preds.loc[store_id]
        preds = store_df.loc['mean'][selector].to_numpy()
        lowest = store_df.loc['5%'][selector].to_numpy()
        lower = store_df.loc['25%'][selector].to_numpy()
        higher = store_df.loc['75%'][selector].to_numpy()
        highest = store_df.loc['95%'][selector].to_numpy()

        lines = [(preds, 'Pred. mean', dict(linestyle='--', color='navy', marker='o', linewidth=3))]
        intervals = [(lower, higher, dict(alpha=0.3, color='navy')),
                     (lowest, highest, dict(alpha=0.3, color='navy'))]
        return plot_sales_preds(orig, lines=lines, intervals=intervals)


def plot_densities(df: pd.DataFrame, xlim: Optional[Tuple[float, float]] = None):
    """Plot the facetted densities

    Args:
        df: Dataframe
        xlim: xlim limits
    """
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="variable", hue="variable", aspect=15, height=1.0, palette=pal)

    g.map(plt.axhline, y=0, lw=2, clip_on=False, color='lightgrey')
    g.map(sns.kdeplot, "value", clip_on=False, shade=True, alpha=1, lw=1.5)

    def label(_, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "value")
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.xlabel('')
    if xlim is not None:
        plt.xlim(*xlim)
