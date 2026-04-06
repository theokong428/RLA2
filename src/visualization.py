"""
Unified visualization module for Q1b/Q1c/Q2.

Figure numbering follows plan.md §6:
  Fig-B1/B2        — Q1b construction results
  Fig-C1..C6       — Q1c improvement results and comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from solution import Solution, CostBreakdown, evaluate, robot_distances
from data_loader import ProblemData

# Cartopy for Antarctic map projection
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ── colour palette ──────────────────────────────────────────────────────────
_ROBOT_CLR = "#4A90D9"
_STATION_CLR = "#E74C3C"
_TRANSPORT_CLR = "#E74C3C"
_LINE_CLR = "#E74C3C"
_METHODS_CLR = {"Q1b": "#3498DB", "SA": "#E67E22", "ALNS": "#2ECC71"}

# Charger count colormap: low-saturation warm palette (light peach → muted red)
_CHARGER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "chargers", ["#FDE8D0", "#F5C89A", "#E8A87C", "#D4836B", "#B85C5C", "#8B3A3A"])
_CHARGER_NORM = mcolors.Normalize(vmin=1, vmax=8)

# Map projection
_PROJ = ccrs.SouthPolarStereo()
_DATA_CRS = ccrs.PlateCarree()


# ============================================================================
# 1. Solution map  (Fig-B1, C1, C2)
# ============================================================================

def plot_solution_map(sol: Solution, data: ProblemData, title: str = "",
                      ax=None, save_path: str | None = None,
                      fig_size: float = 14):
    """Plot robot positions, stations, and transport lines on Antarctic map.

    - South Polar Stereographic projection with coastlines.
    - Only transport-required assignment lines are drawn (red).
    - Station colour = charger count (low-saturation gradient + colorbar).
    - Square figure for balanced aspect ratio.

    Args:
        sol:       Solution to visualise.
        data:      Problem data.
        title:     Plot title.
        ax:        Optional cartopy GeoAxes. Created if None.
        save_path: If given, save figure to this path.
        fig_size:  Side length of the square figure (inches).
    """
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(fig_size, fig_size))
        ax = fig.add_subplot(111, projection=_PROJ)
        created_fig = True
    else:
        fig = ax.figure

    # ── base map ──────────────────────────────────────────────────────────
    ax.set_extent([-180, 180, -90, -60], crs=_DATA_CRS)
    ax.add_feature(cfeature.LAND, facecolor="#F0F0F0", edgecolor="none",
                   zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#AAAAAA",
                   zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4,
                       color="#CCCCCC")
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # ── transport lines (only robots needing transport) ───────────────────
    transport_idx = np.where(sol.needs_transport)[0]
    n_lines = len(transport_idx)
    # Adaptive styling: thinner & fainter when many lines
    lw = 0.25 if n_lines > 200 else (0.45 if n_lines > 50 else 0.7)
    la = 0.12 if n_lines > 200 else (0.25 if n_lines > 50 else 0.45)
    for i in transport_idx:
        j = sol.assignments[i]
        ax.plot(
            [data.robot_lon[i], sol.station_lon[j]],
            [data.robot_lat[i], sol.station_lat[j]],
            color=_LINE_CLR, linewidth=lw, alpha=la, zorder=2,
            transform=_DATA_CRS,
        )

    # ── robots ────────────────────────────────────────────────────────────
    normal = ~sol.needs_transport
    ax.scatter(data.robot_lon[normal], data.robot_lat[normal],
               s=5, c=_ROBOT_CLR, alpha=0.5, zorder=3, label="Robot",
               transform=_DATA_CRS)
    ax.scatter(data.robot_lon[sol.needs_transport],
               data.robot_lat[sol.needs_transport],
               s=10, c=_TRANSPORT_CLR, marker="x", linewidths=0.6,
               zorder=4, label="Needs transport",
               transform=_DATA_CRS)

    # ── stations (colour = charger count) ─────────────────────────────────
    n_st = sol.n_stations
    st_size = 30 if n_st > 40 else (45 if n_st > 15 else 60)
    charger_colors = _CHARGER_CMAP(_CHARGER_NORM(sol.n_chargers))
    ax.scatter(sol.station_lon, sol.station_lat, s=st_size,
               c=charger_colors, marker="^", edgecolors="#555555",
               linewidths=0.5, zorder=5,
               transform=_DATA_CRS)

    # ── colorbar for charger count ────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=_CHARGER_CMAP, norm=_CHARGER_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=20)
    cbar.set_label("Chargers per station", fontsize=9)
    cbar.set_ticks(range(1, 9))

    # ── legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ROBOT_CLR,
               markersize=5, label="Robot"),
        Line2D([0], [0], marker="x", color="w", markerfacecolor=_TRANSPORT_CLR,
               markeredgecolor=_TRANSPORT_CLR, markersize=6,
               label="Needs transport"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#D4836B",
               markeredgecolor="#333333", markersize=7, label="Station"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              framealpha=0.8)

    # ── title ─────────────────────────────────────────────────────────────
    ax.set_title(title or "Solution Map", fontsize=14, fontweight="bold",
                 pad=12)

    if save_path and created_fig:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
    return fig, ax


# ============================================================================
# 2. Before / After comparison  (Fig-C3)
# ============================================================================

def plot_solution_comparison(sol_before: Solution, sol_after: Solution,
                             data: ProblemData,
                             titles: tuple[str, str] = ("Before", "After"),
                             save_path: str | None = None):
    """Top-bottom solution maps on Antarctic projection.

    Args:
        sol_before, sol_after: Two solutions to compare.
        data:      Problem data.
        titles:    Subplot titles.
        save_path: If given, save figure.
    """
    fig = plt.figure(figsize=(14, 26))

    ax1 = fig.add_subplot(211, projection=_PROJ)
    plot_solution_map(sol_before, data, title=titles[0], ax=ax1)

    ax2 = fig.add_subplot(212, projection=_PROJ)
    plot_solution_map(sol_after, data, title=titles[1], ax=ax2)

    fig.tight_layout(pad=2.0)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
    return fig


# ============================================================================
# 3. Cost breakdown — single solution  (Fig-B2)
# ============================================================================

def plot_cost_breakdown(cost: CostBreakdown, title: str = "",
                        ax=None, save_path: str | None = None):
    """Stacked bar chart of cost components.

    Args:
        cost:      CostBreakdown to plot.
        title:     Plot title.
        ax:        Optional Axes.
        save_path: If given, save figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    labels = ["Build", "Maintenance", "Charging", "Transport"]
    values = [cost.build, cost.maintenance, cost.charging, cost.transport]
    colours = ["#3498DB", "#2ECC71", "#E67E22", "#E74C3C"]

    bottom = 0.0
    for label, val, clr in zip(labels, values, colours):
        ax.bar("Total Cost", val, bottom=bottom, color=clr, label=label,
               edgecolor="white", linewidth=0.5)
        if val / cost.total > 0.05:
            ax.text(0, bottom + val / 2, f"\u00a3{val:,.0f}",
                    ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += val

    ax.set_ylabel("Cost (\u00a3)")
    ax.set_title(title or "Cost Breakdown")
    ax.legend(loc="upper right", fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ============================================================================
# 4. Cost comparison — multiple methods  (Fig-C4)
# ============================================================================

def plot_cost_comparison(costs: dict[str, CostBreakdown],
                         title: str = "Method Comparison",
                         save_path: str | None = None):
    """Grouped bar chart comparing cost components across methods.

    Args:
        costs:     {"method_name": CostBreakdown, ...}
        title:     Plot title.
        save_path: If given, save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(costs.keys())
    components = ["build", "maintenance", "charging", "transport"]
    comp_labels = ["Build", "Maintenance", "Charging", "Transport"]
    colours = ["#3498DB", "#2ECC71", "#E67E22", "#E74C3C"]

    x = np.arange(len(methods))
    width = 0.18

    for k, (comp, label, clr) in enumerate(zip(components, comp_labels, colours)):
        vals = [getattr(costs[m], comp) for m in methods]
        ax.bar(x + k * width, vals, width, label=label, color=clr,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Cost (\u00a3)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # Annotate totals above each group
    for i, m in enumerate(methods):
        total = costs[m].total
        ax.text(i + 1.5 * width, total * 0.02 + max(
            getattr(costs[m], c) for c in components),
            f"Total: \u00a3{total:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ============================================================================
# 5. Convergence curve  (Fig-C5)
# ============================================================================

def plot_convergence(histories: dict[str, np.ndarray],
                     title: str = "Convergence",
                     xlabel: str = "Iteration",
                     save_path: str | None = None):
    """Overlay convergence curves for one or more methods.

    Args:
        histories: {"method_name": 1-D array of objective values, ...}
        title:     Plot title.
        xlabel:    X-axis label (could be "Iteration" or "Time (s)").
        save_path: If given, save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, vals in histories.items():
        clr = _METHODS_CLR.get(name, None)
        ax.plot(vals, label=name, color=clr, linewidth=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Objective Value (\u00a3)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ============================================================================
# 6. ALNS operator weight evolution  (Fig-C6)
# ============================================================================

def plot_operator_weights(weight_history: dict[str, np.ndarray],
                          title: str = "ALNS Operator Weights",
                          save_path: str | None = None):
    """Plot how ALNS operator weights evolve over iterations.

    Args:
        weight_history: {"operator_name": 1-D array of weights, ...}
                        Operators should be prefixed with category
                        (e.g. "D: Random-Remove", "R: Greedy-Insert").
        title:     Plot title.
        save_path: If given, save figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    categories = {
        "D": ("Destroy", axes[0]),
        "R": ("Repair", axes[1]),
        "L": ("Local", axes[2]),
    }

    for name, weights in weight_history.items():
        prefix = name.split(":")[0].strip()
        cat_label, ax = categories.get(prefix, ("Other", axes[2]))
        ax.plot(weights, label=name, linewidth=1.0)

    for prefix, (cat_label, ax) in categories.items():
        ax.set_ylabel("Weight")
        ax.set_title(cat_label)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Segment (weight update period)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


# ============================================================================
# 7. K-scan curve  (Fig-B3, C7)
# ============================================================================

def plot_k_scan(k_vals, cost_matrix, method_name: str = "Q1b",
                title: str = "Station Count Sensitivity",
                save_path: str | None = None):
    """Plot k vs cost with breakdown, showing best-of-seeds and U-shape.

    Args:
        k_vals:      1-D array of k values tested.
        cost_matrix: dict with keys "total", "build", "maintenance",
                     "charging", "transport".  Each value is (len(k_vals),)
                     array of the *best-of-seeds* cost for that k.
        method_name: Label for legend.
        title:       Plot title.
        save_path:   If given, save figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # ── Top: total cost U-curve ───────────────────────────────────────────
    clr = _METHODS_CLR.get(method_name, "#333333")
    ax1.plot(k_vals, cost_matrix["total"], "o-", color=clr, linewidth=2,
             markersize=6, label=f"{method_name} total cost", zorder=3)

    best_idx = np.argmin(cost_matrix["total"])
    best_k = k_vals[best_idx]
    best_cost = cost_matrix["total"][best_idx]
    ax1.axvline(best_k, color=clr, linestyle="--", alpha=0.4, zorder=1)
    ax1.scatter([best_k], [best_cost], s=120, color=clr, edgecolors="k",
                linewidths=1.5, zorder=4)
    ax1.annotate(f"k*={best_k}\n\u00a3{best_cost:,.0f}",
                 xy=(best_k, best_cost),
                 xytext=(15, 15), textcoords="offset points",
                 fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="k"))

    ax1.set_ylabel("Total Cost (\u00a3)")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Bottom: stacked cost breakdown ────────────────────────────────────
    comp_keys = ["build", "maintenance", "charging", "transport"]
    comp_labels = ["Build", "Maintenance", "Charging", "Transport"]
    colours = ["#3498DB", "#2ECC71", "#E67E22", "#E74C3C"]

    bottom = np.zeros(len(k_vals))
    for key, label, c in zip(comp_keys, comp_labels, colours):
        vals = np.array(cost_matrix[key])
        ax2.fill_between(k_vals, bottom, bottom + vals, alpha=0.7,
                         color=c, label=label)
        bottom += vals

    ax2.axvline(best_k, color="k", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Number of Stations (k)")
    ax2.set_ylabel("Cost (\u00a3)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, (ax1, ax2)
