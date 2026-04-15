from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

PATHWAY_LENGTH  = 12          # Tmax: maximum funded sessions
FIRST_K         = 5           # k used for the initial spaghetti exploration
SEARCH_K_RANGE  = [2, 3, 4, 5]
CHOSEN_K        = 3           # final k selected after policy analysis
PLATEAU_THRESH  = 0.90        # fraction of total progress defining t*

PALETTE = ["tab:red", "tab:blue", "tab:green", "tab:olive", "tab:pink"]

OUT = Path("output/q2")
OUT.mkdir(parents=True, exist_ok=True)

SESSIONS = np.arange(1, PATHWAY_LENGTH + 1)    # [1, 2, ..., 12]

# ─────────────────────────────────────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def _read_predicted() -> pd.DataFrame:
    raw = pd.read_csv("output/q1/scored_notes.csv")
    raw.columns = raw.columns.str.strip()
    raw["client_id"] = raw["client_id"].astype(str).str.strip()
    raw["session"]   = raw["session"].astype(int)
    raw["score"]     = raw["score"].astype(int)
    return raw


def _read_labeled() -> pd.DataFrame:
    with open("output/q1/evaluated_labeled_results.json", encoding="utf-8") as fh:
        data = json.load(fh)

    entries = []
    for rec in data:
        cid = str(rec["client_id"]).strip()
        for idx, val in enumerate(rec.get("estimated_trajectory_vector", []), start=1):
            entries.append({"client_id": cid, "session": idx, "score": int(val)})

    return pd.DataFrame(entries, columns=["client_id", "session", "score"])


def _pivot_to_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    tbl = df.pivot_table(
        index="client_id", columns="session", values="score", aggfunc="first"
    )
    tbl = tbl.reindex(columns=range(1, PATHWAY_LENGTH)).fillna(0)
    ids = list(tbl.index)
    return tbl.values.astype(float), ids


def _cumulative(raw_matrix: np.ndarray) -> np.ndarray:
    n_clients = raw_matrix.shape[0]
    cum = np.zeros((n_clients, PATHWAY_LENGTH), dtype=float)
    cum[:, 1:] = np.cumsum(raw_matrix, axis=1)
    return cum


def load_all_data() -> Tuple[np.ndarray, List[str]]:
    merged = pd.concat([_read_predicted(), _read_labeled()], ignore_index=True)
    merged = merged.drop_duplicates(subset=["client_id", "session"])
    raw_mx, ids = _pivot_to_matrix(merged)
    return _cumulative(raw_mx), ids

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(data: np.ndarray, n_clusters: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(data)
    model  = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    return model.fit_predict(scaled)

# ─────────────────────────────────────────────────────────────────────────────
# t* COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def stopping_sessions(cum_traj: np.ndarray) -> np.ndarray:
    """Return t* for every client: earliest session where cumulative ≥ 90 % of total."""
    totals  = cum_traj[:, -1]                        # (n,)
    cutoffs = PLATEAU_THRESH * totals                # (n,)
    reached = cum_traj >= cutoffs[:, np.newaxis]     # (n, 12) bool
    # argmax finds the first True per row; default to PATHWAY_LENGTH if none
    first   = np.argmax(reached, axis=1) + 1         # 1-based session index
    # clients with total=0 never reach threshold — keep PATHWAY_LENGTH
    first[totals == 0] = PATHWAY_LENGTH
    return first.astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# NEWSVENDOR SAVINGS CURVES
# ─────────────────────────────────────────────────────────────────────────────

def expected_savings_matrix(t_stars: np.ndarray, assignments: np.ndarray, n: int) -> np.ndarray:
    """Shape (n_clusters, PATHWAY_LENGTH): E[savings](Q) for each Q = 1..12."""
    out = np.zeros((n, PATHWAY_LENGTH))
    for c in range(n):
        group = t_stars[assignments == c]
        for j, Q in enumerate(SESSIONS):
            cdf_val  = np.mean(group <= Q)
            out[c, j] = cdf_val * (PATHWAY_LENGTH - Q)
    return out


def optimal_Q(savings: np.ndarray) -> np.ndarray:
    """Q* = argmax of each row (+1 for 1-based indexing)."""
    return np.argmax(savings, axis=1) + 1

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_spaghetti(
    cum_traj:   np.ndarray,
    assignments: np.ndarray,
    n_clusters:  int,
    save_to:     Path = OUT / "spaghetti_plots.png",
) -> None:
    sessions = SESSIONS
    ncols = 2
    nrows = (n_clusters + 1) // 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows), sharey=True)
    axs = np.array(axs).flatten()

    for hidden in axs[n_clusters:]:
        hidden.set_visible(False)

    for c in range(n_clusters):
        ax     = axs[c]
        colour = PALETTE[c]
        subset = cum_traj[assignments == c]

        for row in subset:
            ax.plot(sessions, row, color=colour, alpha=0.25, linewidth=0.8)

        ax.plot(sessions, subset.mean(axis=0), color=colour, linewidth=2.5,
                label=f"Cluster {c + 1} mean")
        ax.set_title(f"Cluster {c + 1}  (n={len(subset)})", fontsize=11)
        ax.set_xlabel("Session", fontsize=9)
        if c % 2 == 0:
            ax.set_ylabel("Cumulative Progress Score", fontsize=9)
        ax.set_xticks(sessions)
        ax.set_xticklabels([str(s) for s in sessions], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cumulative Progress Trajectories by Cluster (K={n_clusters})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all_savings_plots(
    savings_by_k: Dict[int, np.ndarray],
    q_stars_by_k: Dict[int, np.ndarray],
    save_to: Path = OUT / "all_savings_curves.png",
) -> None:
    k_list = sorted(savings_by_k)
    ncols  = 2
    nrows  = (len(k_list) + 1) // 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=True)
    axs = np.array(axs).flatten()

    for hidden in axs[len(k_list):]:
        hidden.set_visible(False)

    Q_grid = SESSIONS
    for ax, k in zip(axs, k_list):
        curves  = savings_by_k[k]
        q_stars = q_stars_by_k[k]
        for c in range(k):
            col = PALETTE[c]
            ax.plot(Q_grid, curves[c], color=col, linewidth=2,
                    label=f"Cluster {c + 1} (Q*={q_stars[c]})")
            ax.axvline(q_stars[c], color=col, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"K = {k}", fontsize=12)
        ax.set_xlabel("Reassessment session Q", fontsize=10)
        ax.set_xticks(Q_grid)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for row_idx in range(nrows):
        axs[row_idx * ncols].set_ylabel("E[sessions saved per child]", fontsize=10)

    fig.suptitle("Newsvendor E[Savings](Q)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_t_star_histograms(
    t_stars:     np.ndarray,
    assignments: np.ndarray,
    n_clusters:  int,
    save_to:     Path = OUT / "t_star_distributions.png",
) -> None:
    bins = np.arange(0.5, PATHWAY_LENGTH + 1.5, 1)
    fig, axs = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4),
                             sharey=False, sharex=True)
    if n_clusters == 1:
        axs = [axs]

    for c in range(n_clusters):
        ax    = axs[c]
        group = t_stars[assignments == c]
        ax.hist(group, bins=bins, color=PALETTE[c], edgecolor="white", alpha=0.85)
        ax.axvline(group.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={group.mean():.1f}")
        ax.set_title(f"Cluster {c + 1}  (n={len(group)})", fontsize=11)
        ax.set_xlabel("t* (session)", fontsize=9)
        ax.set_xticks(range(1, PATHWAY_LENGTH + 1))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        if c == 0:
            ax.set_ylabel("Count", fontsize=9)

    fig.suptitle("Distribution of t* by Cluster", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_savings_curve_plot(
    savings:    np.ndarray,
    q_stars:    np.ndarray,
    n_clusters: int,
    save_to:    Path = OUT / "savings_curves.png",
) -> None:
    Q_grid = SESSIONS
    fig, ax = plt.subplots(figsize=(8, 5))

    for c in range(n_clusters):
        col = PALETTE[c]
        ax.plot(Q_grid, savings[c], color=col, linewidth=2,
                label=f"Cluster {c + 1} (Q*={q_stars[c]})")
        ax.axvline(q_stars[c], color=col, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Reassessment session Q", fontsize=11)
    ax.set_ylabel("E[sessions saved per child]", fontsize=11)
    ax.set_title("Newsvendor E[Savings](Q) by Cluster", fontsize=13)
    ax.set_xticks(Q_grid)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_grouped_bar(
    t_stars:    np.ndarray,
    assignments: np.ndarray,
    savings:    np.ndarray,
    q_stars:    np.ndarray,
    n_clusters: int,
    save_to:    Path = OUT / "grouped_bar_savings.png",
) -> None:
    optimized = np.array([savings[c, q_stars[c] - 1] for c in range(n_clusters)])
    naive     = np.zeros(n_clusters)
    for c in range(n_clusters):
        grp    = t_stars[assignments == c]
        mean_t = int(np.clip(round(grp.mean()), 1, PATHWAY_LENGTH))
        naive[c] = savings[c, mean_t - 1]

    positions = np.arange(n_clusters)
    bar_w     = 0.35
    fig, ax   = plt.subplots(figsize=(6 + n_clusters, 5))

    b1 = ax.bar(positions - bar_w / 2, optimized, bar_w, label="Optimized Q*",
                color=PALETTE[:n_clusters], edgecolor="white")
    b2 = ax.bar(positions + bar_w / 2, naive, bar_w, label="Baseline mean(t*)",
                color=PALETTE[:n_clusters], edgecolor="black", alpha=0.5)

    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("E[sessions saved per child]", fontsize=11)
    ax.set_title("Optimized vs Baseline Expected Savings per Cluster", fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Cluster {c + 1}" for c in range(n_clusters)], fontsize=10)

    legend_patches = [
        Patch(facecolor="gray", edgecolor="white", label="Optimized Q*"),
        Patch(facecolor="gray", alpha=0.5, edgecolor="white", label="Baseline mean(t*)"),
    ]
    ax.legend(handles=legend_patches, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_policy_table(
    t_stars:     np.ndarray,
    assignments: np.ndarray,
    savings:     np.ndarray,
    q_stars:     np.ndarray,
    n_clusters:  int,
) -> None:
    hdr = (f"{'Cluster':>8}  {'Size':>6}  {'Q*':>4}  "
           f"{'E[saved/child]':>16}  {'% sessions saved':>18}")
    rule = "=" * len(hdr)
    print(f"\n{rule}\n{hdr}\n{rule}")

    total_n   = 0
    total_exp = 0.0
    for c in range(n_clusters):
        mask     = assignments == c
        sz       = int(mask.sum())
        qs       = int(q_stars[c])
        exp_save = savings[c, qs - 1]
        pct      = exp_save / PATHWAY_LENGTH * 100
        print(f"  {c + 1:>6}  {sz:>6}  {qs:>4}  {exp_save:>16.3f}  {pct:>17.1f}%")
        total_n   += sz
        total_exp += sz * exp_save

    overall_pct = total_exp / (total_n * PATHWAY_LENGTH) * 100 if total_n else 0.0
    print(f"{'-' * len(hdr)}")
    print(f"  {'Total':>6}  {'—':>6}  {'—':>4}  {'—':>16}  {overall_pct:>17.1f}%")
    print(rule)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def section_2a(cum_traj: np.ndarray) -> None:
    """Initial spaghetti plot at FIRST_K."""
    labels = fit_kmeans(cum_traj, FIRST_K)
    make_spaghetti(cum_traj, labels, FIRST_K)


def section_2b(cum_traj: np.ndarray) -> None:
    """Sweep K values, print policy tables, and produce combined savings plot."""
    print("\n--- 2b: Optimal Reassessment Policy ---")
    t_stars = stopping_sessions(cum_traj)

    all_savings: Dict[int, np.ndarray] = {}
    all_q:       Dict[int, np.ndarray] = {}

    for k in SEARCH_K_RANGE:
        labels  = fit_kmeans(cum_traj, k)
        sv      = expected_savings_matrix(t_stars, labels, k)
        qs      = optimal_Q(sv)
        all_savings[k] = sv
        all_q[k]       = qs
        print_policy_table(t_stars, labels, sv, qs, k)
        make_spaghetti(
            cum_traj, labels, k,
            save_to=OUT / f"spaghetti_plots_k={k}.png",
        )

    make_all_savings_plots(all_savings, all_q)


def section_2d(cum_traj: np.ndarray, ids: List[str]) -> None:
    """Final three required plots and summary stats for CHOSEN_K."""
    print(f"\n--- 2d: Plots (K={CHOSEN_K}) ---")

    labels  = fit_kmeans(cum_traj, CHOSEN_K)
    t_stars = stopping_sessions(cum_traj)
    savings = expected_savings_matrix(t_stars, labels, CHOSEN_K)
    q_stars = optimal_Q(savings)

    make_t_star_histograms(t_stars, labels, CHOSEN_K)
    make_savings_curve_plot(savings, q_stars, CHOSEN_K)
    make_grouped_bar(t_stars, labels, savings, q_stars, CHOSEN_K)

    # aggregate delta
    base_total = 0.0
    opt_total  = 0.0
    for c in range(CHOSEN_K):
        mask   = labels == c
        n_c    = int(mask.sum())
        mean_t = int(np.clip(round(t_stars[mask].mean()), 1, PATHWAY_LENGTH))
        base_total += savings[c, mean_t - 1] * n_c
        opt_total  += savings[c, q_stars[c] - 1] * n_c

    print(f"\nTotal sessions saved (baseline):  {base_total}")
    print(f"Total sessions saved (optimized): {opt_total}")
    print(f"Delta: {opt_total - base_total:.1f} sessions\n")

    out_df = pd.DataFrame({
        "client_id": ids,
        "cluster":   labels + 1,
        "t_star":    t_stars,
    })
    out_df.to_csv(OUT / "t_star_assignments.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trajectories, client_ids = load_all_data()

    section_2a(trajectories)
    section_2b(trajectories)
    section_2d(trajectories, client_ids)
 