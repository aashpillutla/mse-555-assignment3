from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

CHOSEN_K       = 3
PATHWAY_LENGTH = 12
PALETTE        = ["tab:red", "tab:blue", "tab:green"]
INPUT_FEATURES = ["age_years", "complexity_score", "gender", "referral_reason"]

OUT = Path("output/q3")
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data() -> pd.DataFrame:
    intake       = pd.read_csv("client_features.csv")
    cluster_info = pd.read_csv("output/q2/t_star_assignments.csv")

    for frame in (intake, cluster_info):
        frame["client_id"] = frame["client_id"].astype(str).str.strip()

    return intake.merge(
        cluster_info[["client_id", "cluster", "t_star"]],
        on="client_id", how="inner",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3A: EDA — INTAKE FEATURES BY TRAJECTORY TYPE
# ─────────────────────────────────────────────────────────────────────────────

def section_3a(df: pd.DataFrame) -> None:
    print("\n--- 3a: Exploring Intake Features by Trajectory Type ---")

    group_ids   = sorted(df["cluster"].unique())
    tick_labels = [f"Cluster {g}" for g in group_ids]

    _plot_box_distributions(df, group_ids, tick_labels)
    _plot_stacked_categoricals(df, group_ids, tick_labels)
    _summarise_features(df, group_ids)


def _plot_box_distributions(
    df: pd.DataFrame, group_ids: list, tick_labels: list
) -> None:
    """Box plots for age and complexity score by cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, ylabel, title in [
        (axes[0], "age_years",        "Age (years)",      "Age at Intake by Cluster"),
        (axes[1], "complexity_score", "Complexity Score", "Complexity Score by Cluster"),
    ]:
        grouped = [df.loc[df["cluster"] == g, col].values for g in group_ids]
        bp = ax.boxplot(
            grouped, positions=group_ids, patch_artist=True,
            widths=0.5, medianprops={"color": "black", "linewidth": 2},
        )
        for patch, colour in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        ax.set_xticks(group_ids)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Cluster", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Continuous Intake Features by Trajectory Cluster", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "eda_continuous_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_stacked_categoricals(
    df: pd.DataFrame, group_ids: list, tick_labels: list
) -> None:
    """Stacked bar charts for referral reason and gender proportions."""
    reason_tab = pd.crosstab(df["cluster"], df["referral_reason"], normalize="index")
    gender_tab = pd.crosstab(df["cluster"], df["gender"],          normalize="index")

    fig, (ax_r, ax_g) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(group_ids))

    reason_palette = ["tab:cyan", "tab:olive", "tab:purple", "tab:pink"]
    bottom = np.zeros(len(group_ids))
    for col, colour in zip(reason_tab.columns, reason_palette):
        heights = reason_tab[col].values
        ax_r.bar(x, heights, bottom=bottom, label=col, color=colour, edgecolor="white", width=0.5)
        for xi, (b, h) in enumerate(zip(bottom, heights)):
            if h > 0.06:
                ax_r.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
        bottom += heights

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(tick_labels)
    ax_r.set_ylabel("Proportion", fontsize=10)
    ax_r.set_title("Referral Reason by Cluster", fontsize=11)
    ax_r.legend(title="Referral Reason", fontsize=9, loc="upper right")
    ax_r.set_ylim(0, 1.05)

    gender_palette = ["tab:pink", "tab:blue"]
    bottom = np.zeros(len(group_ids))
    for col, colour in zip(gender_tab.columns, gender_palette):
        heights = gender_tab[col].values
        ax_g.bar(x, heights, bottom=bottom, label=col, color=colour, edgecolor="white", width=0.5)
        for xi, (b, h) in enumerate(zip(bottom, heights)):
            if h > 0.06:
                ax_g.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
        bottom += heights

    ax_g.set_xticks(x)
    ax_g.set_xticklabels(tick_labels)
    ax_g.set_ylabel("Proportion", fontsize=10)
    ax_g.set_title("Gender by Cluster", fontsize=11)
    ax_g.legend(title="Gender", fontsize=9, loc="upper right")
    ax_g.set_ylim(0, 1.05)

    fig.suptitle("Categorical Intake Features by Trajectory Cluster", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "eda_categorical_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summarise_features(df: pd.DataFrame, group_ids: list) -> None:
    """Print descriptive stats using pandas groupby — clean and readable."""
    print("\nContinuous features by cluster:")
    summary = (
        df.groupby("cluster")[["age_years", "complexity_score"]]
        .describe()
        .round(2)
    )
    print(summary.to_string())

    print("\nReferral reason proportions by cluster:")
    print(
        pd.crosstab(df["cluster"], df["referral_reason"], normalize="index")
        .round(3)
        .to_string()
    )

    print("\nGender proportions by cluster:")
    print(
        pd.crosstab(df["cluster"], df["gender"], normalize="index")
        .round(3)
        .to_string()
    )
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 3B: CLASSIFICATION — PREDICT TRAJECTORY GROUP
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough",                              ["age_years", "complexity_score"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["gender", "referral_reason"]),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


CLASSIFIERS = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Random Forest",       RandomForestClassifier(n_estimators=200, random_state=42)),
]

CM_FILENAMES = {
    "Logistic Regression": "confusion_matrix_logreg.png",
    "Random Forest":       "confusion_matrix_rf.png",
}


def section_3b(df: pd.DataFrame) -> Pipeline:
    print("\n--- 3b: Training Classifiers for Trajectory Group Prediction ---")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df[INPUT_FEATURES], df["cluster"],
        test_size=0.2, stratify=df["cluster"], random_state=42,
    )

    class_labels = [1, 2, 3]
    class_names  = [f"Cluster {k}" for k in class_labels]

    records: list[dict] = []
    for name, clf in CLASSIFIERS:
        pipe  = _make_pipeline(clf)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        cm    = confusion_matrix(y_te, preds, labels=class_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
            ax=ax, colorbar=True, cmap="Greens"
        )
        ax.set_title(f"Confusion Matrix — {name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / CM_FILENAMES[name], dpi=150, bbox_inches="tight")
        plt.close(fig)

        records.append({
            "name": name, "pipe": pipe, "preds": preds,
            "accuracy": accuracy_score(y_te, preds), "cm": cm,
        })

    _report_results(records, y_te, class_labels, class_names, len(X_tr), len(X_te))

    # return the logistic regression pipeline
    return next(r["pipe"] for r in records if r["name"] == "Logistic Regression")


def _report_results(
    records: list[dict],
    y_te,
    class_labels: list,
    class_names:  list,
    n_tr: int,
    n_te: int,
) -> None:
    print(f"\n{'Model':<28}  {'Train N':>8}  {'Test N':>7}  {'Accuracy':>9}")
    print("-" * 58)
    for r in records:
        print(f"{r['name']:<28}  {n_tr:>8}  {n_te:>7}  {r['accuracy']:>8.1%}")

    for r in records:
        print(f"\n--- {r['name']} ---")
        print(classification_report(
            y_te, r["preds"],
            labels=class_labels, target_names=class_names,
        ))

# ─────────────────────────────────────────────────────────────────────────────
# 3C: WAITLIST CAPACITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_policy(t_vals: np.ndarray) -> tuple[int, float, float]:
    """Return (Q*, F(Q*), E[savings at Q*]) for a cluster's t* distribution."""
    Q_grid   = np.arange(1, PATHWAY_LENGTH + 1)
    cdf      = np.mean(t_vals[:, np.newaxis] <= Q_grid[np.newaxis, :], axis=0)
    savings  = cdf * (PATHWAY_LENGTH - Q_grid)
    best     = int(np.argmax(savings))
    return int(Q_grid[best]), float(cdf[best]), float(savings[best])


def section_3c(pipe: Pipeline) -> None:
    print("\n--- 3c: Waitlist Capacity Estimation ---")

    waitlist = pd.read_csv("waitlist.csv")
    waitlist["client_id"] = waitlist["client_id"].astype(str).str.strip()
    waitlist["predicted_cluster"] = pipe.predict(waitlist[INPUT_FEATURES])

    assignments = pd.read_csv("output/q2/t_star_assignments.csv")

    # build policy per cluster
    policy: dict[int, dict] = {}
    for c in sorted(assignments["cluster"].unique()):
        t_vals     = assignments.loc[assignments["cluster"] == c, "t_star"].values
        q, f, _    = _cluster_policy(t_vals)
        # E[sessions delivered] = F * Q + (1-F) * T_max
        e_del      = f * q + (1 - f) * PATHWAY_LENGTH
        policy[c]  = {"q_star": q, "f": f, "e_delivered": e_del}

    waitlist["q_star"]      = waitlist["predicted_cluster"].map({c: v["q_star"]      for c, v in policy.items()})
    waitlist["e_delivered"] = waitlist["predicted_cluster"].map({c: v["e_delivered"] for c, v in policy.items()})

    _report_capacity(waitlist, policy)

    waitlist[["client_id", "predicted_cluster", "q_star", "e_delivered"]].to_csv(
        OUT / "waitlist_predictions.csv", index=False,
    )


def _report_capacity(waitlist: pd.DataFrame, policy: dict) -> None:
    n_total  = len(waitlist)
    baseline = n_total * PATHWAY_LENGTH

    rows = []
    for c, p in policy.items():
        n_c     = int((waitlist["predicted_cluster"] == c).sum())
        total_c = n_c * p["e_delivered"]
        rows.append({
            "Cluster":         f"Cluster {c}",
            "N predicted":     n_c,
            "Q*":              p["q_star"],
            "F(Q*)":           round(p["f"], 3),
            "E[sessions]":     round(p["e_delivered"], 2),
            "Total sessions":  round(total_c, 1),
        })

    summary_df = pd.DataFrame(rows).set_index("Cluster")
    print("\nWaitlist capacity by cluster:")
    print(summary_df.to_string())

    grand_total = summary_df["Total sessions"].sum()
    saved       = baseline - grand_total
    pct         = saved / baseline * 100

    print(f"\nBaseline ({n_total} clients × {PATHWAY_LENGTH} sessions): {baseline}")
    print(f"Under Q* policy:                              {round(grand_total)}")
    print(f"Sessions saved:                               {round(saved)}")
    print(f"Capacity reduction:                           {pct:.1f}%\n")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data      = load_all_data()
    section_3a(data)
    best_pipe = section_3b(data)
    section_3c(best_pipe)
