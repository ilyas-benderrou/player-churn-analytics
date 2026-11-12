"""
Player analytics & churn prediction on World of Warcraft Avatar History (2008).

One command:
    python src/run_analysis.py

Parts
-----
A. Exploratory analysis of the whole year (activity, sessions, levels, progression).
B. Churn dataset: at a snapshot date T we take every avatar active in the 30 days
   before T, describe its behaviour with features computed ONLY from data before T,
   and label it "churned" if it is not seen at all during the CHURN_WINDOW days after T.
C. Player segmentation with K-means.
D. Churn prediction (Logistic Regression, Random Forest, Gradient Boosting),
   feature importance (permutation) and partial dependence.
E. Out-of-time check: train at an earlier snapshot, test at T.
All numbers in the README come from the files written to outputs/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from prepare_data import load_all, sessionize

warnings.filterwarnings("ignore")
SEED = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG = PROJECT_DIR / "figures"
OUT = PROJECT_DIR / "outputs"

SNAPSHOT = pd.Timestamp("2008-07-01")      # main prediction date
SNAPSHOT_EARLY = pd.Timestamp("2008-05-01")  # for the out-of-time check
CHURN_WINDOW = 60                           # days without any login = churn
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

NUMERIC = ["level", "level_gain_90d", "is_max_level", "tenure_days", "recency_days",
           "active_days_30d", "active_days_90d", "sessions_30d", "sessions_90d",
           "playtime_h_30d", "playtime_h_90d", "avg_session_min_90d", "playtime_trend",
           "n_zones_90d", "share_city", "share_questing", "share_dungeon", "share_pvp",
           "in_guild", "guild_size", "n_guilds_90d"]
CATEGORICAL = ["race", "charclass"]
SEGMENT_FEATURES = ["playtime_h_90d", "active_days_90d", "avg_session_min_90d", "level",
                    "share_questing", "share_dungeon", "share_pvp", "share_city", "guild_size"]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# A. Exploratory analysis
# =========================================================================== #
def eda(rec: pd.DataFrame, ses: pd.DataFrame) -> dict:
    facts = {}
    # --- monthly / daily active avatars --------------------------------------
    daily = rec.groupby(rec["timestamp"].dt.normalize())["char"].nunique()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily.values, color=PALETTE[0], lw=1, alpha=0.6, label="daily")
    ax.plot(daily.index, daily.rolling(7, center=True).mean(), color=PALETTE[0], lw=2.5,
            label="7-day average")
    for date, label in [("2008-10-14", "Patch 3.0.2"), ("2008-11-13", "Wrath of the\nLich King")]:
        ax.axvline(pd.Timestamp(date), color="k", ls="--", lw=1)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.97, " " + label, va="top", fontsize=9)
    ax.axvspan(SNAPSHOT - pd.Timedelta(days=30), SNAPSHOT, color=PALETTE[1], alpha=0.15,
               label="churn study: active window")
    ax.axvspan(SNAPSHOT, SNAPSHOT + pd.Timedelta(days=CHURN_WINDOW), color=PALETTE[3],
               alpha=0.12, label="churn study: label window")
    ax.set_ylabel("Distinct avatars online")
    ax.set_title("Daily active avatars on the server (2008, Horde faction)")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "01_daily_active_avatars.png")
    facts["daily_active_median"] = float(daily.median())

    # --- when do people play? weekday x hour heatmap -------------------------
    t = rec["timestamp"]
    heat = rec.groupby([t.dt.dayofweek, t.dt.hour]).size().unstack(fill_value=0)
    heat = heat / heat.values.sum() * 100
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(heat.values, aspect="auto", cmap="magma")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of the day (server time)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% of all play snapshots")
    ax.set_title("When are players online?")
    save(fig, "02_activity_heatmap.png")
    facts["peak_hour"] = int(rec.groupby(t.dt.hour).size().idxmax())

    # --- sessions -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(1, np.log10(ses["minutes"].quantile(0.999)), 40)
    axes[0].hist(ses["minutes"], bins=bins, color=PALETTE[0])
    axes[0].set_xscale("log")
    axes[0].axvline(ses["minutes"].median(), color="k", ls="--",
                    label=f"median = {ses['minutes'].median():.0f} min")
    axes[0].set_xlabel("Session length (minutes, log scale)")
    axes[0].set_ylabel("Sessions")
    axes[0].set_title("Session length")
    axes[0].legend()
    per_char = ses.groupby("char").size()
    axes[1].hist(per_char, bins=np.logspace(0, np.log10(per_char.max()), 40), color=PALETTE[2])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Sessions per avatar in 2008 (log scale)")
    axes[1].set_ylabel("Avatars (log scale)")
    axes[1].set_title("A few avatars play a lot, most play little")
    save(fig, "03_sessions.png")
    facts["median_session_min"] = float(ses["minutes"].median())
    facts["n_sessions"] = int(len(ses))
    facts["share_avatars_single_session"] = float((per_char == 1).mean())

    # --- levels and classes -----------------------------------------------------
    last = rec.groupby("char").last()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(last["level"], bins=np.arange(1, 82), color=PALETTE[4])
    axes[0].set_xlabel("Last observed level")
    axes[0].set_ylabel("Avatars")
    axes[0].set_title("Level distribution (peaks = level caps 70 and 80)")
    cls = last["charclass"].value_counts()
    axes[1].barh(cls.index[::-1].astype(str), cls.values[::-1], color=PALETTE[0])
    axes[1].set_xlabel("Avatars")
    axes[1].set_title("Class popularity")
    save(fig, "04_levels_and_classes.png")
    facts["share_level_70_or_more"] = float((last["level"] >= 70).mean())

    # --- progression: playtime needed to reach each level -----------------------
    # avatars first seen at level 1 (created in 2008): cumulative playtime per level
    first_lvl = rec.groupby("char")["level"].first()
    new_chars = first_lvl[first_lvl == 1].index
    s = ses[ses["char"].isin(new_chars)].sort_values(["char", "start"]).copy()
    s["cum_hours"] = s.groupby("char")["minutes"].cumsum() / 60
    reach = s.groupby(["char", "level_end"])["cum_hours"].min().reset_index()
    prog = reach.groupby("level_end")["cum_hours"].agg(["median", lambda x: x.quantile(.25),
                                                        lambda x: x.quantile(.75), "size"])
    prog.columns = ["median", "q25", "q75", "n"]
    prog = prog[(prog["n"] >= 30) & (prog.index <= 70)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(prog.index, prog["median"], color=PALETTE[3], lw=2, label="median")
    ax.fill_between(prog.index, prog["q25"], prog["q75"], color=PALETTE[3], alpha=0.2,
                    label="25-75 %")
    ax.set_xlabel("Level reached")
    ax.set_ylabel("Cumulative play time (hours)")
    ax.set_title(f"Levelling speed of avatars created in 2008 (n = {len(new_chars):,})")
    ax.legend()
    save(fig, "05_levelling_progression.png")
    reached = reach.groupby("char")["level_end"].max()
    facts["n_new_avatars"] = int(len(new_chars))
    facts["new_avatars_reaching_10"] = float((reached >= 10).mean())
    facts["new_avatars_reaching_70"] = float((reached >= 70).mean())
    if 70 in prog.index:
        facts["median_hours_to_70"] = float(prog.loc[70, "median"])

    # --- where is time spent? ---------------------------------------------------
    share = rec["zone_type"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(share.index[::-1].astype(str), share.values[::-1], color=PALETTE[5])
    ax.set_xlabel("% of play snapshots")
    ax.set_title("Where do avatars spend their time?")
    save(fig, "06_time_by_zone_type.png")
    return facts


# =========================================================================== #
# B. Churn dataset
# =========================================================================== #
def build_churn_dataset(rec: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Features from data strictly before T; label from [T, T + CHURN_WINDOW)."""
    before = rec[rec["timestamp"] < T]
    active = before.loc[before["timestamp"] >= T - pd.Timedelta(days=30), "char"].unique()
    hist = before[before["char"].isin(active)]
    w90 = hist[hist["timestamp"] >= T - pd.Timedelta(days=90)]
    w30 = hist[hist["timestamp"] >= T - pd.Timedelta(days=30)]
    w60_30 = hist[(hist["timestamp"] >= T - pd.Timedelta(days=60)) &
                  (hist["timestamp"] < T - pd.Timedelta(days=30))]

    g = hist.groupby("char")
    f = pd.DataFrame({
        "level": g["level"].last(),
        "race": g["race"].last().astype(str),
        "charclass": g["charclass"].last().astype(str),
        "tenure_days": (T - g["timestamp"].min()).dt.total_seconds() / 86400,
        "recency_days": (T - g["timestamp"].max()).dt.total_seconds() / 86400,
        "last_guild": g["guild"].last(),
    })
    f["level_gain_90d"] = f["level"] - w90.groupby("char")["level"].first()
    f["is_max_level"] = (f["level"] >= 70).astype(int)
    f["active_days_30d"] = w30.groupby("char")["timestamp"].apply(lambda s: s.dt.normalize().nunique())
    f["active_days_90d"] = w90.groupby("char")["timestamp"].apply(lambda s: s.dt.normalize().nunique())

    s90 = sessionize(w90)
    s30 = s90[s90["start"] >= T - pd.Timedelta(days=30)]
    s_prev = sessionize(w60_30) if len(w60_30) else s90.iloc[0:0]
    f["sessions_90d"] = s90.groupby("char").size()
    f["sessions_30d"] = s30.groupby("char").size()
    f["playtime_h_90d"] = s90.groupby("char")["minutes"].sum() / 60
    f["playtime_h_30d"] = s30.groupby("char")["minutes"].sum() / 60
    prev = s_prev.groupby("char")["minutes"].sum() / 60
    f["avg_session_min_90d"] = s90.groupby("char")["minutes"].mean()
    # > 0: plays more than the month before; < 0: plays less
    f["playtime_trend"] = np.log1p(f["playtime_h_30d"].fillna(0)) - np.log1p(prev.reindex(f.index).fillna(0))

    f["n_zones_90d"] = w90.groupby("char")["zone"].nunique()
    zt = pd.crosstab(w90["char"], w90["zone_type"], normalize="index")
    f["share_city"] = zt.get("City", 0)
    f["share_questing"] = zt.get("Zone", 0)
    f["share_dungeon"] = zt.get("Dungeon", 0)
    f["share_pvp"] = zt.get("Battleground", 0) + zt.get("Arena", 0)
    f["n_guilds_90d"] = w90.groupby("char")["guild"].nunique()
    f["in_guild"] = (f["last_guild"] != -1).astype(int)
    gsize = w30[w30["guild"] != -1].groupby("guild")["char"].nunique()
    f["guild_size"] = f["last_guild"].map(gsize).where(f["in_guild"] == 1, 0)
    f = f.drop(columns="last_guild").fillna(0)
    f[NUMERIC] = f[NUMERIC].astype(float)  # scikit-learn's partial dependence needs floats

    after = rec[(rec["timestamp"] >= T) & (rec["timestamp"] < T + pd.Timedelta(days=CHURN_WINDOW))]
    f["churn"] = (~f.index.isin(after["char"].unique())).astype(int)
    return f


def churn_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bands = pd.cut(df["level"], [0, 20, 40, 60, 69, 70], labels=["1-20", "21-40", "41-60", "61-69", "70"])
    r = df.groupby(bands, observed=True)["churn"].mean() * 100
    axes[0].bar(r.index.astype(str), r.values, color=PALETTE[3])
    axes[0].set_xlabel("Level band")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_title("Churn by level")
    r = df.groupby("in_guild")["churn"].mean() * 100
    axes[1].bar(["No guild", "In a guild"], r.values, color=[PALETTE[7], PALETTE[2]])
    axes[1].set_title("Churn by guild membership")
    q = pd.qcut(df["active_days_30d"], 5, duplicates="drop")
    r = df.groupby(q, observed=True)["churn"].mean() * 100
    axes[2].bar([f"{int(i.left) + 1}-{int(i.right)} days" for i in r.index], r.values,
                color=PALETTE[0])
    axes[2].set_xlabel("Active days in the last 30 days (quantile groups)")
    axes[2].set_title("Churn by recent activity")
    for ax in axes:
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=9)
    save(fig, "07_churn_by_group.png")


# =========================================================================== #
# C. Segmentation
# =========================================================================== #
def segmentation(df: pd.DataFrame) -> pd.DataFrame:
    X = df[SEGMENT_FEATURES].copy()
    for c in ["playtime_h_90d", "avg_session_min_90d", "guild_size"]:
        X[c] = np.log1p(X[c])  # very skewed -> log
    Xs = StandardScaler().fit_transform(X)

    ks, inertia, sil = range(2, 9), [], []
    rng = np.random.RandomState(SEED)
    sample = rng.choice(len(Xs), size=min(5000, len(Xs)), replace=False)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs[sample], km.labels_[sample]))
    # Choose k: best silhouette among k = 3..6 (2 groups are too coarse to act on,
    # more than 6 is hard to explain to a game team).
    k_best = int(max(range(3, 7), key=lambda k: sil[k - 2]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(list(ks), inertia, "o-")
    axes[0].set_title("Elbow (inertia)")
    axes[1].plot(list(ks), sil, "o-", color=PALETTE[1])
    axes[1].axvline(k_best, color="k", ls="--")
    axes[1].set_title(f"Silhouette (chosen k = {k_best})")
    for ax in axes:
        ax.set_xlabel("Number of clusters k")
    save(fig, "08_kmeans_choice.png")

    km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit(Xs)
    df = df.copy()
    df["segment"] = km.labels_

    prof = df.groupby("segment")[SEGMENT_FEATURES + ["churn"]].mean()
    prof["size"] = df.groupby("segment").size()
    prof["name"] = [name_segment(r) for _, r in prof.iterrows()]
    # make names unique if two clusters get the same label
    prof["name"] = [f"{n} ({i})" if (prof["name"] == n).sum() > 1 else n
                    for i, n in zip(prof.index, prof["name"])]
    prof.to_csv(OUT / "segment_profiles.csv")
    df["segment_name"] = df["segment"].map(prof["name"])

    # profile heatmap (z-scores of the segment means)
    z = (prof[SEGMENT_FEATURES] - df[SEGMENT_FEATURES].mean()) / df[SEGMENT_FEATURES].std()
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.8 * k_best + 2.5),
                             gridspec_kw={"width_ratios": [3, 1]})
    im = axes[0].imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    axes[0].set_xticks(range(len(SEGMENT_FEATURES)), SEGMENT_FEATURES, rotation=35, ha="right")
    axes[0].set_yticks(range(k_best), prof["name"])
    for i in range(k_best):
        for j, c in enumerate(SEGMENT_FEATURES):
            v = prof.iloc[i][c]
            axes[0].text(j, i, f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}",
                         ha="center", va="center", fontsize=8)
    axes[0].grid(False)
    axes[0].set_title("Segment profiles (colour = above/below average, text = mean value)")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    axes[1].barh(prof["name"], prof["churn"] * 100, color=PALETTE[3])
    axes[1].invert_yaxis()
    for i, (v, n) in enumerate(zip(prof["churn"] * 100, prof["size"])):
        axes[1].text(v, i, f" {v:.0f}% (n={n})", va="center", fontsize=8)
    axes[1].set_xlabel("Churn rate (%)")
    axes[1].set_yticks([])
    axes[1].set_title("Churn per segment")
    save(fig, "09_segments_profiles.png")

    pcs = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for s in range(k_best):
        m = df["segment"].values == s
        ax.scatter(pcs[m, 0], pcs[m, 1], s=4, alpha=0.4, color=PALETTE[s], label=prof.loc[s, "name"])
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Player segments (2-D projection)")
    ax.legend(markerscale=4, fontsize=8)
    save(fig, "10_segments_pca.png")
    print("\nSegments:\n", prof.round(2).to_string())
    return df


def name_segment(r: pd.Series) -> str:
    """Simple readable name from the dominant behaviour of a cluster (rules
    applied in order; the numbers behind each name are in segment_profiles.csv)."""
    if r["share_city"] > 0.6:
        return "City idlers"
    if r["level"] >= 60 and r["playtime_h_90d"] > 50:
        return "Engaged end-game players"
    if r["share_pvp"] > 0.30:
        return "PvP fighters"
    if r["share_dungeon"] > 0.25:
        return "Dungeon / raid players"
    if r["level"] < 60 and r["share_questing"] > 0.4:
        return "Levelling questers"
    if r["playtime_h_90d"] < 20:
        return "Casual players"
    return "Regular players"


# =========================================================================== #
# D. Prediction
# =========================================================================== #
def make_models() -> dict:
    ohe = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    ordinal = ColumnTransformer([("num", "passthrough", NUMERIC),
                                 ("cat", OrdinalEncoder(), CATEGORICAL)])
    return {
        "Logistic Regression": Pipeline([("prep", ohe), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", ohe), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", ordinal), ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced",
            categorical_features=[len(NUMERIC), len(NUMERIC) + 1], random_state=SEED))]),
    }


def evaluate(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "F1": f1_score(y, pred), "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred), "Accuracy": accuracy_score(y, pred)}


def prediction(df: pd.DataFrame, df_early: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    X, y = df[NUMERIC + CATEGORICAL], df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    rows, probs, fitted = [], {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        probs[name], fitted[name] = p, model
        cv_auc = cross_val_score(make_models()[name], X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        # out-of-time: train on the May snapshot, test on the July snapshot
        oot = make_models()[name].fit(df_early[NUMERIC + CATEGORICAL], df_early["churn"])
        oot_auc = roc_auc_score(y, oot.predict_proba(X)[:, 1])
        rows.append({"Model": name, **evaluate(yte, p), "CV_ROC_AUC_mean": cv_auc.mean(),
                     "CV_ROC_AUC_std": cv_auc.std(), "OutOfTime_ROC_AUC": oot_auc})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_results.csv", index=False)
    print("\nModel results:\n", res.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (name, p), c in zip(probs.items(), PALETTE):
        fpr, tpr, _ = roc_curve(yte, p)
        axes[0].plot(fpr, tpr, color=c, label=f"{name} (AUC={roc_auc_score(yte, p):.3f})")
        pr, rc, _ = precision_recall_curve(yte, p)
        axes[1].plot(rc, pr, color=c, label=f"{name} (AP={average_precision_score(yte, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (test set)")
    axes[1].axhline(yte.mean(), color="k", ls="--", label=f"base rate ({yte.mean():.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve (test set)")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "11_model_curves.png")

    best = res.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
    pi = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=10,
                                random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=PALETTE[0])
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title(f"What predicts churn? ({best}, permutation importance)")
    save(fig, "12_feature_importance.png")

    top4 = [f for f in imp["feature"] if f in NUMERIC][:4]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    PartialDependenceDisplay.from_estimator(fitted[best],
                                            Xte.sample(min(1000, len(Xte)), random_state=SEED),
                                            top4, ax=axes, grid_resolution=30,
                                            percentiles=(0.02, 0.98))
    fig.suptitle(f"Partial dependence: predicted churn probability vs. feature ({best})", y=1.02)
    save(fig, "13_partial_dependence.png")
    return res, best, imp


# =========================================================================== #
def main():
    for d in (FIG, OUT):
        d.mkdir(exist_ok=True)
    rec, ses = load_all()
    print(f"{len(rec):,} records, {rec['char'].nunique():,} avatars, {len(ses):,} sessions")

    facts = eda(rec, ses)
    df = build_churn_dataset(rec, SNAPSHOT)
    df_early = build_churn_dataset(rec, SNAPSHOT_EARLY)
    df.to_parquet(PROJECT_DIR / "data" / "processed" / "churn_dataset.parquet")
    print(f"\nChurn dataset at {SNAPSHOT.date()}: {len(df):,} active avatars, "
          f"churn rate = {df['churn'].mean():.3f}")
    print(f"Early snapshot {SNAPSHOT_EARLY.date()}: {len(df_early):,} avatars, "
          f"churn rate = {df_early['churn'].mean():.3f}")
    churn_eda(df)
    seg = segmentation(df)
    res, best, imp = prediction(df, df_early)

    facts.update({
        "snapshot": str(SNAPSHOT.date()), "churn_window_days": CHURN_WINDOW,
        "n_active_avatars": int(len(df)), "churn_rate": float(df["churn"].mean()),
        "n_active_avatars_early": int(len(df_early)),
        "churn_rate_early": float(df_early["churn"].mean()),
        "churn_rate_in_guild": float(df.loc[df.in_guild == 1, "churn"].mean()),
        "churn_rate_no_guild": float(df.loc[df.in_guild == 0, "churn"].mean()),
        "churn_rate_level70": float(df.loc[df.level >= 70, "churn"].mean()),
        "churn_rate_below70": float(df.loc[df.level < 70, "churn"].mean()),
        "best_model": best, "n_segments": int(seg["segment"].nunique()),
    })
    (OUT / "key_facts.json").write_text(json.dumps(facts, indent=2))
    print("\nKey facts:", json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
