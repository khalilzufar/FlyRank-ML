"""Reproducible analysis helpers for the FlyRank Machine Learning track.

The public starter slice is used when the gated warehouse is not available.  The
target is an observed snapshot proxy (trend_direction == ``down``), so every
public conclusion must remain directional and decision-support oriented.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
RAW_URL = "https://raw.githubusercontent.com/khalilzufar/FlyRank-ML/main/data/raw/content_refresh_anonymized.csv"
WORK_OUTPUTS = ROOT / "work" / "outputs"
WORK_FIGURES = ROOT / "work" / "figures"

RANDOM_STATE = 42
TARGET = "is_declining_label"

# These are available before the snapshot label.  trend_pct and trend_direction
# are deliberately absent because the label is derived from trend_direction.
NUMERIC_FEATURES = [
    # Only pre-label-window fields belong in the model matrix.  The last-30d
    # fields are retained on the frame for outcome reporting and action context,
    # but are intentionally not features.
    "search_volume",
    "competition",
    "cpc",
    "word_count",
    "char_count",
    "impressions_prev_30d",
    "clicks_prev_30d",
    "sessions_prev_30d",
    "users_prev_30d",
    "engaged_sessions_prev_30d",
    "ai_sessions_prev_30d",
    "scroll_events_prev_30d",
    "days_with_impressions_prev_30d",
    "days_with_sessions_prev_30d",
    "content_age_days",
    "age_tier_order",
    "days_since_last_update",
    "ctr_prev_30d",
    "avg_position_prev_30d",
    "engagement_rate_prev_30d",
    "scroll_rate_prev_30d",
    "ai_traffic_pct_prev_30d",
    "query_impressions_prev_30d",
    "query_clicks_prev_30d",
    "query_count_prev_30d",
    "query_position_prev_30d",
    "query_ctr_prev_30d",
]
CATEGORICAL_FEATURES = [
    "competition_level",
    "content_type",
    "main_intent",
    "age_tier",
    "freshness_tier",
    "word_count_tier",
]


def ensure_dirs() -> None:
    WORK_OUTPUTS.mkdir(parents=True, exist_ok=True)
    WORK_FIGURES.mkdir(parents=True, exist_ok=True)


def load_starter(path: str | Path | None = None) -> pd.DataFrame:
    """Load and lightly clean the anonymized 30k-row starter slice."""

    source = Path(path) if path else RAW_PATH
    if source.exists():
        frame = pd.read_csv(source)
        source_label = "bundled anonymized starter slice"
    else:
        frame = pd.read_csv(RAW_URL)
        source_label = "public anonymized starter slice from GitHub"

    frame = frame.copy()
    for column in NUMERIC_FEATURES + ["trend_pct"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["is_declining_label"] = (
        frame["trend_direction"].fillna("").astype(str).str.lower().eq("down").astype(int)
    )
    frame["ctr_prev_30d"] = (
        frame["clicks_prev_30d"] / frame["impressions_prev_30d"].replace(0, np.nan) * 100
    )
    frame["avg_position_prev_30d"] = pd.to_numeric(frame["avg_position"], errors="coerce")
    frame["engagement_rate_prev_30d"] = pd.to_numeric(frame["engagement_rate"], errors="coerce")
    frame["scroll_rate_prev_30d"] = pd.to_numeric(frame["scroll_rate"], errors="coerce")
    frame["ai_traffic_pct_prev_30d"] = pd.to_numeric(frame["ai_traffic_pct"], errors="coerce")
    frame["query_impressions_prev_30d"] = np.nan
    frame["query_clicks_prev_30d"] = np.nan
    frame["query_count_prev_30d"] = np.nan
    frame["query_position_prev_30d"] = np.nan
    frame["query_ctr_prev_30d"] = np.nan
    frame["users_prev_30d"] = frame.get("users_90d", 0)
    frame["engaged_sessions_prev_30d"] = frame.get("engaged_sessions_90d", 0)
    frame["ai_sessions_prev_30d"] = frame.get("ai_sessions_90d", 0)
    frame["scroll_events_prev_30d"] = frame.get("scroll_events_90d", 0)
    frame["days_with_impressions_prev_30d"] = frame.get("days_with_impressions", 0)
    frame["days_with_sessions_prev_30d"] = frame.get("days_with_sessions", 0)
    frame = frame[
        (frame["impressions_90d"].fillna(0) > 0)
        & (frame["content_age_days"].fillna(0) >= 90)
    ].drop_duplicates("content_id").reset_index(drop=True)
    frame.attrs["source_label"] = source_label
    frame.attrs["target_definition"] = "trend_direction == 'down'"
    return frame


def _warehouse_connection() -> Any:
    """Connect DuckDB to the gated release using the local HF token cache."""

    import duckdb
    from huggingface_hub import get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Use `hf auth login` first.")
    connection = duckdb.connect()
    escaped = token.replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{escaped}')"
    )
    return connection


def load_warehouse(*, force: bool = False) -> pd.DataFrame:
    """Aggregate the gated warehouse into one safe, content-level modeling frame.

    The daily fact table is scanned only for the final 60-day feature/outcome
    window.  The resulting aggregate is cached as a local ignored Parquet file;
    raw warehouse rows never enter the repository's public artifacts.
    """

    ensure_dirs()
    cache_path = WORK_OUTPUTS / "warehouse_content_features.parquet"
    if cache_path.exists() and not force:
        frame = pd.read_parquet(cache_path)
        frame["query_ctr_prev_30d"] = (
            frame["query_clicks_prev_30d"]
            / frame["query_impressions_prev_30d"].replace(0, np.nan)
            * 100
        )
        frame.attrs["source_label"] = "gated FlyRank warehouse release v20260703"
        frame.attrs["target_definition"] = "last-30d impressions < 80% of prior-30d impressions"
        frame.attrs["snapshot_end"] = "2026-06-30"
        return frame

    con = _warehouse_connection()
    rel = "hf://datasets/FlyRank/internship-warehouse"
    daily = f"read_parquet('{rel}/fact_content_daily_performance/**/*.parquet')"
    content = f"read_parquet('{rel}/dim_content.parquet')"
    query = f"read_parquet('{rel}/fact_content_query_90d.parquet')"

    daily_sql = f"""
        WITH bounds AS (
            SELECT MAX(report_date) AS end_d
            FROM {daily}
        )
        SELECT
            f.client_hash_id,
            f.content_hash_id,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.gsc_impressions, 0) ELSE 0 END) AS impressions_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.gsc_impressions, 0) ELSE 0 END) AS impressions_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.gsc_clicks, 0) ELSE 0 END) AS clicks_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.gsc_clicks, 0) ELSE 0 END) AS clicks_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY AND f.ga4_data_available
                     THEN COALESCE(f.ga4_sessions, 0) ELSE 0 END) AS sessions_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY AND f.ga4_data_available
                     THEN COALESCE(f.ga4_sessions, 0) ELSE 0 END) AS sessions_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY AND f.ga4_data_available
                     THEN COALESCE(f.ga4_users, 0) ELSE 0 END) AS users_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY AND f.ga4_data_available
                     THEN COALESCE(f.ga4_users, 0) ELSE 0 END) AS users_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY AND f.ga4_data_available
                     THEN COALESCE(f.ga4_engaged_sessions, 0) ELSE 0 END) AS engaged_sessions_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY AND f.ga4_data_available
                     THEN COALESCE(f.ga4_engaged_sessions, 0) ELSE 0 END) AS engaged_sessions_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.sessions_ai, 0) ELSE 0 END) AS ai_sessions_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.sessions_ai, 0) ELSE 0 END) AS ai_sessions_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.scroll_events, 0) ELSE 0 END) AS scroll_events_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     THEN COALESCE(f.scroll_events, 0) ELSE 0 END) AS scroll_events_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     AND COALESCE(f.gsc_impressions, 0) > 0 THEN 1 ELSE 0 END) AS days_with_impressions_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     AND COALESCE(f.gsc_impressions, 0) > 0 THEN 1 ELSE 0 END) AS days_with_impressions_prev_30d,
            SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     AND COALESCE(f.ga4_sessions, 0) > 0 AND f.ga4_data_available THEN 1 ELSE 0 END) AS days_with_sessions_last_30d,
            SUM(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     AND COALESCE(f.ga4_sessions, 0) > 0 AND f.ga4_data_available THEN 1 ELSE 0 END) AS days_with_sessions_prev_30d,
            AVG(CASE WHEN f.report_date > b.end_d - INTERVAL 30 DAY
                     THEN NULLIF(f.gsc_avg_position, 0) END) AS avg_position_last_30d,
            AVG(CASE WHEN f.report_date <= b.end_d - INTERVAL 30 DAY
                     THEN NULLIF(f.gsc_avg_position, 0) END) AS avg_position_prev_30d,
            MAX(b.end_d) AS snapshot_end
        FROM {daily} f
        CROSS JOIN bounds b
        WHERE f.report_date > b.end_d - INTERVAL 60 DAY
          AND COALESCE(f.gsc_data_available, FALSE)
        GROUP BY 1, 2
        HAVING impressions_prev_30d >= 100
    """
    daily_frame = con.sql(daily_sql).df()

    query_sql = f"""
        SELECT
            client_hash_id,
            content_hash_id,
            SUM(COALESCE(impressions_prev30, 0)) AS query_impressions_prev_30d,
            SUM(COALESCE(clicks_prev30, 0)) AS query_clicks_prev_30d,
            SUM(CASE WHEN COALESCE(impressions_prev30, 0) > 0 THEN 1 ELSE 0 END) AS query_count_prev_30d,
            AVG(CASE WHEN COALESCE(impressions_prev30, 0) > 0 THEN avg_position_prev30 END) AS query_position_prev_30d
        FROM {query}
        GROUP BY 1, 2
    """
    query_frame = con.sql(query_sql).df()
    content_frame = con.sql(
        f"""
        SELECT client_hash_id, content_hash_id, content_type, search_volume,
               competition, competition_level, cpc, main_intent, backlinks,
               category_count, char_count, word_count, content_created_date,
               last_optimized_date, is_published, is_deleted
        FROM {content}
        """
    ).df()
    con.close()

    frame = daily_frame.merge(
        content_frame, on=["client_hash_id", "content_hash_id"], how="left"
    ).merge(query_frame, on=["client_hash_id", "content_hash_id"], how="left")
    for column in frame.columns:
        if column.endswith("_date") or column == "snapshot_end":
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    frame["client_id"] = frame["client_hash_id"]
    frame["content_id"] = frame["content_hash_id"]
    frame["impressions_90d"] = frame["impressions_last_30d"] + frame["impressions_prev_30d"]
    frame["clicks_90d"] = frame["clicks_last_30d"] + frame["clicks_prev_30d"]
    frame["sessions_90d"] = frame["sessions_last_30d"] + frame["sessions_prev_30d"]
    frame["users_90d"] = frame["users_last_30d"] + frame["users_prev_30d"]
    frame["engaged_sessions_90d"] = frame["engaged_sessions_last_30d"] + frame["engaged_sessions_prev_30d"]
    frame["ai_sessions_90d"] = frame["ai_sessions_last_30d"] + frame["ai_sessions_prev_30d"]
    frame["scroll_events_90d"] = frame["scroll_events_last_30d"] + frame["scroll_events_prev_30d"]
    frame["days_with_impressions"] = frame["days_with_impressions_last_30d"] + frame["days_with_impressions_prev_30d"]
    frame["days_with_sessions"] = frame["days_with_sessions_last_30d"] + frame["days_with_sessions_prev_30d"]
    frame["ctr"] = frame["clicks_last_30d"] / frame["impressions_last_30d"].replace(0, np.nan) * 100
    frame["ctr_prev_30d"] = frame["clicks_prev_30d"] / frame["impressions_prev_30d"].replace(0, np.nan) * 100
    frame["avg_position"] = frame["avg_position_last_30d"]
    frame["engagement_rate"] = frame["engaged_sessions_last_30d"] / frame["sessions_last_30d"].replace(0, np.nan) * 100
    frame["engagement_rate_prev_30d"] = frame["engaged_sessions_prev_30d"] / frame["sessions_prev_30d"].replace(0, np.nan) * 100
    frame["scroll_rate"] = frame["scroll_events_last_30d"] / frame["sessions_last_30d"].replace(0, np.nan) * 100
    frame["scroll_rate_prev_30d"] = frame["scroll_events_prev_30d"] / frame["sessions_prev_30d"].replace(0, np.nan) * 100
    frame["ai_traffic_pct"] = frame["ai_sessions_last_30d"] / frame["sessions_last_30d"].replace(0, np.nan) * 100
    frame["ai_traffic_pct_prev_30d"] = frame["ai_sessions_prev_30d"] / frame["sessions_prev_30d"].replace(0, np.nan) * 100
    frame["query_ctr_prev_30d"] = frame["query_clicks_prev_30d"] / frame["query_impressions_prev_30d"].replace(0, np.nan) * 100
    frame["trend_pct"] = (frame["impressions_last_30d"] / frame["impressions_prev_30d"] - 1) * 100
    frame["trend_direction"] = np.where(frame["trend_pct"] <= -20, "down", "not_down")
    frame[TARGET] = (frame["trend_pct"] <= -20).astype(int)
    frame["content_age_days"] = (
        frame["snapshot_end"] - frame["content_created_date"]
    ).dt.days.clip(lower=0)
    update_date = frame["last_optimized_date"].fillna(frame["content_created_date"])
    frame["days_since_last_update"] = (frame["snapshot_end"] - update_date).dt.days.clip(lower=0)
    frame["age_tier_order"] = pd.cut(
        frame["content_age_days"], [-1, 180, 365, np.inf], labels=[1, 2, 3]
    ).astype(float)
    frame["age_tier"] = pd.cut(
        frame["content_age_days"], [-1, 180, 365, np.inf], labels=["young", "mid", "old"]
    ).astype(str)
    frame["freshness_tier"] = pd.cut(
        frame["days_since_last_update"], [-1, 90, 180, np.inf], labels=["fresh", "recent", "stale"]
    ).astype(str)
    frame["word_count_tier"] = pd.cut(
        frame["word_count"], [-np.inf, 600, 1500, np.inf], labels=["short", "medium", "long"]
    ).astype(str)
    frame["position_tier"] = np.select(
        [frame["avg_position"].le(10), frame["avg_position"].le(20)],
        ["top10", "top20"],
        default="page2_or_missing",
    )
    frame["impression_tier"] = pd.qcut(
        frame["impressions_prev_30d"].rank(method="first"), 4,
        labels=["low", "mid", "high", "very_high"],
    ).astype(str)
    frame = frame[
        frame["impressions_prev_30d"].gt(0)
        & frame["is_deleted"].fillna(False).eq(False)
    ].drop_duplicates(["client_id", "content_id"]).reset_index(drop=True)
    frame.attrs["source_label"] = "gated FlyRank warehouse release v20260703"
    frame.attrs["target_definition"] = "last-30d impressions < 80% of prior-30d impressions"
    frame.attrs["snapshot_end"] = str(frame["snapshot_end"].max().date())
    frame.to_parquet(cache_path, index=False)
    return frame


def load_analysis_frame(*, force_warehouse: bool = False) -> pd.DataFrame:
    """Use the approved warehouse when available; otherwise use the starter slice."""

    if os.environ.get("FLYRANK_USE_STARTER") == "1":
        return load_starter()
    try:
        if not os.environ.get("HF_TOKEN"):
            try:
                from google.colab import userdata

                colab_token = userdata.get("HF_TOKEN")
                if colab_token:
                    os.environ["HF_TOKEN"] = colab_token
            except Exception:
                pass
        from huggingface_hub import get_token

        if force_warehouse or os.environ.get("HF_TOKEN") or get_token():
            return load_warehouse(force=force_warehouse)
    except Exception as exc:
        print(f"Warehouse unavailable; using starter slice ({type(exc).__name__}).")
    return load_starter()


def make_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create a leakage-aware numeric/categorical matrix with missingness flags."""

    numeric = [column for column in NUMERIC_FEATURES if column in frame.columns]
    numeric_frame = pd.DataFrame(index=frame.index)
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        missing = values.isna()
        if column == "avg_position":
            # In this dataset zero means no position data, not rank zero.
            missing = missing | values.eq(0)
            values = values.mask(values.eq(0))
        numeric_frame[column] = values
        numeric_frame[f"missing_{column}"] = missing.astype(float)
        median = values.median()
        numeric_frame[column] = values.replace([np.inf, -np.inf], np.nan).fillna(
            0.0 if pd.isna(median) else float(median)
        )

    categorical = [column for column in CATEGORICAL_FEATURES if column in frame.columns]
    categorical_frame = frame[categorical].fillna("unknown").astype(str)
    encoded = pd.get_dummies(categorical_frame, prefix=categorical, dtype=float)
    features = pd.concat(
        [numeric_frame.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1
    )
    return features.astype(float), list(features.columns)


def percentile_rank(values: pd.Series, *, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    if not ascending:
        numeric = -numeric
    return numeric.rank(method="average", pct=True).fillna(0.0)


def baseline_scores(frame: pd.DataFrame) -> pd.Series:
    """Transparent pre-label rule: stale, visible, low-CTR pages are reviewed first."""

    position_column = "avg_position_prev_30d" if "avg_position_prev_30d" in frame else "avg_position"
    position = pd.to_numeric(frame[position_column], errors="coerce").replace(0, np.nan)
    low_ctr = percentile_rank(frame["ctr_prev_30d"], ascending=False)
    position_risk = percentile_rank(position, ascending=True)
    return (
        0.35 * percentile_rank(frame["days_since_last_update"])
        + 0.35 * percentile_rank(np.log1p(frame["impressions_prev_30d"]))
        + 0.20 * low_ctr
        + 0.10 * position_risk
    ).astype(float)


def precision_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    values = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(scores)})
    top = values.sort_values("score", ascending=False).head(min(k, len(values)))
    return float(top["y"].mean()) if len(top) else 0.0


def metrics(y_true: pd.Series, scores: pd.Series) -> dict[str, float]:
    base_rate = float(pd.Series(y_true).mean())
    payload = {
        "base_rate": base_rate,
        "precision_at_20": precision_at_k(y_true, scores, 20),
        "precision_at_50": precision_at_k(y_true, scores, 50),
        "precision_at_100": precision_at_k(y_true, scores, 100),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores))
        if pd.Series(y_true).nunique() == 2
        else 0.5,
    }
    return {key: round(value, 6) for key, value in payload.items()}


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def split_indices(frame: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    if strategy == "random":
        train, test = train_test_split(
            indices,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=frame[TARGET],
        )
        return np.asarray(train), np.asarray(test)
    if strategy != "grouped_client":
        raise ValueError(f"Unknown split strategy: {strategy}")

    groups = frame["client_id"].fillna("unknown").astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train, test = next(splitter.split(frame, frame[TARGET], groups=groups))
    if frame.iloc[train][TARGET].nunique() < 2 or frame.iloc[test][TARGET].nunique() < 2:
        raise ValueError("Grouped split does not contain both target classes")
    return np.asarray(train), np.asarray(test)


def run_validation(frame: pd.DataFrame, strategy: str) -> dict[str, Any]:
    features, feature_names = make_feature_matrix(frame)
    train_idx, test_idx = split_indices(frame, strategy)
    model = build_model()
    model.fit(features.iloc[train_idx], frame[TARGET].iloc[train_idx])
    probabilities = pd.Series(
        model.predict_proba(features.iloc[test_idx])[:, 1], index=test_idx
    )
    baseline = baseline_scores(frame).iloc[test_idx]
    y_test = frame[TARGET].iloc[test_idx]
    return {
        "strategy": strategy,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_clients": int(frame.iloc[train_idx]["client_id"].nunique()),
        "test_clients": int(frame.iloc[test_idx]["client_id"].nunique()),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "target": TARGET,
        "target_definition": frame.attrs.get("target_definition", "trend_direction == 'down'"),
        "model": metrics(y_test, probabilities),
        "baseline": metrics(y_test, baseline),
        "false_positives_top_20": int(
            ((probabilities.sort_values(ascending=False).head(20).index).to_series().map(frame[TARGET]) == 0).sum()
        ),
        "top_feature_importance": [
            {"feature": str(name), "importance": round(float(value), 8)}
            for name, value in sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:15]
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_action_queue(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, feature_names = make_feature_matrix(frame)
    model = build_model()
    model.fit(features, frame[TARGET])
    model_probability = pd.Series(model.predict_proba(features)[:, 1], index=frame.index)
    baseline = baseline_scores(frame)
    baseline_min, baseline_max = float(baseline.min()), float(baseline.max())
    baseline_norm = (baseline - baseline_min) / max(baseline_max - baseline_min, 1e-9)

    queue = frame.copy()
    queue["model_probability"] = model_probability
    queue["baseline_score"] = baseline
    queue["final_score"] = (100 * (0.70 * model_probability + 0.30 * baseline_norm)).clip(0, 100)

    def reasons(row: pd.Series) -> str:
        output: list[str] = []
        if row["model_probability"] >= 0.65:
            output.append("model_decline_risk")
        visible_impressions = row["impressions_prev_30d"]
        visible_sessions = row.get("sessions_prev_30d", 0)
        visible_ctr = row.get("ctr_prev_30d", row.get("ctr", np.nan))
        visible_position = row.get("avg_position_prev_30d", row.get("avg_position", np.nan))
        if visible_impressions >= 500 and row["model_probability"] >= 0.50:
            output.append("visible_model_opportunity")
        if row["days_since_last_update"] >= 180 and visible_impressions >= 100:
            output.append("stale_visible_page")
        if 0 < visible_position <= 20 and visible_ctr < 0.5 and visible_impressions >= 500:
            output.append("ctr_review_candidate")
        engagement = row.get("engagement_rate_prev_30d", row.get("engagement_rate", np.nan))
        scroll = row.get("scroll_rate_prev_30d", row.get("scroll_rate", np.nan))
        if visible_sessions >= 30 and (
            0 < engagement < 30 or 0 < scroll < 30
        ):
            output.append("engagement_review_candidate")
        word_count = pd.to_numeric(row.get("word_count", np.nan), errors="coerce")
        if pd.notna(word_count) and word_count < 600 and visible_impressions >= 100:
            output.append("thin_visible_page")
        return "|".join(output or ["general_review"])

    queue["reason_codes"] = queue.apply(reasons, axis=1)

    def action(row: pd.Series) -> str:
        reason_set = set(row["reason_codes"].split("|"))
        if "thin_visible_page" in reason_set:
            return "expand_and_refresh"
        if "ctr_review_candidate" in reason_set and "model_decline_risk" in reason_set:
            return "refresh_and_review_ctr"
        if "engagement_review_candidate" in reason_set and "model_decline_risk" in reason_set:
            return "refresh_and_review_engagement"
        if reason_set.intersection({"model_decline_risk", "stale_visible_page", "visible_model_opportunity"}):
            return "refresh"
        return "monitor"

    queue["suggested_action"] = queue.apply(action, axis=1)
    high_cut = float(queue["final_score"].quantile(0.80))
    medium_cut = float(queue["final_score"].quantile(0.50))
    queue["confidence"] = np.select(
        [queue["final_score"] >= high_cut, queue["final_score"] >= medium_cut],
        ["high", "medium"],
        default="low",
    )
    queue = queue.sort_values(
        ["final_score", "impressions_prev_30d", "sessions_prev_30d"], ascending=[False, False, False]
    ).reset_index(drop=True)
    queue.insert(0, "rank", np.arange(1, len(queue) + 1))

    summary = {
        "rows_scored": int(len(queue)),
        "target_base_rate": round(float(frame[TARGET].mean()), 6),
        "action_counts": {str(k): int(v) for k, v in queue["suggested_action"].value_counts().items()},
        "confidence_counts": {str(k): int(v) for k, v in queue["confidence"].value_counts().items()},
        "top_feature_importance": [
            {"feature": str(name), "importance": round(float(value), 8)}
            for name, value in sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:15]
        ],
        "public_safety": "Queue is reviewer decision-support, not an automatic publishing decision.",
    }
    return queue, summary


def save_figures(queue: pd.DataFrame, validation: dict[str, Any]) -> list[str]:
    ensure_dirs()
    paths: list[str] = []
    plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.titlesize": 13})

    comparison = pd.DataFrame(
        {
            "approach": ["Baseline", "Model"],
            "precision_at_50": [
                validation["baseline"]["precision_at_50"],
                validation["model"]["precision_at_50"],
            ],
        }
    )
    ax = comparison.plot.bar(x="approach", y="precision_at_50", legend=False, color=["#B07AA1", "#4E79A7"])
    ax.set_ylabel("Precision@50")
    ax.set_xlabel("")
    ax.set_title("Model vs. transparent baseline on grouped client holdout")
    ax.bar_label(ax.containers[0], fmt="%.3f")
    path = WORK_FIGURES / "model_vs_baseline.svg"
    plt.tight_layout(); plt.savefig(path, format="svg"); plt.close()
    paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    counts = queue["suggested_action"].value_counts()
    ax = counts.plot.bar(color="#426B69")
    ax.set_ylabel("Rows")
    ax.set_xlabel("")
    ax.set_title("Suggested action mix")
    path = WORK_FIGURES / "action_mix.svg"
    plt.tight_layout(); plt.savefig(path, format="svg"); plt.close()
    paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    importance = pd.DataFrame(validation["top_feature_importance"]).head(10).sort_values("importance")
    ax = importance.plot.barh(x="feature", y="importance", legend=False, color="#6F4E7C")
    ax.set_xlabel("Random forest importance")
    ax.set_ylabel("")
    ax.set_title("Top features (association, not causation)")
    path = WORK_FIGURES / "feature_importance.svg"
    plt.tight_layout(); plt.savefig(path, format="svg"); plt.close()
    paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return paths


def run_artifacts() -> dict[str, Any]:
    """Run the selected analysis source and write all reusable receipts."""

    ensure_dirs()
    frame = load_analysis_frame()
    random_result = run_validation(frame, "random")
    grouped_result = run_validation(frame, "grouped_client")
    queue, queue_summary = make_action_queue(frame)
    validation_payload = {
        "source": frame.attrs.get("source_label"),
        "rows": int(len(frame)),
        "clients": int(frame["client_id"].nunique()),
        "target": TARGET,
        "target_definition": frame.attrs.get("target_definition"),
        "leakage_excluded": ["trend_pct", "trend_direction", "content_id", "client_id"],
        "random_split": random_result,
        "grouped_split": grouped_result,
        "interpretation": "The grouped result is the primary estimate because client_id repeats across rows.",
    }
    write_json(WORK_OUTPUTS / "ml09_validation_results.json", validation_payload)
    write_json(WORK_OUTPUTS / "ml10_action_playbook_results.json", queue_summary)
    queue.to_csv(WORK_OUTPUTS / "ml10_action_queue.csv", index=False)
    figures = save_figures(queue, grouped_result)
    validation_payload["figures"] = figures
    write_json(WORK_OUTPUTS / "capstone_results.json", validation_payload | {"queue": queue_summary})
    return {"frame": frame, "validation": validation_payload, "queue": queue, "queue_summary": queue_summary}


def write_paper_page(artifacts: dict[str, Any]) -> Path:
    """Write a self-contained static paper page for GitHub Pages."""

    ensure_dirs()
    validation = artifacts["validation"]
    grouped = validation["grouped_split"]
    random_result = validation["random_split"]
    queue_summary = artifacts["queue_summary"]
    paper_dir = ROOT / "docs"
    asset_dir = paper_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for source in [WORK_FIGURES / "model_vs_baseline.svg", WORK_FIGURES / "action_mix.svg", WORK_FIGURES / "feature_importance.svg"]:
        if source.exists():
            shutil.copyfile(source, asset_dir / source.name)

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    source_label = validation.get("source", "an anonymized release")
    source_lower = str(source_label).lower()
    source_description = (
        "the gated FlyRank warehouse release"
        if "warehouse" in source_lower
        else "the bundled anonymized starter slice"
    )
    target_definition = validation.get(
        "target_definition", "an observed decline proxy"
    )

    action_rows = "".join(
        f"<tr><td>{action.replace('_', ' ')}</td><td>{count:,}</td></tr>"
        for action, count in queue_summary["action_counts"].items()
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which content pages should be reviewed first?</title>
<style>
:root {{ color-scheme: light; --ink:#17232b; --muted:#5c6a72; --accent:#426b69; --paper:#f7f4ee; --card:#fff; }}
* {{ box-sizing:border-box }} body {{ margin:0; font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); background:var(--paper) }}
main {{ max-width:960px; margin:0 auto; padding:48px 20px 72px }} h1 {{ font-size:clamp(2rem,5vw,4rem); line-height:1.05; max-width:800px }} h2 {{ margin-top:48px; font-size:1.55rem }}
p {{ max-width:780px }} .eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.08em; text-transform:uppercase }} .lede {{ font-size:1.15rem; color:var(--muted) }}
.card {{ background:var(--card); border:1px solid #e4dfd5; border-radius:16px; padding:20px; margin:18px 0 }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px }} .metric {{ font-size:1.65rem; font-weight:750 }} .muted {{ color:var(--muted) }}
figure {{ margin:22px 0 }} figure img {{ width:100%; height:auto; background:#fff; border-radius:12px }} figcaption {{ color:var(--muted); font-size:.92rem; margin-top:6px }} table {{ width:100%; border-collapse:collapse; background:#fff }} th,td {{ padding:10px; text-align:left; border-bottom:1px solid #e4dfd5 }} a {{ color:var(--accent) }} code {{ background:#eee9df; padding:2px 5px; border-radius:5px }}
</style></head><body><main>
<p class="eyebrow">FlyRank ML internship · content refresh priority</p>
<h1>Which content pages look worth reviewing first?</h1>
<p class="lede">A reproducible ranking analysis for editorial review, built from {source_description}.</p>
<section class="card"><h2>Abstract</h2><p>We asked whether a model could rank content pages for refresh review better than a transparent stale-and-visible rule. We used {validation['rows']:,} anonymized content rows across {validation['clients']} pseudonymous client groups and defined the observed target as <code>{target_definition}</code>. A random forest was evaluated against the rule with a client-grouped holdout and leakage exclusions. On that holdout, the model reached Precision@50 {grouped['model']['precision_at_50']:.3f} versus {grouped['baseline']['precision_at_50']:.3f} for the rule, with a {pct(grouped['model']['base_rate'])} base rate. The output is decision support for human review, not a claim about what will cause Google traffic to change.</p></section>
<h2>1. Introduction / problem</h2><p>Editorial teams have limited time, so the useful output is a ranked queue rather than a yes/no flag. A false positive spends review time on a page that may not need attention; a false negative leaves a potentially important declining page unreviewed. The model helps organize signals, while an editor still checks the page and its context.</p>
<h2>2. Data</h2><p>This reproducible run uses {source_description}: {validation['rows']:,} rows and {validation['clients']} pseudonymous client groups. The target is the observed outcome <code>{target_definition}</code>; it is not a future intervention outcome. We excluded label-derived fields and identifiers from features. No client names, domains, URLs, titles, queries, credentials, or raw exports appear in the paper.</p>
<h2>3. Methodology</h2><p>Features are numeric performance and content fields plus one-hot categorical context, with missingness flags and median imputation. The transparent baseline combines percentile ranks for staleness, visibility, CTR risk, and position risk. The model is a seeded random forest. We report both a stratified random split and a client-grouped split; the grouped split is primary because rows share pseudonymous client context.</p>
<h2>4. Results</h2><div class="grid"><div class="card"><div class="metric">{grouped['model']['precision_at_50']:.3f}</div><div class="muted">Model Precision@50, grouped</div></div><div class="card"><div class="metric">{grouped['baseline']['precision_at_50']:.3f}</div><div class="muted">Baseline Precision@50, grouped</div></div><div class="card"><div class="metric">{pct(grouped['model']['base_rate'])}</div><div class="muted">Test base rate</div></div></div>
<table><thead><tr><th>Approach</th><th>Precision@20</th><th>Precision@50</th><th>Average precision</th><th>ROC AUC</th></tr></thead><tbody><tr><td>Baseline · random split</td><td>{random_result['baseline']['precision_at_20']:.3f}</td><td>{random_result['baseline']['precision_at_50']:.3f}</td><td>{random_result['baseline']['average_precision']:.3f}</td><td>{random_result['baseline']['roc_auc']:.3f}</td></tr><tr><td>Model · random split</td><td>{random_result['model']['precision_at_20']:.3f}</td><td>{random_result['model']['precision_at_50']:.3f}</td><td>{random_result['model']['average_precision']:.3f}</td><td>{random_result['model']['roc_auc']:.3f}</td></tr><tr><td>Baseline · grouped split</td><td>{grouped['baseline']['precision_at_20']:.3f}</td><td>{grouped['baseline']['precision_at_50']:.3f}</td><td>{grouped['baseline']['average_precision']:.3f}</td><td>{grouped['baseline']['roc_auc']:.3f}</td></tr><tr><td>Model · grouped split</td><td>{grouped['model']['precision_at_20']:.3f}</td><td>{grouped['model']['precision_at_50']:.3f}</td><td>{grouped['model']['average_precision']:.3f}</td><td>{grouped['model']['roc_auc']:.3f}</td></tr></tbody></table>
<figure><img src="assets/model_vs_baseline.svg" alt="Bar chart comparing grouped holdout Precision at 50"></figure>
<h2>5. Limitations & honest framing</h2><p>These are observed associations in {source_description}. The target is a snapshot proxy, so the analysis does not establish that refreshing a page will cause traffic recovery, nor does it predict or reverse-engineer Google's ranking system. Client-grouped performance is a better stress test than a random row split, but it still does not establish performance on a future time window or on a different warehouse release. Low-volume pages, missing history, seasonality, and unobserved editorial decisions can change the recommendation.</p>
<h2>6. Ranked recommendations</h2><p>Use the queue as a reviewer aid: start with high-confidence items that combine model risk with visible demand, then check CTR, engagement, staleness, and content depth. No page should be rewritten automatically from this score. The current queue contains:</p><table><thead><tr><th>Suggested action</th><th>Rows</th></tr></thead><tbody>{action_rows}</tbody></table><figure><img src="assets/action_mix.svg" alt="Bar chart of suggested action mix"></figure>
<h2>7. Reproducibility</h2><p>From a fresh clone, install <code>requirements.txt</code>, then run the completed notebooks in <code>work/notebooks/</code> from top to bottom. The analysis uses random seed {RANDOM_STATE}; metrics receipts are stored in <code>work/outputs/ml09_validation_results.json</code>, <code>work/outputs/ml10_action_playbook_results.json</code>, and <code>work/outputs/capstone_results.json</code>. The source notebooks and helper module are in the public repo.</p><figure><img src="assets/feature_importance.svg" alt="Bar chart of top model features"></figure>
<h2>8. Acknowledgments & data credit</h2><p>Built on the <a href="https://flyrank.ai" rel="noopener">FlyRank ML Internship dataset</a>. This page uses {source_description} and follows the repository's public-safety rules.</p>
</main></body></html>"""
    output = paper_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    report = f"""# Capstone Report — Content Refresh Priority

- **Author:** Khalil Zufar
- **Lane:** Content Refresh Priority
- **Repo:** https://github.com/khalilzufar/FlyRank-ML

## 0. Abstract

We asked whether a model could rank content pages for refresh review better than a transparent stale-and-visible rule. We used {validation['rows']:,} anonymized rows across {validation['clients']} pseudonymous client groups and used `{target_definition}` as an observed target. A seeded random forest was compared with the rule under a client-grouped holdout. The model reached grouped Precision@50 {grouped['model']['precision_at_50']:.3f} versus {grouped['baseline']['precision_at_50']:.3f} for the rule, with a {pct(grouped['model']['base_rate'])} base rate. The output is decision support for human review, not a causal claim about Google traffic.

## 1. Problem framing

Editorial teams have limited review capacity. The output is a ranked queue of pages to inspect first, where a human checks intent, demand, seasonality, business priority, and the live page before choosing an action.

## 2. Data safety

This run uses {source_description}. Label-derived fields and pseudonymous IDs are excluded from the model matrix; IDs are used only for tracing and grouped validation. No names, domains, URLs, titles, raw queries, credentials, or raw warehouse exports are written to public artifacts.

## 3. Baseline

The transparent baseline combines percentile ranks for staleness, visibility, CTR risk, and position risk. On the grouped holdout it achieved Precision@50 {grouped['baseline']['precision_at_50']:.3f}.

## 4. Model / analysis

The model is a seeded random forest over numeric performance/content fields with missingness flags and one-hot categorical context. The target is the observed outcome `{target_definition}`.

## 5. Evaluation

The primary split is a client-grouped holdout: {grouped['train_clients']} client groups for training and {grouped['test_clients']} for testing. The model's grouped Precision@50 was {grouped['model']['precision_at_50']:.3f}, average precision {grouped['model']['average_precision']:.3f}, and ROC AUC {grouped['model']['roc_auc']:.3f}. The test base rate was {pct(grouped['model']['base_rate'])}. A random row holdout is reported in `work/outputs/capstone_results.json` as a comparison, not the primary estimate.

## 6. Interpretation

The strongest model features in this run were previous/current impression measures, visibility days, content age, position, and current engagement/search fields. These are associations used for prioritization; feature importance is not a causal effect.

## 7. Recommendation

Start with high-confidence rows combining model risk and visible demand. Review CTR/snippets, engagement/readability, staleness, and content depth as appropriate. Never auto-publish, delete, redirect, or claim that a refresh will cause recovery from this queue.

## 8. Reproducibility

Install `requirements.txt`, then run `work/notebooks/w06_validation_audit.ipynb`, `work/notebooks/w07_action_playbook.ipynb`, and `work/notebooks/capstone.ipynb` top to bottom. The seed is 42. Receipts are in `work/outputs/`, and charts are in `work/figures/` and `docs/assets/`.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). This paper uses {source_description}.
"""
    (ROOT / "work" / "capstone_report.md").write_text(report, encoding="utf-8")
    return output
