"""
Detection performance on the labelled test-bench.

Because the attack injector labels every track, we can score the detector the way
a real security product must be scored: ROC/AUC, precision, recall and the
confusion matrix at the operating threshold. These are the numbers that turn a
demo into evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

import config


def evaluate(track_scores: pd.DataFrame, truth: pd.DataFrame,
             threshold: float = config.DETECT.alert_threshold) -> dict:
    """Score per-track predictions against ground truth.

    `track_scores` needs [track_id, score]; `truth` needs [track_id, is_attack].
    """
    m = track_scores.merge(truth[["track_id", "is_attack"]], on="track_id", how="inner")
    y_true = m["is_attack"].astype(int).to_numpy()
    y_score = m["score"].to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    out = {"n": len(m), "n_attacks": int(y_true.sum()), "threshold": threshold}
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        out.update(auc=np.nan, precision=np.nan, recall=np.nan, f1=np.nan)
        return out

    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, thr = roc_curve(y_true, y_score)
    out.update(
        auc=float(roc_auc_score(y_true, y_score)),
        precision=float(pr), recall=float(rc), f1=float(f1),
        confusion=cm.tolist(),
        roc={"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()},
    )
    return out


def per_attack_recall(track_scores: pd.DataFrame, truth: pd.DataFrame,
                      threshold: float = config.DETECT.alert_threshold) -> pd.DataFrame:
    """Recall broken down by attack type - shows which signatures we catch best."""
    m = track_scores.merge(truth, on="track_id", how="inner")
    m = m[m["is_attack"]]
    if m.empty:
        return pd.DataFrame(columns=["attack_type", "n", "detected", "recall"])
    m["detected"] = m["score"] >= threshold
    g = m.groupby("attack_type").agg(n=("track_id", "size"),
                                     detected=("detected", "sum")).reset_index()
    g["recall"] = (g["detected"] / g["n"]).round(3)
    return g.sort_values("recall", ascending=False).reset_index(drop=True)
