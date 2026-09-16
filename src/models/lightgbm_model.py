from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


FEATURE_COLS_EXCLUDE = {"object_id", "label"}


def prepare_xy(feat_df: pd.DataFrame, label_encoder: LabelEncoder | None = None):
    feature_cols = [c for c in feat_df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = feat_df[feature_cols].values
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(feat_df["label"].values)
    else:
        y = label_encoder.transform(feat_df["label"].values)
    return X, y, feature_cols, label_encoder


def train_lightgbm(X_train, y_train, X_val, y_val, num_class: int, params: dict | None = None,
                    class_weight: str | None = None):
    """
    class_weight: None (default, every sample weighted equally) or
    "balanced" (samples weighted inversely proportional to their class's
    frequency in y_train, via sklearn's standard convention -- same
    effect as sklearn's class_weight="balanced"). Used by
    ablation_class_weight.py to test whether weighting matters for this
    dataset's actual class balance.
    """
    default_params = dict(
        objective="multiclass",
        num_class=num_class,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=10,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=42,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    sample_weight = None
    if class_weight == "balanced":
        classes, counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        weight_per_class = {c: n_samples / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_per_class[label] for label in y_train])
    elif class_weight is not None:
        raise ValueError(f"Unknown class_weight: {class_weight!r} (use None or 'balanced')")

    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def evaluate_lightgbm(model, X_test, y_test, label_encoder: LabelEncoder):
    proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(proba, axis=1)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return {"proba": proba, "y_pred": y_pred, "report": report, "confusion_matrix": cm}


if __name__ == "__main__":
    print("Import and call train_lightgbm(...) / evaluate_lightgbm(...) from your notebook.")
