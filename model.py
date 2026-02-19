# src/model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

os.makedirs("outputs/plots", exist_ok=True)


# ─────────────────────────────────────────────
# 1. DEFINE ALL MODELS
# ─────────────────────────────────────────────

def get_models() -> dict:
    """
    Returns a dictionary of models to compare.
    Each model is tuned appropriately for fraud detection.
    """
    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,       # number of trees
            max_depth=10,           # prevent overfitting
            class_weight='balanced',
            random_state=42,
            n_jobs=-1               # use all CPU cores
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=99,    # handles imbalance:
                                    # approx ratio of negatives to positives
            eval_metric='logloss',  # use_label_encoder removed (deprecated)
            random_state=42,
            n_jobs=-1
        )

    }
    return models


# ─────────────────────────────────────────────
# 2. TRAIN AND EVALUATE ALL MODELS
# ─────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains all models, evaluates each, prints comparison table,
    and returns the best model based on F1 score for fraud class.
    """
    models = get_models()
    results = {}

    print("\n" + "="*60)
    print("TRAINING AND EVALUATING ALL MODELS")
    print("="*60)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Train and time it
        start = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_prob)
        avg_prec  = average_precision_score(y_test, y_prob)

        fraud_detected = int(((y_pred == 1) & (y_test == 1)).sum())
        total_fraud    = int((y_test == 1).sum())
        false_neg      = total_fraud - fraud_detected

        print(f"  Training Time    : {train_time}s")
        print(f"  Accuracy         : {accuracy:.4f}")
        print(f"  Precision        : {precision:.4f}")
        print(f"  Recall           : {recall:.4f}")
        print(f"  F1 Score         : {f1:.4f}")
        print(f"  ROC-AUC          : {roc_auc:.4f}")
        print(f"  Avg Precision    : {avg_prec:.4f}")
        print(f"  Fraud Detected   : {fraud_detected} / {total_fraud}")
        print(f"  Missed Fraud     : {false_neg}")

        results[name] = {
            "model"          : model,
            "y_pred"         : y_pred,
            "y_prob"         : y_prob,
            "accuracy"       : accuracy,
            "precision"      : precision,
            "recall"         : recall,
            "f1"             : f1,
            "roc_auc"        : roc_auc,
            "avg_precision"  : avg_prec,
            "fraud_detected" : fraud_detected,
            "total_fraud"    : total_fraud,
            "false_negatives": false_neg,
            "train_time"     : train_time
        }

    # Print comparison table
    print_comparison_table(results)

    # Generate all plots
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_metric_comparison(results)

    # Feature importance for tree-based models
    plot_feature_importance(results, X_train)

    # Pick best model by F1 score
    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\n✅ Best Model by F1 Score: {best_name}")
    print(f"   F1 = {results[best_name]['f1']:.4f}")

    return results, best_name


# ─────────────────────────────────────────────
# 3. COMPARISON TABLE
# ─────────────────────────────────────────────

def print_comparison_table(results: dict):
    print("\n" + "="*75)
    print("MODEL COMPARISON SUMMARY")
    print("="*75)
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-"*75)

    for name, r in results.items():
        print(
            f"{name:<25} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['roc_auc']:>9.4f}"
        )
    print("="*75)


# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────

def plot_roc_curves(results: dict, y_test):
    """
    All models ROC curves on one plot for easy comparison.
    """
    colors = ['tomato', 'steelblue', 'seagreen']
    plt.figure(figsize=(8, 6))

    for (name, r), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        plt.plot(fpr, tpr, color=color,
                 label=f"{name} (AUC={r['roc_auc']:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/roc_curves_comparison.png")
    plt.close()
    print("\nSaved: roc_curves_comparison.png")


def plot_precision_recall_curves(results: dict, y_test):
    """
    Precision-Recall curves — more informative than ROC for imbalanced data.
    """
    colors = ['tomato', 'steelblue', 'seagreen']
    plt.figure(figsize=(8, 6))

    for (name, r), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, r['y_prob'])
        plt.plot(recall, precision, color=color,
                 label=f"{name} (AP={r['avg_precision']:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/precision_recall_comparison.png")
    plt.close()
    print("Saved: precision_recall_comparison.png")


def plot_confusion_matrices(results: dict, y_test):
    """
    Side by side confusion matrices for all models.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud']
        )
        ax.set_title(f"{name}\nF1={r['f1']:.4f}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrices_comparison.png")
    plt.close()
    print("Saved: confusion_matrices_comparison.png")


def plot_metric_comparison(results: dict):
    """
    Bar chart comparing key metrics across models.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25
    colors = ['tomato', 'steelblue', 'seagreen']

    plt.figure(figsize=(12, 6))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [results[name][m] for m in metrics]
        plt.bar(x + i * width, values, width, label=name, color=color, alpha=0.85)

    plt.xticks(x + width, [m.replace('_', ' ').title() for m in metrics])
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Model Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/metric_comparison.png")
    plt.close()
    print("Saved: metric_comparison.png")


def plot_feature_importance(results: dict, X_train):
    """
    Feature importance for Random Forest and XGBoost.
    """
    feature_names = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'type_encoded',
        'error_balance_orig', 'error_balance_dest',
        'orig_drained', 'amount_ratio'
    ]

    tree_models = {
        k: v for k, v in results.items()
        if k in ["Random Forest", "XGBoost"]
    }

    for name, r in tree_models.items():
        importances = r['model'].feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 5))
        plt.bar(
            range(len(importances)),
            importances[indices],
            color='steelblue', alpha=0.85
        )
        plt.xticks(
            range(len(importances)),
            [feature_names[i] for i in indices],
            rotation=45, ha='right'
        )
        plt.title(f"Feature Importance — {name}")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_")
        plt.savefig(f"outputs/plots/feature_importance_{filename}.png")
        plt.close()
        print(f"Saved: feature_importance_{filename}.png")


# ─────────────────────────────────────────────
# 5. DETAILED REPORT FOR BEST MODEL
# ─────────────────────────────────────────────

def detailed_report(results: dict, best_name: str, y_test):
    """
    Print full classification report for the best model.
    """
    r = results[best_name]
    print(f"\n{'='*60}")
    print(f"DETAILED REPORT — {best_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_test, r['y_pred'],
        target_names=['Legit', 'Fraud']
    ))


# ─────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_best_model(results: dict, best_name: str, scaler, features: list):
    """
    Save the best model along with scaler and feature list.
    """
    best_model = results[best_name]['model']

    joblib.dump({
        'model'     : best_model,
        'model_name': best_name,
        'scaler'    : scaler,
        'features'  : features
    }, "outputs/model.pkl")

    print(f"\nBest model ({best_name}) saved to outputs/model.pkl")