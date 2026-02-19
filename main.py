# main.py

from data_loader import load_data
from eda import run_eda
from preprocessing import engineer_features, prepare_data
from model import (
    train_and_compare,
    detailed_report,
    save_best_model
)

def main():
    # ── Step 1: Load Data ──────────────────────────────────────
    df = load_data("AIML Dataset.csv")

    # ── Step 2: EDA ────────────────────────────────────────────
    run_eda(df)

    # ── Step 3: Feature Engineering ───────────────────────────
    df_engineered = engineer_features(df)

    # ── Step 4: Preprocessing + SMOTE ─────────────────────────
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(df_engineered)

    # ── Step 5: Train + Compare All Models ────────────────────
    results, best_name = train_and_compare(X_train, X_test, y_train, y_test)

    # ── Step 6: Detailed Report for Best Model ─────────────────
    detailed_report(results, best_name, y_test)

    # ── Step 7: Save Best Model ────────────────────────────────
    save_best_model(results, best_name, scaler, features)

    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    main()