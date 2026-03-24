# chained_controller.py
# Author: Sailee Mestry
# Design Choice 1: Chained Multi-Output Classification Controller
#
# This controller orchestrates the three-stage chained classification pipeline.
# Each stage trains a separate RandomForest model on the same TF-IDF features,
# classifying the following chained targets defined in Config.py:
#   Stage 1 → Type2
#   Stage 2 → Type2 + Type3
#   Stage 3 → Type2 + Type3 + Type4
#
# The Data object (from modelling/data_model.py) already prepares all three
# chained label arrays (y2_train, y2_y3_train, y2_y3_y4_train), so this
# controller simply iterates through each stage using the same X_train/X_test
# TF-IDF features and a fresh RandomForest instance per stage.

from modelling.data_model import Data
from model.randomforest import RandomForest
from preprocess import get_input_data, de_duplication, noise_remover
from embeddings import get_tfidf_embd
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def run_chained_controller():
    """
    Entry point for Design Choice 1: Chained Multi-Output Classification.

    Steps:
      1. Load and preprocess raw email data
      2. Generate TF-IDF embeddings
      3. Build a Data object (encapsulates all train/test splits and chained labels)
      4. For each of the 3 chained targets, train a RandomForest and print results
    """

    print("=" * 60)
    print("  Design Choice 1: Chained Multi-Output Classification")
    print("=" * 60)

    # Step 1: Load and preprocess data
    print("\n[1/4] Loading data...")
    df = get_input_data()

    print("[2/4] Preprocessing data...")
    df = de_duplication(df)
    df = noise_remover(df)

    # Step 2: Generate TF-IDF embeddings
    print("[3/4] Generating TF-IDF embeddings...")
    X = get_tfidf_embd(df)

    # Step 3: Build Data object — encapsulates X_train, X_test,
    #         y_train, y_test and all chained label arrays
    print("[4/4] Building data object and running chained modelling...\n")
    data = Data(X, df)

    if data.X_train is None:
        print("Not enough data to proceed. Exiting.")
        return

    # Step 4: Define the three chained stages
    # Each tuple is (stage label, y_train array, y_test array)
    chained_stages = [
        ("Stage 1 — Type2",
         data.y2_train,
         data.y2_test),

        ("Stage 2 — Type2 + Type3",
         data.y2_y3_train,
         data.y2_y3_test),

        ("Stage 3 — Type2 + Type3 + Type4",
         data.y2_y3_y4_train,
         data.y2_y3_y4_test),
    ]

    # Step 5: Train and evaluate one RandomForest per stage
    for stage_label, y_train, y_test in chained_stages:

        print(f"\n{'=' * 60}")
        print(f"  {stage_label}")
        print(f"{'=' * 60}")

        # Skip if labels not available
        if y_train is None or y_test is None:
            print("  Skipping: chained labels not available in data.")
            continue

        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        # Remove null label rows
        train_mask = pd.notna(y_train_arr)
        test_mask = pd.notna(y_test_arr)

        X_train_curr = data.X_train[train_mask]
        y_train_curr = y_train_arr[train_mask]
        X_test_curr = data.X_test[test_mask]
        y_test_curr = y_test_arr[test_mask]

        # Skip if too few samples or classes
        if len(y_train_curr) == 0 or len(y_test_curr) == 0:
            print("  Skipping: no valid labels after cleaning.")
            continue

        if len(np.unique(y_train_curr)) < 2:
            print("  Skipping: fewer than 2 classes in training labels.")
            continue

        # Instantiate a fresh RandomForest for this stage
        # Uses BaseModel interface: train(), predict(), print_results()
        model = RandomForest(
            model_name=stage_label,
            embeddings=data.get_embeddings(),
            y=y_train_curr
        )

        # Train using the consistent interface defined in BaseModel
        # We call mdl.fit directly here to use the masked subsets
        model.mdl.fit(X_train_curr, y_train_curr)

        # Predict on test set
        model.predictions = model.mdl.predict(X_test_curr)

        # Print classification report (precision, recall, F1 per class)
        print(f"\n  Classification Report:")
        print(classification_report(y_test_curr, model.predictions))


# Allow this controller to be run directly or imported into main.py
if __name__ == "__main__":
    run_chained_controller()
