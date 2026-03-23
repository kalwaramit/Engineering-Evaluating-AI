from model.randomforest import RandomForest
from model.SDG import SGD
from model.randomforest import RandomForest
from model.adaboost import AdaBoost
from model.hist_gb import Hist_GB
from model.random_trees_ensembling import RandomTreesEmbedding
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def model_predict(data, df, name):

    targets = [
        ("Type2", data.y2_train, data.y2_test),
        ("Type2+Type3", data.y2_y3_train, data.y2_y3_test),
        ("Type2+Type3+Type4", data.y2_y3_y4_train, data.y2_y3_y4_test),
    ]

    for label, y_train, y_test in targets:
        if y_train is None or y_test is None:
            continue

        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        train_mask = pd.notna(y_train_arr)
        test_mask = pd.notna(y_test_arr)

        X_train_curr = data.X_train[train_mask]
        y_train_curr = y_train_arr[train_mask]
        X_test_curr = data.X_test[test_mask]
        y_test_curr = y_test_arr[test_mask]

        if len(y_train_curr) == 0 or len(y_test_curr) == 0:
            print(f"\n===== {label} =====")
            print("Skipping: no non-null labels available for train/test.")
            continue

        if len(np.unique(y_train_curr)) < 2:
            print(f"\n===== {label} =====")
            print("Skipping: training target has fewer than 2 classes after cleaning.")
            continue

        print(f"\n===== {label} =====")

        model = RandomForest("RF", data.get_embeddings(), y_train)
        model.mdl.fit(X_train_curr, y_train_curr)
        predictions = model.mdl.predict(X_test_curr)
        print(classification_report(y_test_curr, predictions))


def model_evaluate(model, data):
    model.print_results(data)