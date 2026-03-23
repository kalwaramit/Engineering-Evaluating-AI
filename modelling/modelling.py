from model.randomforest import RandomForest
from model.SGD import SGD
from model.randomforest import RandomForest
from model.adaboost import AdaBoost
from model.voting import Voting
from model.hist_gb import Hist_GB
from model.random_trees_ensembling import RandomTreesEmbedding


def model_predict(data, df, name):

    targets = [
        ("Type2", data.y2_train, data.y2_test),
        ("Type2+Type3", data.y2_y3_train, data.y2_y3_test),
        ("Type2+Type3+Type4", data.y2_y3_y4_train, data.y2_y3_y4_test),
    ]

    for label, y_train, y_test in targets:

        print(f"\n===== {label} =====")

        model = RandomForest("RF", data.get_embeddings(), y_train)

        model.train_custom(data.X_train, y_train)
        model.predict(data.X_test)
        model.print_results_custom(y_test)


def model_evaluate(model, data):
    model.print_results(data)