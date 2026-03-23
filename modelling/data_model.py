import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        X_DL = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
        X_DL = X_DL.to_numpy()
        y = df.y.to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value)<1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        good_mask = y_series.isin(good_y_value).to_numpy()
        y_good = y[good_mask]
        X_good = X[good_mask]
        df_good = df.loc[good_mask].reset_index(drop=True)

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        new_test_size = min(max(new_test_size, 0.1), 0.5)

        indices = np.arange(len(y_good))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=new_test_size,
            random_state=0,
            stratify=y_good
        )

        self.X_train = X_good[train_idx]
        self.X_test = X_good[test_idx]
        self.y_train = y_good[train_idx]
        self.y_test = y_good[test_idx]
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X
        self.train_df = df_good.iloc[train_idx].reset_index(drop=True)
        self.test_df = df_good.iloc[test_idx].reset_index(drop=True)

        if {'y2', 'y3', 'y4'}.issubset(df_good.columns):
            self.y2 = df_good['y2'].to_numpy()
            self.y2_y3 = (
                df_good['y2'] + ' + ' + df_good['y3']
            ).to_numpy()
            self.y2_y3_y4 = (
                df_good['y2'] + ' + ' + df_good['y3'] + ' + ' + df_good['y4']
            ).to_numpy()

            self.y2_train = self.y2[train_idx]
            self.y2_test = self.y2[test_idx]
            self.y2_y3_train = self.y2_y3[train_idx]
            self.y2_y3_test = self.y2_y3[test_idx]
            self.y2_y3_y4_train = self.y2_y3_y4[train_idx]
            self.y2_y3_y4_test = self.y2_y3_y4[test_idx]
        else:
            self.y2 = None
            self.y2_y3 = None
            self.y2_y3_y4 = None
            self.y2_train = None
            self.y2_test = None
            self.y2_y3_train = None
            self.y2_y3_test = None
            self.y2_y3_y4_train = None
            self.y2_y3_y4_test = None

    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df


