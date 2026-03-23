import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

def train_custom(self, X_train, y_train):
    self.model.fit(X_train, y_train)

def print_results_custom(self, y_test):
    from sklearn.metrics import accuracy_score
    print("Accuracy:", accuracy_score(y_test, self.predictions))
