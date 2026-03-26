#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from chained_controller import run_chained_controller
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
     print(name)
    X, group_df = get_embeddings(group_df)
    data = get_data_object(X, group_df)
    perform_modelling(data, group_df, name)
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)
# Code will start executing from following line
if __name__ == '__main__':
    run_chained_controller()

